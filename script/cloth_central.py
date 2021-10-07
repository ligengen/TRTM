import numpy as np
import bpy
from mathutils import Vector
import pickle
import bmesh
import glob
import os
import time
import sys


# generate Plane, Hand, Cloth Objects
def generate_models():
    # set simulation parameters
    cloth_division = 80
    baseZ = 0.05

    # delect old objects
    bpy.ops.object.select_all(action='DESELECT')
    for object in bpy.context.scene.objects:
        object.select_set(True)
    bpy.ops.object.delete()

    # add Plane object with Collision Modifier
    bpy.ops.mesh.primitive_plane_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.context.object.scale = [4, 4, 1]
    bpy.context.object.collision.cloth_friction = 100
    bpy.ops.object.modifier_add(type='COLLISION')

    # add Cloth Material
    MaterialCloth = bpy.data.materials.new('MaterialCloth')
    MaterialCloth.diffuse_color = (0.1, 0.8, 0.8, 1)

    # add Cloth object with Cloth Material
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=cloth_division, y_subdivisions=cloth_division, enter_editmode=False, align='WORLD',
                                    location=(0, 0, baseZ), scale=(1, 1, 1))
    bpy.context.object.name = 'Cloth'
    bpy.context.object.active_material = MaterialCloth
    # add VERTEX_WEIGHT_MIX, HOOK and CLOTH Modifiers
    bpy.ops.object.modifier_add(type='VERTEX_WEIGHT_MIX')
    bpy.ops.object.modifier_add(type='HOOK')
    bpy.context.object.modifiers['Hook'].name = "Hook1"
    bpy.ops.object.modifier_add(type='HOOK')
    bpy.context.object.modifiers['Hook'].name = "Hook2"
    bpy.ops.object.modifier_add(type='CLOTH')
    bpy.ops.object.shade_smooth()
    # set Cloth parameters
    bpy.context.object.modifiers['Cloth'].settings.bending_stiffness = 50
    bpy.context.object.modifiers['Cloth'].collision_settings.use_self_collision = True
    bpy.context.object.modifiers['Cloth'].collision_settings.self_friction = 80
    bpy.context.object.modifiers['Cloth'].collision_settings.self_distance_min = 0.005
    # add Camera and Sun object
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 5), rotation=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 3), scale=(1, 1, 1))
    bpy.context.scene.render.resolution_x = 1080
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.camera = bpy.data.objects['Camera']



## data manipulation functions
# convert Blender Mesh Vertices Coordinate to Numpy Array: np_mesh = convert_mesh_to_np(cloth.data)
def convert_mesh_to_np(mesh):
    n_vtx = int(len(mesh.vertices))
    np_mesh = np.empty((n_vtx, 3))
    for i in range(n_vtx):
        np_mesh[i] = np.array(mesh.vertices[i].co)
    print(n_vtx, ' mesh converted to array of shape', np_mesh.shape)
    return np_mesh

# save state information to state.pickle or init.pickle
def save_state(ms, meshes, vtx_picks, co_picks, moves, base_dir, convert=False, as_init=False):
    # convert Blender Meshes to Numpy Array: np_meshes[ms+1, n, 3]
    if convert:
        for mesh in meshes: print('vertices:', len(mesh.vertices))
        np_meshes = np.array([[[v.co[0], v.co[1], v.co[2]] for v in mesh.vertices] for mesh in meshes])
    else:
        np_meshes = meshes
    # construct state dictionary
    state = {'ms': ms,
             'meshes': np_meshes,
             'vtx_picks': vtx_picks,
             'co_picks': co_picks,
             'moves': moves}
    # save state as init.pickle
    if as_init:
        with open(base_dir + 'init.pickle', mode='wb') as f:
            pickle.dump(state, f)
    # save state as state.pickle
    with open(base_dir + 'state[pid' + str(os.getpid()) + '].pickle', mode='wb') as f:
        pickle.dump(state, f)

# save current cloth as init.pickle
def save_current_cloth_as_init(base_dir):
    print('saving current state as initial state...')
    # load current Cloth Mesh
    cloth = bpy.data.objects['Cloth']
    # convert current Cloth Mesh to 3D Numpy Array
    save_state(0, convert_mesh_to_np(cloth.data)[None, ...], [], [], [], base_dir, as_init=True)
    print('ok!')

# load pickle file
def load_pickle_file(dir):
    state = pickle.load(open(dir, mode='rb'))
    print('state: ', state)
    print('shape of meshes: ', state['meshes'].shape)

# save pickle file to csv file
def save_pickle_to_csv(dir):
    state = pickle.load(open(dir, mode='rb'))
    meshes = state['meshes']
    print('meshes.shape:', meshes.shape)
    np.savetxt(base_dir + 'data.csv', meshes[0], delimiter=',')

# load and return meshes from init.pickle
def load_init_state_meshes(base_dir):
    try:
        state = pickle.load(open(base_dir + 'init.pickle', mode='rb'))
        meshes = state['meshes']
        return meshes
    except IOError:
        return np.array([])

# load and return state from state.pickle
def load_state(base_dir):
    try:
        state = pickle.load(open(base_dir + 'state[pid' + str(os.getpid()) + '].pickle', mode='rb'))
        ms = state['ms']
        meshes = state['meshes']
        vtx_picks = state['vtx_picks']
        co_picks = state['co_picks']
        moves = state['moves']
        return ms, meshes, vtx_picks, co_picks, moves
    # detect IOError
    except IOError:
        return 0, np.array([]), [], [], []

# assign numpy mesh data to cloth.data.vertices
def assign(cloth, np_mesh):
    #    print('assign np_mesh with shape', np_mesh.shape, 'to cloth.data.vertices of size', len(cloth.data.vertices))
    for i in range(len(cloth.data.vertices)):
        for c in range(3):
            cloth.data.vertices[i].co[c] = np_mesh[i, c]

# centralize numpy mesh to (x, y) mean center
def centralize_np_mesh(np_mesh):
    center = np.mean(np_mesh, axis=0)
    print('cloth center before centralization:', center)
    center[2] = 0
    for i in range(np_mesh.shape[0]):
        np_mesh[i] = np_mesh[i] - center
    print('cloth center after centralization:', np.mean(np_mesh, axis=0))
    return np_mesh


# reset cloth simulation state with init.pickle and save into state.pickle
def cloth_simulation_reset(base_dir):
    print('cloth simulation initializing...')
    # initialize simulation parameters
    baseZ = 0.05

    # initialize cloth object
    cloth = bpy.data.objects['Cloth']
    cloth.location = (0, 0, baseZ)
    # clear cloth simulation data
    cloth.animation_data_clear()
    cloth.vertex_groups.clear()
    # assign cloth to active object
    bpy.context.view_layer.objects.active = cloth

    # load meshes data from init.pickle
    meshes = load_init_state_meshes(base_dir)
    # assign init np_meshes vertices data to cloth.data.vertices
    assign(cloth, centralize_np_mesh(meshes[0]))
    # assign cloth data back to init np_meshes
    meshes = np.array([[[v.co[0], v.co[1], v.co[2]] for v in cloth.data.vertices]])

    # delete Hand objects
    bpy.ops.object.select_all(action='DESELECT')
    for object in bpy.context.scene.objects:
        if object.name == 'Hand1':
            object.select_set(True)
        if object.name == 'Hand2':
            object.select_set(True)
    bpy.ops.object.delete()

    # initialize frame
    action_sequence = 1
    bpy.context.scene.frame_set(0)

    # initialize state information
    ms = 0
    vtx_picks = []
    co_picks = np.random.uniform(-1.0, 1.0, (action_sequence, 2, 2))
    moves = np.random.uniform(-1.0, 1.0, (action_sequence, 2))
    # save initial state information into newest state.pickle
    save_state(ms, meshes, vtx_picks, co_picks, moves, base_dir)
    print('cloth simulation initialzed...\n')


# simulate Cloth forward with drag manipulations
def cloth_simulation_drag_step(base_dir, as_init=False):
    # initialize simulation parameters
    baseZ = 0.05

    # initialize cloth object
    cloth = bpy.data.objects['Cloth']
    # clear animation for cloth
    cloth.animation_data_clear()
    cloth.vertex_groups.clear()
    # assign cloth to active object
    bpy.context.view_layer.objects.active = cloth

    # load state information from newest state.pickle
    ms, meshes, vtx_picks, co_picks, fixed_moves = load_state(base_dir)
    move = fixed_moves[ms]
    # update cloth.data.vertices with the newest mesh data
    assign(cloth, meshes[ms])

#    # assign movement information
#    # assign hand1 pick position
#    co_picks[ms, 0] = np.array([0.5, 0.5])
#    # assign hand2 pick position
#    co_picks[ms, 1] = np.array([-0.5, -0.5])
#    # assign move vector
#    move = np.array([[0, 0], [0, 0]])
#    # assign lift height
#    lift_height1 = 0.8
#    lift_height2 = 0.8
#
#    # assign movement information
#    # assign hand1 pick position
#    co_picks[ms, 0] = np.array([0, 0])
#    # assign hand2 pick position
#    co_picks[ms, 1] = np.array([0, -0])
#    # assign move vector
#    move = np.array([[0, 0], [0, 0]])
#    # assign lift height
#    lift_height1 = 1
#    lift_height2 = 1

    # assign movement information
    # assign hand1 pick position
    co_picks[ms, 0] = np.array([0.3, -0.3])
    # assign hand2 pick position
    co_picks[ms, 1] = np.array([-0.3, 0.3])
    # assign move vector
    move = np.array([[0.6, -0.6], [-0.6, 0.6]])
    # assign lift height
    lift_height1 = 0.2
    lift_height2 = 0.2

    # find pick vertices from pick positions: co_picks
    pick_list1 = []
    pick_list2 = []
    pick_list = []
    vtxs = meshes[ms]
    grasp_range = 0.05
    for i in range(vtxs.shape[0]):
        # get distance between vertex and pick point
        d1 = ((vtxs[i, 0] - co_picks[ms][0][0]) ** 2 + (vtxs[i, 1] - co_picks[ms][0][1]) ** 2) ** 0.5
        d2 = ((vtxs[i, 0] - co_picks[ms][1][0]) ** 2 + (vtxs[i, 1] - co_picks[ms][1][1]) ** 2) ** 0.5
        if d1 < grasp_range:
            pick_list1.append(i)
        if d2 < grasp_range and d1 >= grasp_range:
            pick_list2.append(i)
    # pick top point from pick_list1
    if pick_list1 == []:
        print('pick_list1 is empty!')
    else:
        pick_point1 = pick_list1[0]
        for i in pick_list1:
            if vtxs[i, 2] > vtxs[pick_point1, 2]:
                pick_point1 = i
        temp_list = pick_list1
        pick_list1 = []
        for i in temp_list:
            if i in [pick_point1, pick_point1+1, pick_point1-1, pick_point1+100, pick_point1-100]:
                pick_list1.append(i)
                pick_list.append(i)
    # pick top point from pick_list2
    if pick_list2 == []:
        print('pick_list2 is empty!')
    else:
        pick_point2 = pick_list2[0]
        for i in pick_list2:
            if vtxs[i, 2] > vtxs[pick_point2, 2]:
                pick_point2 = i
        temp_list = pick_list2
        pick_list2 = []
        for i in temp_list:
            if i in [pick_point2, pick_point2+1, pick_point2-1, pick_point2+100, pick_point2-100]:
                pick_list2.append(i)
                pick_list.append(i)

    print('pick_list1 vertices:', pick_list1)
    print('pick_list2 vertices:', pick_list2)
    print('pick_list vertices:', pick_list)

    # samulate forward
    # stablize initial Cloth
    frame_num = 0
    # wait for stablization
    frame_num += 20

    # initialize vertex groups
    empty = cloth.vertex_groups.new(name='empty')
    pick1 = cloth.vertex_groups.new(name='pick1')
    pick2 = cloth.vertex_groups.new(name='pick2')
    pick = cloth.vertex_groups.new(name='pick')
    # add vertex group, weight paint the pick-up vertices
    pick1.add(pick_list1, 1.0, 'REPLACE')
    pick2.add(pick_list2, 1.0, 'REPLACE')
    pick.add(pick_list, 1.0, 'REPLACE')
    # assign vertex groups to Hook
    cloth.modifiers['Hook1'].vertex_group = 'pick1'
    cloth.modifiers['Hook2'].vertex_group = 'pick2'

    # set cloth modifiers
    cloth.modifiers["Cloth"].settings.vertex_group_mass = 'empty'
    cloth.modifiers["VertexWeightMix"].vertex_group_a = 'empty'
    cloth.modifiers["VertexWeightMix"].vertex_group_b = 'pick'
    cloth.modifiers["VertexWeightMix"].mix_mode = 'ADD'  # should be set already
    cloth.modifiers["VertexWeightMix"].mix_set = 'OR'  # should be set already
    frame_num += 1

    # computing movement trajectory
    # get initial hand pick positions
    x10 = co_picks[ms][0][0]
    x1 = x10
    y10 = co_picks[ms][0][1]
    y1 = y10
    z1 = baseZ
    x20 = co_picks[ms][1][0]
    x2 = x20
    y20 = co_picks[ms][1][1]
    y2 = y20
    z2 = baseZ

    # add Hand1 and Hand2 as Empty Object
    bpy.ops.object.empty_add(type='SINGLE_ARROW', align='WORLD', location=(x10, y10, baseZ), scale=(1, 1, 1))
    bpy.context.object.name = 'Hand1'
    bpy.context.object.scale = [1, 1, 0.2]
    bpy.ops.object.empty_add(type='SINGLE_ARROW', align='WORLD', location=(x20, y20, baseZ), scale=(1, 1, 1))
    bpy.context.object.name = 'Hand2'
    bpy.context.object.scale = [1, 1, 0.2]
    # assign Hand1 and Hand2 to Hook1 and Hook2
    cloth.modifiers["Hook1"].object = bpy.data.objects["Hand1"]
    cloth.modifiers["Hook2"].object = bpy.data.objects["Hand2"]
    # clear animation for hand1 and hand2
    hand1 = bpy.data.objects['Hand1']
    hand1.animation_data_clear()
    hand2 = bpy.data.objects['Hand2']
    hand2.animation_data_clear()
    # initialize Hand and Cloth simulation
    hand1.location = (x10, y10, baseZ)
    hand1.keyframe_insert(data_path="location", frame=frame_num)
    hand2.location = (x20, y20, baseZ)
    hand2.keyframe_insert(data_path="location", frame=frame_num)
    frame_num += 1

    # divide movement into smaller steps
    xy_step = 0.04
    z_step = 0.04
    # get move distance and move normal
    move_length1 = (move[0][0] ** 2 + move[0][1] ** 2) ** .5
    if move_length1 == 0:
        move_normal1 = np.array([1, 0])
    else:
        move_normal1 = move[0] / move_length1
    move_length2 = (move[1][0] ** 2 + move[1][1] ** 2) ** .5
    if move_length2 == 0:
        move_normal2 = np.array([1, 0])
    else:
        move_normal2 = move[1] / move_length2

    # simulate lift move action
    for i in range(0, 400):
        # iterative z1 hand positions with smaller steps
        z1 = min(z1 + z_step, baseZ + lift_height1)
        hand1.location = (x1, y1, z1)
        # insert hand keyframe
        hand1.keyframe_insert(data_path="location", frame=frame_num)
        # iterative z2 hand positions with smaller steps
        z2 = min(z2 + z_step, baseZ + lift_height2)
        hand2.location = (x2, y2, z2)
        # insert hand keyframe
        hand2.keyframe_insert(data_path="location", frame=frame_num)
        frame_num += 1
        if z1 == baseZ + lift_height1 and z2 == baseZ + lift_height2: break
    frame_num += 20

    # simulate horizontal move action
    for i in range(0, 400):
        # iterative x1, y1 hand positions with smaller steps
        x1 = x10 + np.sign(move[0][0]) * min(abs(x1 - x10 + move_normal1[0] * xy_step), abs(move[0][0]))
        y1 = y10 + np.sign(move[0][1]) * min(abs(y1 - y10 + move_normal1[1] * xy_step), abs(move[0][1]))
        hand1.location = (x1, y1, z1)
        # insert hand keyframe
        hand1.keyframe_insert(data_path="location", frame=frame_num)
        # iterative x2, y2 hand positions with smaller steps
        x2 = x20 + np.sign(move[1][0]) * min(abs(x2 - x20 + move_normal2[0] * xy_step), abs(move[1][0]))
        y2 = y20 + np.sign(move[1][1]) * min(abs(y2 - y20 + move_normal2[1] * xy_step), abs(move[1][1]))
        hand2.location = (x2, y2, z2)
        # insert hand keyframe
        hand2.keyframe_insert(data_path="location", frame=frame_num)
        frame_num += 1
        if x1 == x10 + move[0][0] and y1 == y10 + move[0][1] and x2 == x20 + move[1][0] and y2 == y20 + move[1][1]: break
    frame_num += 5

    # release pin points
    frame_num += 1
    cloth.modifiers["VertexWeightMix"].mask_constant = 1
    cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant', frame=frame_num)
    frame_num += 1
    release_at = frame_num
    cloth.modifiers["VertexWeightMix"].mask_constant = 0
    cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant', frame=frame_num)
    frame_num += 1

    # wait for stablization
    frame_num += 20

    # initialize hand height
    hand1.location = (x1, y1, baseZ)
    hand1.keyframe_insert(data_path="location", frame=frame_num)
    hand2.location = (x2, y2, baseZ)
    hand2.keyframe_insert(data_path="location", frame=frame_num)
    frame_num += 1

    # initialize start time
    t_start = time.time()
    # simulation frame forward
    for i in range(frame_num + 1):
        t_frame = time.time()
        # set scene frame
        bpy.context.scene.frame_set(i)
        print('play', i, '/', frame_num, 'time:', time.time() - t_frame)
        # take top-view pictures at start and end frame
        camera = bpy.data.objects['Camera']
        sun = bpy.data.objects['Sun']
        sun.data.energy = 1.5
        if i == 0:
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.context.scene.render.filepath = base_dir + 'start.png'
            bpy.ops.render.render(write_still = 1)
        if i == frame_num:
#            camera.location = (1, 1, 5)
#            sun.location = (1, 1, 6)
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.context.scene.render.filepath = base_dir + 'end.png'
            bpy.ops.render.render(write_still = 1)

    # set end frame
    bpy.data.scenes["Scene"].frame_end = frame_num
    print('manipulation time:', time.time() - t_start)
    # deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # get final Cloth mesh
    result_mesh = convert_mesh_to_np(cloth.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh())
    meshes = result_mesh.reshape(1, result_mesh.shape[0], result_mesh.shape[1])
    # save final Cloth mesh into init.pickle
    if as_init:
        save_state(ms,meshes,vtx_picks,co_picks,fixed_moves, base_dir, convert=False, as_init=True)


base_dir = '/Users/aaronw/Desktop/Blender/Week2/cloth_central/'
generate_models()
#save_current_cloth_as_init(base_dir)

cloth_simulation_reset(base_dir)
cloth_simulation_drag_step(base_dir, as_init=False)

#save_pickle_to_csv(base_dir+'init.pickle')

