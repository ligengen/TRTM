import numpy as np
import bpy
from mathutils import Vector
import pickle
import bmesh
import glob
import os
import time
import sys


# generate Plane, Hand, Cloth Model
def genrate_models():
    # add Plane with Collision Modifier
    bpy.ops.mesh.primitive_plane_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.context.object.scale = [4, 4, 1]
    bpy.context.object.collision.cloth_friction = 50
    bpy.ops.object.modifier_add(type='COLLISION')

    # add Hand with Empty Cube
    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0.05), scale=(1, 1, 1))
    bpy.context.object.name = 'Hand'

    # add Cloth Material
    MaterialCloth = bpy.data.materials.new('MaterialCloth')
    MaterialCloth.diffuse_color = (0.1, 0.8, 0.8, 1)

    # add Cloth with Cloth Modifier
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=80, y_subdivisions=80, enter_editmode=False, align='WORLD',
                                    location=(0, 0, 0), scale=(1, 1, 1))
    bpy.context.object.name = 'Cloth'
    bpy.context.object.active_material = MaterialCloth
    bpy.context.object.parent = bpy.data.objects['Hand']
    bpy.ops.object.modifier_add(type='VERTEX_WEIGHT_MIX')
    bpy.ops.object.modifier_add(type='CLOTH')
    bpy.ops.object.shade_smooth()
    bpy.context.object.modifiers["Cloth"].settings.bending_stiffness = 20
    bpy.context.object.modifiers["Cloth"].collision_settings.self_friction = 80
    bpy.context.object.modifiers["Cloth"].collision_settings.self_distance_min = 0.005




# convert Blender Mesh Vertices Coordinate to Numpy Array
def convert_mesh_to_np(mesh):
    n_vtx = int(len(mesh.vertices))
    np_mesh = np.empty((n_vtx, 3))
    for i in range(n_vtx):
        np_mesh[i] = np.array(mesh.vertices[i].co)
    print(n_vtx, ' mesh converted to array of shape', np_mesh.shape)
    return np_mesh


# save state information to state.pickle or init.pickle
def save_state(ms, meshes, vtx_picks, co_picks, moves, base_dir, convert=False, as_init=False):
    # convert Blender Meshes to Numpy Array
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


def load_pickle_file(dir):
    state = pickle.load(open(dir, mode='rb'))
    print('state: ', state)
    print('shape of meshes: ', state['meshes'].shape)


def load_init_state_meshes(base_dir):
    try:
        state = pickle.load(open(base_dir + 'init.pickle', mode='rb'))
        meshes = state['meshes']
        return meshes
    except IOError:
        return np.array([])


# load and return state information from state.pickle
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
    print('assign np_mesh with shape', np_mesh.shape, 'to cloth.data.vertices of size', len(cloth.data.vertices))
    for i in range(len(cloth.data.vertices)):
        for c in range(3):
            cloth.data.vertices[i].co[c] = np_mesh[i, c]


# initialize cloth simulation in state.pickle
def cloth_simulation_init(base_dir):
    print('cloth simulation initializing...')

    # initialize cloth object
    cloth = bpy.data.objects['Cloth']
    cloth.location = (0, 0, 0.05)
    # clear cloth object data
    cloth.animation_data_clear()
    cloth.vertex_groups.clear()
    # assign cloth to active object
    bpy.context.view_layer.objects.active = cloth

    # load meshes data from init.pickle
    meshes = load_init_state_meshes(base_dir)
    # assign an np_mesh vertices data to cloth.data.vertices
    assign(cloth, meshes[0])
    # assign cloth data back to init meshes
    meshes = np.array([[[v.co[0], v.co[1], v.co[2]] for v in cloth.data.vertices]])

    # initialize hand object
    hand = bpy.data.objects['Hand']
    hand.animation_data_clear()
    hand.location = (0, 0, 0.05)
    hand.rotation_euler = (0, 0, 0)

    # initialize frame
    action_sequence = 1
    frames_per_manipulation = 100
    bpy.context.scene.frame_set(0)

    # initialize state information
    ms = 0
    vtx_picks = []
    co_picks = np.random.uniform(-1.0, 1.0, (action_sequence, 2, 2))
    moves = np.random.uniform(-1.0, 1.0, (action_sequence, 2))

    #    co_picks = np.random.uniform(-1.0, 1.0, (sequence_length, 2, 2))
    #    move_dists = np.random.uniform(0.0, context.scene.max_movement, [sequence_length])
    #    move_angles = np.random.uniform(0, 360, [sequence_length])
    #    print('move distances:', move_dists)
    #    print('move angles:', move_angles)
    #    moves = np.stack([move_dists * np.cos(move_angles), move_dists * np.sin(move_angles)], 1)
    #    print('saving moves:', moves)

    # save initial state information in newest state.pickle
    save_state(ms, meshes, vtx_picks, co_picks, moves, base_dir)


# cloth_simulation_init(base_dir)

def cloth_simulation_step(base_dir):
    # initialize simulation parameters
    baseZ = 0.05
    wait_time = 0
    move_time = 15

    # get cloth object
    cloth = bpy.data.objects['Cloth']
    # clear animation for cloth and hand
    cloth.animation_data_clear()
    cloth.vertex_groups.clear()
    bpy.context.view_layer.objects.active = cloth
    hand = bpy.data.objects['Hand']
    hand.animation_data_clear()

    # load state information from newest state.pickle
    ms, meshes, vtx_picks, co_picks, fixed_moves = load_state(base_dir)
    move = fixed_moves[ms]
    print('loading current state information...')
    print('running ms:', ms)
    print('meshes.shape:', meshes.shape)
    print('vtx_picks:', vtx_picks)
    print('co_picks:', co_picks)
    print('fixed_moves.shape:', fixed_moves)
    print('move:', move)

    # update cloth.data.vertices with the newest mesh data
    assign(cloth, meshes[ms])

    # assign movement information
    # assign left hand pick position
    co_picks[ms, 0] = np.array([1, 1])
    # assign right hand pick position
    co_picks[ms, 1] = np.array([1, -1])
    # assign move vector
    move = np.array([-1, 0])
    print('picking coordinates:', co_picks[ms])
    print('move vector:', move)

    # find pick vertices from pick positions
    pick_list = []
    vtxs = meshes[ms]
    for i in range(vtxs.shape[0]):
        for p in range(2):
            # get distance between vertex and pick point
            d = ((vtxs[i, 0] - co_picks[ms][p][0]) ** 2 + (vtxs[i, 1] - co_picks[ms][p][1]) ** 2) ** 0.5
            if d < 0.05: pick_list.append(i)
    print('pick vertices:', pick_list)

    # step forward
    if pick_list == []:
        result_mesh = vtxs
    else:
        # initialize vertex groups
        vg = cloth.vertex_groups.new(name='vg')
        vg_add = cloth.vertex_groups.new(name='vg_add')

        # add vertex group, weight paint the pick-up vertices
        vg_add.add(pick_list, 1.0, 'REPLACE')

        # set cloth modifiers
        simulation_type = ['cloth', 'softbody'][0]
        if simulation_type == 'cloth':
            cloth.modifiers["Cloth"].settings.vertex_group_mass = 'vg'
        else:
            cloth.modifiers["Softbody"].settings.vertex_group_goal = 'vg_add'
        cloth.modifiers["VertexWeightMix"].vertex_group_a = 'vg'
        cloth.modifiers["VertexWeightMix"].vertex_group_b = 'vg_add'
        cloth.modifiers["VertexWeightMix"].mix_mode = 'ADD'  # should be set already
        cloth.modifiers["VertexWeightMix"].mix_set = 'OR'  # should be set already

        # start simulation from frame number 0
        frame_num = 0
        hand.location = (0, 0, baseZ)
        hand.keyframe_insert(data_path="location", frame=frame_num)
        cloth.modifiers["VertexWeightMix"].mask_constant = 0
        cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant', frame=frame_num)

        frame_num += 1
        cloth.modifiers["VertexWeightMix"].mask_constant = 1
        cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant', frame=frame_num)

        # wait time
        frame_num += 10

        # computing movement trajectory
        # get initial origin
        x = 0
        y = 0
        z = baseZ
        # divide movement into smaller steps
        xy_step = 0.02
        z_step = 0.02
        lift_height = 0.15
        # get move distance and move normal
        move_length = (move[0] ** 2 + move[1] ** 2) ** .5
        move_normal = move / move_length

        # forward simulation for 100 steps until finish move
        for i in range(0, 100):
            #            print('move', move)
            # iterative x, y, z pick positions with smaller steps
            x = np.sign(move[0]) * min(abs(x + move_normal[0] * xy_step), abs(move[0]))
            y = np.sign(move[1]) * min(abs(y + move_normal[1] * xy_step), abs(move[1]))
            z = min(z + z_step, baseZ + lift_height)
            print('hand to', x, y, z, 'move:', move)
            hand.location = (x, y, z)

            # insert hand keyframe
            hand.keyframe_insert(data_path="location", frame=frame_num)
            frame_num += 1
            # print('key',frame_num)
            if x == move[0] and y == move[1]: break
            # if i == 99: print('movement taking too many frames: aborting...')

        cloth.modifiers["VertexWeightMix"].mask_constant = 1
        cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant', frame=frame_num)

        frame_num += 1
        release_at = frame_num

        cloth.modifiers["VertexWeightMix"].mask_constant = 0
        cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant', frame=frame_num)

        frame_num += 1
        hand.location = (x, y, baseZ)
        hand.keyframe_insert(data_path="location", frame=frame_num)

        # waiting for final stabilisation
        frame_num += 20

        # initialize start time
        t_start = time.time()
        # simulation frame forward
        for i in range(frame_num + 1):
            t_frame = time.time()
            # set scene frame
            bpy.context.scene.frame_set(i)
            print('play', i, '/', frame_num, 'time:', time.time() - t_frame)
            # bpy.context.scene.update()

            if i == release_at:
                print('release')

        bpy.data.scenes["Scene"].frame_end = frame_num

        print('manipulation time:', time.time() - t_start)

        # make snapshot of resulting mesh


#        result_mesh = convert_to_np(cloth.to_mesh(scene=bpy.context.scene, apply_modifiers=True, settings='RENDER'))


base_dir = '/Users/aaronw/Desktop/Blender/Week2/Milestone1_Grasp/'
#genrate_models()
#save_current_cloth_as_init(base_dir)
# load_pickle_file(base_dir+'init.pickle')
cloth_simulation_init(base_dir)
cloth_simulation_step(base_dir)

