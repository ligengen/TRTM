import numpy as np
import bpy
import pickle
import os
import time
import cv2 as cv
import matplotlib.pyplot as plt

# generate Plane, Hand, Cloth Objects, Modifiers
def generate_models(shear=40, bend=40, division=50):
    # set simulation parameters
    baseZ = 0.05
    bpy.context.scene.frame_set(0)

    # delete old objects
    bpy.ops.object.select_all(action='DESELECT')
    for object in bpy.context.scene.objects:
        object.select_set(True)
    bpy.ops.object.delete()

    # add Ground object with Collision Modifier
    bpy.ops.mesh.primitive_plane_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.context.object.scale = [4, 4, 1]
    bpy.context.object.name = 'Ground'
    bpy.context.object.collision.cloth_friction = 80  # friction damping between ground and cloth
    bpy.ops.object.modifier_add(type='COLLISION')

    # add Cloth Material
    MaterialCloth = bpy.data.materials.new('MaterialCloth')
    MaterialCloth.diffuse_color = (0.1, 0.8, 0.8, 1)

    # add Cloth object with Cloth Material
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=division, y_subdivisions=division, enter_editmode=False,
                                    align='WORLD', location=(0, 0, baseZ), scale=(1, 1, 1))
    bpy.context.object.name = 'Cloth'
    bpy.context.object.active_material = MaterialCloth
    bpy.ops.object.shade_smooth()

    # add VERTEX_WEIGHT_MIX, HOOK and CLOTH Modifiers
    bpy.ops.object.modifier_add(type='TRIANGULATE')
    bpy.context.object.modifiers["Triangulate"].quad_method = 'FIXED'
    bpy.ops.object.modifier_add(type='VERTEX_WEIGHT_MIX')
    bpy.ops.object.modifier_add(type='HOOK')
    bpy.context.object.modifiers['Hook'].name = "Hook1"
    bpy.ops.object.modifier_add(type='HOOK')
    bpy.context.object.modifiers['Hook'].name = "Hook2"
    bpy.ops.object.modifier_add(type='CLOTH')

    # set Cloth parameters
    bpy.context.object.modifiers['Cloth'].settings.mass = 0.5  # cloth vertex mass
    bpy.context.object.modifiers['Cloth'].settings.air_damping = 1  # air viscosity

    bpy.context.object.modifiers['Cloth'].settings.tension_stiffness = 10  # resistance to tension between nodes
    bpy.context.object.modifiers['Cloth'].settings.compression_stiffness = 10  # resistance to compression between nodes
    bpy.context.object.modifiers['Cloth'].settings.shear_stiffness = shear  # resistance to shear within rectangular
    bpy.context.object.modifiers['Cloth'].settings.bending_stiffness = bend  # resistance to bend between rectangular

    bpy.context.object.modifiers['Cloth'].settings.tension_damping = 10
    bpy.context.object.modifiers['Cloth'].settings.compression_damping = 10
    bpy.context.object.modifiers['Cloth'].settings.shear_damping = 2
    bpy.context.object.modifiers['Cloth'].settings.bending_damping = 2

    bpy.context.object.modifiers['Cloth'].collision_settings.use_self_collision = True
    bpy.context.object.modifiers['Cloth'].collision_settings.self_friction = 40
    bpy.context.object.modifiers['Cloth'].collision_settings.self_distance_min = 0.01

    # add Camera and Sun object
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 5), rotation=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 6), scale=(1, 1, 1))
    bpy.context.scene.render.resolution_x = 1080
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.camera = bpy.data.objects['Camera']


## state data manipulation functions
# state = {np_mesh(nvtx, 3), np_values(1, 4), np_picks(2, 2), np_moves(2, 2), np_heights(1, 2), ID, parent_ID}
# save state information to state.pickle or init.pickle
def save_state(np_mesh, np_values, np_picks, np_moves, np_heights, ID, parent_ID, base_dir, as_init=False):
    # construct state dictionary
    state = {'mesh': np_mesh,
             'values': np_values,
             'picks': np_picks,
             'moves': np_moves,
             'heights': np_heights,
             'ID': ID,
             'parent_ID': parent_ID}
    # save state as init.pickle, initialize IDvalue.csv
    if as_init:
        # save state into init.pickle
        with open(base_dir + 'init.pickle', mode='wb') as f:
            pickle.dump(state, f)
        # initialize IDvalue.csv
        data = np.append(ID, np_values)
        data = np.append(data, parent_ID)
        data = data.reshape(1, data.shape[0])
        np.savetxt(base_dir + 'IDvalue.csv', data, delimiter=',', fmt='%1.3e')
    # save state as state[ID].pickle, append IDvalue.csv
    else:
        data = np.append(ID, np_values)
        data = np.append(data, parent_ID)
        data = data.reshape(1, data.shape[0])
        IDvalue = np.genfromtxt(base_dir + 'IDvalue.csv', delimiter=',')
        IDvalue = IDvalue.reshape(-1, data.shape[1])
        if IDvalue.shape[0] > ID:
            IDvalue[ID] = data
        else:
            IDvalue = np.append(IDvalue, data, axis=0)
        np.savetxt(base_dir + 'IDvalue.csv', IDvalue, delimiter=',', fmt='%1.3e')

    # save state into state[ID].pickle
    with open(base_dir + 'state[' + str(ID) + '].pickle', mode='wb') as f:
        pickle.dump(state, f)

# save current cloth as init.pickle, initialize IDvalue.csv
def save_current_cloth_as_init(base_dir):
    print('saving current Cloth as initial state...')
    # load current Cloth Mesh
    cloth = bpy.data.objects['Cloth']
    # convert current Cloth Mesh to 3D Numpy Array
    save_state(convert_mesh_to_np_mesh(cloth.data, centralize=True), np.zeros(4), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2), 0, 0, base_dir, as_init=True)


# load and return np_mesh from state[ID].pickle
def load_state_mesh(base_dir, ID):
    try:
        state = pickle.load(open(base_dir + 'state[' + str(ID) + '].pickle', mode='rb'))
        np_mesh = state['mesh']
        return np_mesh
    except IOError:
        return np.array([])

# load and return state from state[ID].pickle
def load_state(base_dir, ID):
    try:
        state = pickle.load(open(base_dir + 'state[' + str(ID) + '].pickle', mode='rb'))
        return state
    # detect IOError
    except IOError:
        return np.array([])

# load and return last state ID
def load_ID_number(base_dir):
    IDvalue = np.genfromtxt(base_dir + 'IDvalue.csv', delimiter=',')
#    IDvalue = IDvalue.reshape(-1, 6)
    print('current IDvalue shape:', IDvalue.shape)
    return IDvalue.shape[0]


# centralize numpy mesh to (x, y) mean center
def centralize_np_mesh(np_mesh):
    center = np.mean(np_mesh, axis=0)
    print('cloth center before centralization:', center)
    center[2] = 0
    for i in range(np_mesh.shape[0]):
        np_mesh[i] = np_mesh[i] - center
    print('cloth center after centralization:', np.mean(np_mesh, axis=0))
    return np_mesh

# convert Blender Mesh to Coordinate Numpy Array: np_mesh = convert_mesh_to_np_mesh(cloth.data)
def convert_mesh_to_np_mesh(mesh, centralize=False):
    n_vtx = len(mesh.vertices)
    np_mesh = np.empty((n_vtx, 3))
    for i in range(n_vtx):
        np_mesh[i] = np.array(mesh.vertices[i].co)
    if centralize:
        np_mesh = centralize_np_mesh(np_mesh)
    print(n_vtx, ' mesh converted to array of shape', np_mesh.shape)
    return np_mesh

# convert Blender Mesh to Normal Vector Numpy Array: np_normal = convert_mesh_to_np_normal(cloth.data)
def convert_mesh_to_np_normal(mesh):
    n_vtx = len(mesh.vertices)
    np_normal = np.empty((n_vtx, 3))
    for i in range(n_vtx):
        np_normal[i] = np.array(mesh.vertices[i].normal)
    return np_normal

# assign np_mesh data to Cloth data
def assign(cloth, np_mesh):
    for i in range(len(cloth.data.vertices)):
        cloth.data.vertices[i].co = np_mesh[i]

# find pick lists from co_picks
def find_pick_lists(np_mesh, co_picks, grasp_range=0.05):
    # initialize pick lists
    pick_list1 = []
    pick_list2 = []
    pick_list = []

    # pick Cloth nodes near the co_picks
    for i in range(np_mesh.shape[0]):
        # get distance between vertex and pick point
        d1 = ((np_mesh[i, 0] - co_picks[0][0]) ** 2 + (np_mesh[i, 1] - co_picks[0][1]) ** 2) ** 0.5
        d2 = ((np_mesh[i, 0] - co_picks[1][0]) ** 2 + (np_mesh[i, 1] - co_picks[1][1]) ** 2) ** 0.5
        if d1 < grasp_range:
            pick_list1.append(i)
        if d2 < grasp_range and d1 >= grasp_range:
            pick_list2.append(i)

    # only pick the top nodes
    dimension = int(np.sqrt(np_mesh.shape[0]))
    # pick topest node from pick_list1
    if pick_list1 == []:
        print('pick_list1 is empty!')
    else:
        pick_point1 = pick_list1[0]
        for i in pick_list1:
            if np_mesh[i, 2] > np_mesh[pick_point1, 2]:
                pick_point1 = i
        temp_list = pick_list1
        pick_list1 = []
        # append nodes around topest node
        for i in temp_list:
            if i in [pick_point1, pick_point1+1, pick_point1-1, pick_point1+dimension, pick_point1-dimension]:
                pick_list1.append(i)
                pick_list.append(i)
    # pick topest node from pick_list2
    if pick_list2 == []:
        print('pick_list2 is empty!')
    else:
        pick_point2 = pick_list2[0]
        for i in pick_list2:
            if np_mesh[i, 2] > np_mesh[pick_point2, 2]:
                pick_point2 = i
        temp_list = pick_list2
        pick_list2 = []
        for i in temp_list:
            # append nodes around topest node
            if i in [pick_point2, pick_point2+1, pick_point2-1, pick_point2+dimension, pick_point2-dimension]:
                pick_list2.append(i)
                pick_list.append(i)

    print('pick_list vertices:', pick_list)
    return pick_list1, pick_list2, pick_list


# simulate Cloth forward with one action sequence: start_ID, end_ID, np_picks, np_moves, np_heights
def cloth_simulation_forward(base_dir, start_ID, end_ID, np_picks=np.zeros((2, 2)), np_moves=np.zeros((2, 2)), np_heights=np.zeros(2), as_init=False, random=False, unfold=False, save=True):
    # initialize simulation
    baseZ = 0.05
    if unfold: generate_models(shear=20, bend=20, division=50)
    else: generate_models(shear=40, bend=40, division=50)
    print('cloth simulation reset with: state[' + str(start_ID) + '].pickle')

    # initialize cloth object
    cloth = bpy.data.objects['Cloth']
    cloth.location = (0, 0, baseZ)
    # clear animation for cloth
    cloth.animation_data_clear()
    cloth.vertex_groups.clear()
    # assign cloth to active object
    bpy.context.view_layer.objects.active = cloth

    # load np_mesh from state[start_ID].pickle
    np_mesh = load_state_mesh(base_dir, start_ID)
    # assign np_mesh vertices data to cloth.data.vertices
    assign(cloth, np_mesh)

    # generate random action sequence
    if random:
        # generate random pick points, assert pick_list is not empty
        while True:
            # generate random pick points
            np_picks = np.random.uniform(-1, 1, (2, 2))
            np_picks[1] = np_picks[0]
            np.around(np_picks, 3)
            # find pick nodes from pick positions: np_picks
            pick_list1, pick_list2, pick_list = find_pick_lists(np_mesh, np_picks)
            if pick_list != []:
                break
        # generate random move vector
        np_moves = np.random.uniform(-1, 1, (2, 2))
        np_moves[1] = np_moves[0]
        np.around(np_moves, 3)
        # generate random lift height, assert safe lift height
        np_heights = np.random.uniform(0, 1, (2))
        np_heights[0] = max(np_heights[0], np.amax(np_mesh, axis=0)[2]+0.2)
        np_heights[1] = np_heights[0]
        np.around(np_heights, 3)
        print('np_picks:', np_picks)
        print('np_moves:', np_moves)
        print('np_heights:', np_heights)

    # simulate one action sequence
    # get action sequence
    co_picks = np_picks
    co_moves = np_moves
    lift_height1 = np_heights[0]
    lift_height2 = np_heights[1]

    # find pick nodes from pick positions: co_picks
    pick_list1, pick_list2, pick_list = find_pick_lists(np_mesh, co_picks)

    # initialize frame number
    frame_num = 0
    # wait for stablization
    frame_num += 10

    # initialize cloth vertex groups
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

    # computing movement trajectory
    # get initial hand pick positions
    x10 = co_picks[0][0]
    x1 = x10
    y10 = co_picks[0][1]
    y1 = y10
    z1 = baseZ
    x20 = co_picks[1][0]
    x2 = x20
    y20 = co_picks[1][1]
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
    cloth.modifiers["Cloth"].settings.pin_stiffness = 20
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
    move_length1 = (co_moves[0][0] ** 2 + co_moves[0][1] ** 2) ** .5
    if move_length1 == 0:
        move_normal1 = np.array([1, 0])
    else:
        move_normal1 = co_moves[0] / move_length1
    move_length2 = (co_moves[1][0] ** 2 + co_moves[1][1] ** 2) ** .5
    if move_length2 == 0:
        move_normal2 = np.array([1, 0])
    else:
        move_normal2 = co_moves[1] / move_length2

    # simulate lift move action
    for i in range(0, 200):
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
    frame_num += 10

    # simulate horizontal move action
    for i in range(0, 200):
        # iterative x1, y1 hand positions with smaller steps
        x1 = x10 + np.sign(co_moves[0][0]) * min(abs(x1 - x10 + move_normal1[0] * xy_step), abs(co_moves[0][0]))
        y1 = y10 + np.sign(co_moves[0][1]) * min(abs(y1 - y10 + move_normal1[1] * xy_step), abs(co_moves[0][1]))
        hand1.location = (x1, y1, z1)
        # insert hand keyframe
        hand1.keyframe_insert(data_path="location", frame=frame_num)
        # iterative x2, y2 hand positions with smaller steps
        x2 = x20 + np.sign(co_moves[1][0]) * min(abs(x2 - x20 + move_normal2[0] * xy_step), abs(co_moves[1][0]))
        y2 = y20 + np.sign(co_moves[1][1]) * min(abs(y2 - y20 + move_normal2[1] * xy_step), abs(co_moves[1][1]))
        hand2.location = (x2, y2, z2)
        # insert hand keyframe
        hand2.keyframe_insert(data_path="location", frame=frame_num)
        frame_num += 1
        if x1 == x10 + co_moves[0][0] and y1 == y10 + co_moves[0][1] and x2 == x20 + co_moves[1][0] and y2 == y20 + co_moves[1][1]: break
    frame_num += 5

    # release hand pin points
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
    # wait for stablization
    frame_num += 20

    # initialize start time
    t_start = time.time()
    # simulation frame forward
    for i in range(frame_num + 1):
        t_frame = time.time()
        # set scene frame
        bpy.context.scene.frame_set(i)
        print('play', i, '/', frame_num, 'time:', time.time() - t_frame)

    # set end frame
    bpy.data.scenes["Scene"].frame_end = frame_num
    print('simulation time:', time.time() - t_start)
    # deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # get final Cloth mesh, centralize Cloth mesh around Cloth mean center
#    result_cloth = cloth.evaluated_get(bpy.context.evaluated_depsgraph_get())
    result_np_mesh = convert_mesh_to_np_mesh(cloth.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh(), centralize=True)
    # save state information with zero np_values
    if save:
        np_values = np.zeros(4)
        save_state(result_np_mesh, np_values, np_picks, np_moves, np_heights, end_ID, start_ID, base_dir)


## OpenCv evaluation funstion
# patch Cloth Values Text at the Upper Left Corner
def patch_image_notation(image):
    for i in range(0, 50):
        for j in range(image.shape[0]):
            image[i, j] = (156, 156, 156)
    return image

# evaluate Cloth Coverage Value from Cloth Top-View PNG
def cloth_coverage_value(base_dir, ID):
    low = (152, 152, 152)
    high = (160, 160, 160)
    img = cv.imread(base_dir + 'state[' + str(ID) + '].png')
    img = patch_image_notation(img)
    mask = cv.inRange(img, low, high)
    # coverage_value = Percent of Cloth in Image
    coverage_value = 1 - cv.countNonZero(mask)/(mask.shape[0]*mask.shape[1])
#    print('coverage_value of state[' + str(ID) + ']:', coverage_value)
    return coverage_value

# find edge vertices of n_vertex
def find_loop_vertices(n_vertex, dimension):
    # initialize loop_list with potential edge vertices
    loop_list = [n_vertex+1, n_vertex-1, n_vertex+dimension, n_vertex-dimension, n_vertex+dimension+1, n_vertex-dimension-1]
#    print('start loop list:', loop_list)
    # delete non-existing edge vertices
    if n_vertex%dimension == 0:
        try: loop_list.remove(n_vertex-1)
        except: pass
        try: loop_list.remove(n_vertex-dimension-1)
        except: pass
    if n_vertex%dimension == dimension-1:
        try: loop_list.remove(n_vertex+1)
        except: pass
        try: loop_list.remove(n_vertex+dimension+1)
        except: pass
    if n_vertex < dimension:
        try: loop_list.remove(n_vertex-dimension)
        except: pass
        try: loop_list.remove(n_vertex-dimension-1)
        except: pass
    if dimension*dimension - n_vertex <= dimension:
        try: loop_list.remove(n_vertex+dimension)
        except: pass
        try: loop_list.remove(n_vertex+dimension+1)
        except: pass
#    print('final loop list:', loop_list)
    return loop_list

# evaluate Cloth Curvature Value from Cloth np_mesh and np_normal
def cloth_curvature_value(np_mesh, np_normal):
    # initialize np_curvature
    np_curvature = np.zeros(np_mesh.shape[0])
    # get np_mesh dimension
    dimension = int(np.sqrt(np_mesh.shape[0]))
    # compute average curvature of each mesh vertex
    for n_vertex in range(np_mesh.shape[0]):
        # get loop_list around vertex
        loop_list = find_loop_vertices(n_vertex, dimension)
        # compute curvature for each edge
        for loop_vertex in loop_list:
            n_diff = np_normal[loop_vertex] - np_normal[n_vertex]
            p_diff = np_mesh[loop_vertex] - np_mesh[n_vertex]
            curvature = n_diff.dot(p_diff) / np.linalg.norm(p_diff)
            np_curvature[n_vertex] += abs(curvature)
        # average curvature over all vertex edges
        np_curvature[n_vertex] = np_curvature[n_vertex] / len(loop_list)
    # curvature_value = Average curvature for entire mesh
    curvature_value = np.sum(np_curvature) / np_curvature.shape[0]
#    print('curvature_value of state[' + str(ID) + ']:', curvature_value)
    return curvature_value

# evaluate Cloth Values, save to IDvalue.csv and state[ID].pickle
def evaluate_cloth(base_dir, ID, coverage=False, curvature=False):
    # initialize simulation
    baseZ = 0.05
    generate_models()
    print('cloth simulation initialize with: state[' + str(ID) + '].pickle')

    # initialize cloth object
    cloth = bpy.data.objects['Cloth']
    cloth.location = (0, 0, baseZ)
    # clear animation for cloth
    cloth.animation_data_clear()
    cloth.vertex_groups.clear()
    # assign cloth to active object
    bpy.context.view_layer.objects.active = cloth

    # load state mesh information from state[ID].pickle
    np_mesh = load_state_mesh(base_dir, ID)
    # assign np_mesh vertices data to cloth.data.vertices
    assign(cloth, np_mesh)

    # initialize frame number
    frame_num = 0
    # wait for stablization
    if ID == 0: frame_num += 30
    else: frame_num += 1

    # initialize start time
    t_start = time.time()
    # simulation frame forward
    for i in range(frame_num + 1):
        t_frame = time.time()
        # set scene frame
        bpy.context.scene.frame_set(i)
#        print('play', i, '/', frame_num, 'time:', time.time() - t_frame)
    # set end frame
    bpy.data.scenes["Scene"].frame_end = frame_num
#    print('simulation time:', time.time() - t_start)
    # deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # get final Cloth object and mesh
    result_cloth = cloth.evaluated_get(bpy.context.evaluated_depsgraph_get())
    result_np_mesh = convert_mesh_to_np_mesh(result_cloth.to_mesh(), centralize=False)

    # evaluate Cloth Coverage Value
    coverage_value = 0
    if coverage:
        # render Top-View Cloth into state[ID].png
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = base_dir + 'state[' + str(ID) + '].png'
        bpy.ops.render.render(write_still=1)
        # get cloth_coverage_value
        coverage_value = cloth_coverage_value(base_dir, ID)
        print('coverage_value of state[' + str(ID) + ']:', coverage_value)

    # evaluate Cloth Curvature Value
    curvature_value = 0
    if curvature:
        result_np_normal = convert_mesh_to_np_normal(result_cloth.to_mesh())
        # get cloth_curvature_value
        curvature_value = cloth_curvature_value(result_np_mesh, result_np_normal)
        print('curvature_value of state[' + str(ID) + ']:', curvature_value)

    return coverage_value, curvature_value


# update IDvalue.csv and pickle with evaluation state values
def update_IDvalue_pickle(base_dir):
    # get IDvalue from IDvalue.csv
    IDvalue = np.genfromtxt(base_dir + 'IDvalue.csv', delimiter=',')
    # evaluate State Values for all Clothes
    for ID in range(0, IDvalue.shape[0]):
        # evaluate state values
        coverage_value, curvature_value = evaluate_cloth(base_dir, ID, coverage=True, curvature=True)
        # update IDvalue with state values
        IDvalue[ID, 2] = coverage_value
        IDvalue[ID, 3] = curvature_value
        # load Cloth state information
        state = pickle.load(open(base_dir + 'state[' + str(ID) + '].pickle', mode='rb'))
        # update Cloth state values in pickle
        state['values'][0] = coverage_value
        state['values'][1] = curvature_value
        # save updated state information into state[ID].pickle
        with open(base_dir + 'state[' + str(ID) + '].pickle', mode='wb') as f:
            pickle.dump(state, f)
    # save updated IDvalue into IDvalue.csv
    np.savetxt(base_dir + 'IDvalue.csv', IDvalue, delimiter=',', fmt='%.4f')


# update Cloth Value Text at the Upper Left Corner of the Image from IDvalue.csv
def update_image_notation(base_dir):
    # get Cloth IDvalue
    IDvalue = np.genfromtxt(base_dir + 'IDvalue.csv', delimiter=',')
    for ID in range(IDvalue.shape[0]):
        img = cv.imread(base_dir + 'state[' + str(ID) + '].png', 1)
        img = patch_image_notation(img)
        font = cv.FONT_HERSHEY_SIMPLEX
        notation = 'state[' + str(IDvalue[ID, 0]) + ']' + ' values: ' + str(IDvalue[ID, 2]) + ', ' + str(IDvalue[ID, 3]) + ', ' + str(IDvalue[ID, 4]) + ', ' + str(IDvalue[ID, 5])
        cv.putText(img, notation, (0, 30), font, 1, (0, 0, 0), 2, cv.LINE_AA)
        cv.imwrite(base_dir + 'state[' + str(ID) + '].png', img)


base_dir = '/Users/aaronw/Desktop/Blender/week4/cloth_evaluate/data/'


## simple test actions
#cloth_simulation_forward(base_dir, 20, 1, np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), np.array([0.5, 0.5]), random=False, unfold=False, save=False)
#cloth_simulation_forward(base_dir, 1, 2, np.array([[0.3, -0.3], [-0.3, 0.3]]), np.array([[0.5, -0.5], [-0.5, 0.5]]), np.array([0.2, 0.2]), random=False, unfold=True, save=False)
#cloth_simulation_forward(base_dir, 0, 1, np.array([[0.5, 0.5], [-0.5, -0.5]]), np.array([[-0.5, 0], [0.5, 0]]), np.array([0.5, 0.5]), random=False, unfold=False, save=False)
#cloth_simulation_forward(base_dir, 1, 2, np.array([[0.5, -0.5], [-0.5, 0.5]]), np.array([[0.2, -0.2], [-0.2, 0.2]]), np.array([0.2, 0.2]), random=False, unfold=True, save=False)


## system initialization
generate_models()
#save_current_cloth_as_init(base_dir)
#load_cloth(base_dir, 0)

## generate random initial states
## generate 10 Base Cloth ID: 1~10, with Random action sequence
#for i in range(10):
#    cloth_simulation_forward(base_dir, 0, i+1, random=True)

## generate 4 Cloth ID: 11~50 from each of 10 Base Cloth ID: 1~10, with Random action sequence
#for i in range(10):
#    for j in range(4):
#        ID = load_ID_number(base_dir)
#        cloth_simulation_forward(base_dir, i+1, ID, random=True)

## generate 1 Cloth ID: 51~100 from each of 50 Cloth ID: 1~50, with Random action sequence
#for i in range(50):
#    ID = load_ID_number(base_dir)
#    cloth_simulation_forward(base_dir, i+1, ID, random=True)


## evaluate  and update Cloth Values
#update_IDvalue_pickle(base_dir)
#update_image_notation(base_dir)


## imshow image
#cv.namedWindow(name)
#cv.imshow(name, mask)
#cv.waitKey (0)
#cv.destroyAllWindows()


