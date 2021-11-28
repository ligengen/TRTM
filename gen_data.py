import numpy as np
import bpy
import os
import time

# generate Plane, Hand, Cloth Objects, Modifiers
def generate_models(shear=40, bend=40, division=50, multi=False):
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
    bpy.context.object.hide_render = True
    bpy.ops.object.modifier_add(type='COLLISION')

    # add Cloth Material
    MaterialCloth = bpy.data.materials.new('MaterialCloth')
    MaterialCloth.diffuse_color = (0.4, 0.031, 0.4549, 1)

    # add Cloth object with Cloth Material
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=division, y_subdivisions=division, enter_editmode=False,
                                    align='WORLD', location=(0, 0, baseZ), scale=(1, 1, 1))
    bpy.context.object.name = 'Cloth'
    bpy.context.object.active_material = MaterialCloth
    bpy.ops.object.shade_smooth()

    # add VERTEX_WEIGHT_MIX, HOOK and CLOTH Modifiers
    bpy.ops.object.modifier_add(type='VERTEX_WEIGHT_MIX')
    bpy.ops.object.modifier_add(type='HOOK')
    bpy.context.object.modifiers['Hook'].name = "Hook1"
    if multi:
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
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 4.5), rotation=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 6), scale=(1, 1, 1))
    bpy.context.scene.render.resolution_x = 1080
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.camera = bpy.data.objects['Camera']
    

# centralize numpy mesh to (x, y) mean center
def centralize_np_mesh(np_mesh):
    # center = np.mean(np_mesh, axis=0)
    center = np_mesh[1300]
    center[2] = 0
    np_mesh -= center
    return np_mesh

# convert Blender Mesh to Coordinate Numpy Array: np_mesh = convert_mesh_to_np_mesh(cloth.data)
def convert_mesh_to_np_mesh(mesh, centralize=False):
    n_vtx = len(mesh.vertices)
    np_mesh = np.empty((n_vtx, 3))
    for i in range(n_vtx):
        np_mesh[i] = np.array(mesh.vertices[i].co)
    if centralize:
        np_mesh = centralize_np_mesh(np_mesh)
    return np_mesh

# assign np_mesh data to Cloth data
def assign(cloth, np_mesh):
    for i in range(len(cloth.data.vertices)):
        cloth.data.vertices[i].co = np_mesh[i]

# find pick lists from co_picks 
def find_pick_lists(np_mesh, co_picks, grasp_range=0.05, multi=False):
    # initialize pick lists
    pick_list1 = []
    if multi:
        pick_list2 = []
    pick_list = []
    
    # pick Cloth nodes near the co_picks
    for i in range(np_mesh.shape[0]):
        # get distance between vertex and pick point
        d1 = ((np_mesh[i, 0] - co_picks[0][0]) ** 2 + (np_mesh[i, 1] - co_picks[0][1]) ** 2) ** 0.5
        if multi:
            d2 = ((np_mesh[i, 0] - co_picks[1][0]) ** 2 + (np_mesh[i, 1] - co_picks[1][1]) ** 2) ** 0.5
        if d1 < grasp_range:
            pick_list1.append(i)
        if multi:
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
    if multi:
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
    if multi:
        return pick_list1, pick_list2, pick_list
    return pick_list1, pick_list


def cloth_simulation_forward(np_mesh, random=False, unfold=False, multi=False):
    # initialize simulation
    baseZ = 0.05
    if unfold: generate_models(shear=20, bend=20, division=50, multi=multi)
    else: generate_models(shear=40, bend=40, division=50, multi=multi)
    
    # initialize cloth object
    cloth = bpy.data.objects['Cloth']
    cloth.location = (0, 0, baseZ)
    # clear animation for cloth
    cloth.animation_data_clear()
    cloth.vertex_groups.clear()
    # assign cloth to active object
    bpy.context.view_layer.objects.active = cloth

    # np_mesh = np.loadtxt(init_path)
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
            if multi:
                pick_list1, pick_list2, pick_list = find_pick_lists(np_mesh, np_picks, multi=multi)
            else:
                pick_list1, pick_list = find_pick_lists(np_mesh, np_picks, multi=multi)
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
    
    # simulate one action sequence
    # get action sequence
    co_picks = np_picks
    co_moves = np_moves
    lift_height1 = np_heights[0]
    lift_height2 = np_heights[1]
    
    # find pick nodes from pick positions: co_picks
    if multi:
        pick_list1, pick_list2, pick_list = find_pick_lists(np_mesh, co_picks, multi=multi)
    else:
        pick_list1, pick_list = find_pick_lists(np_mesh, co_picks, multi=multi)
    
    # initialize frame number
    frame_num = 0
    # wait for stablization
    frame_num += 10
    
    # initialize cloth vertex groups
    empty = cloth.vertex_groups.new(name='empty')
    pick1 = cloth.vertex_groups.new(name='pick1')
    if multi:
        pick2 = cloth.vertex_groups.new(name='pick2')
    pick = cloth.vertex_groups.new(name='pick')
    # add vertex group, weight paint the pick-up vertices
    pick1.add(pick_list1, 1.0, 'REPLACE')
    if multi:
        pick2.add(pick_list2, 1.0, 'REPLACE')
    pick.add(pick_list, 1.0, 'REPLACE')
    # assign vertex groups to Hook
    cloth.modifiers['Hook1'].vertex_group = 'pick1'
    if multi:
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
    if multi:
        bpy.ops.object.empty_add(type='SINGLE_ARROW', align='WORLD', location=(x20, y20, baseZ), scale=(1, 1, 1))
        bpy.context.object.name = 'Hand2'
        bpy.context.object.scale = [1, 1, 0.2]
    # assign Hand1 and Hand2 to Hook1 and Hook2
    cloth.modifiers["Hook1"].object = bpy.data.objects["Hand1"]
    if multi:
        cloth.modifiers["Hook2"].object = bpy.data.objects["Hand2"]
    cloth.modifiers["Cloth"].settings.pin_stiffness = 20
    # clear animation for hand1 and hand2
    hand1 = bpy.data.objects['Hand1']
    hand1.animation_data_clear()
    if multi:
        hand2 = bpy.data.objects['Hand2']
        hand2.animation_data_clear()
    # initialize Hand and Cloth simulation
    hand1.location = (x10, y10, baseZ)
    hand1.keyframe_insert(data_path="location", frame=frame_num)
    if multi:
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
        if multi:
            hand2.location = (x2, y2, z2)
            # insert hand keyframe
            hand2.keyframe_insert(data_path="location", frame=frame_num)
        frame_num += 1
        if multi:
            if z1 == baseZ + lift_height1 and z2 == baseZ + lift_height2: break
        else:
            if z1 == baseZ + lift_height1: # and z2 == baseZ + lift_height2: break
                break
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
        if multi:
            hand2.location = (x2, y2, z2)
            # insert hand keyframe
            hand2.keyframe_insert(data_path="location", frame=frame_num)
        frame_num += 1
        if multi:
            if x1 == x10 + co_moves[0][0] and y1 == y10 + co_moves[0][1] and x2 == x20 + co_moves[1][0] and y2 == y20 + co_moves[1][1]: break
        else:
            if x1 == x10 + co_moves[0][0] and y1 == y10 + co_moves[0][1]: # and x2 == x20 + co_moves[1][0] and y2 == y20 + co_moves[1][1]: break
                break
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
    if multi:
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
    
    # set end frame
    bpy.data.scenes["Scene"].frame_end = frame_num
    print('simulation time:', time.time() - t_start)
    # deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    
    # get final Cloth mesh, centralize Cloth mesh around Cloth mean center
#    result_cloth = cloth.evaluated_get(bpy.context.evaluated_depsgraph_get())
    result_np_mesh = convert_mesh_to_np_mesh(cloth.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh(), centralize=False)
    return result_np_mesh


base_dir = '/home/crl-5/Desktop/dataset/'

bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

for n in tree.nodes:
    tree.nodes.remove(n)
for n in tree.links:
    tree.links.remove(n)

scene = bpy.context.scene
scene.render.use_multiview = False
scene.render.views_format = 'STEREO_3D'
rl = tree.nodes.new(type="CompositorNodeRLayers")
composite = tree.nodes.new(type = "CompositorNodeComposite")
composite.location = 200,0

map = tree.nodes.new(type="CompositorNodeMapValue")
# Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
map.size = [0.2]
map.use_min = True
map.min = [0]
map.use_max = True
map.max = [255]
links.new(rl.outputs['Depth'], map.inputs[0])

fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
fileOutput.base_path = base_dir
links.new(map.outputs[0], fileOutput.inputs[0])


## system initialization
generate_models()
num_data = 0
for i in range(50000):
    multi = False
    if np.random.uniform() < 0.3:
        multi = True

    r = np.random.uniform()
    if r <= 0.4:
        drag_time = 1
    elif 0.4 < r <= 0.7:
        drag_time = 2
    elif 0.7 < r <= 0.9:
        drag_time = 3
    else:
        drag_time = 4

    mesh = np.loadtxt('/home/crl-5/Desktop/cloth_recon/start_state.txt')
    for j in range(drag_time):
        mesh = cloth_simulation_forward(mesh, random=True, multi=multi)
    np.savetxt(os.path.join(base_dir, '%05d.txt'%num_data), mesh)
    image_name = '%05d' % num_data + '.png'
    bpy.data.scenes['Scene'].render.filepath = os.path.join(fileOutput.base_path, image_name)
    fileOutput.file_slots[0].path = image_name + '#'
    bpy.ops.render.render(write_still=True)
    os.remove(os.path.join(fileOutput.base_path, '%05d.png' % num_data))
    num_data += 1

