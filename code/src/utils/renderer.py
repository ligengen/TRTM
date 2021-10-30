import numpy as np
import bpy
import os
import sys
# import multiprocessing
import pathos.pools as pp


class Render:
    def __init__(self):
        # Output resolution
        bpy.context.scene.render.resolution_x = 1080
        bpy.context.scene.render.resolution_y = 1080
        self.cloth_division = 50 # set simulation parameters

        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)
        for n in tree.links:
            tree.links.remove(n)
        # Create new scene:
        scene = bpy.context.scene
        scene.render.use_multiview = False
        scene.render.views_format = 'STEREO_3D'
        rl = tree.nodes.new(type="CompositorNodeRLayers")
        composite = tree.nodes.new(type = "CompositorNodeComposite")
        composite.location = 200,0

        # TODO! view the depth map
        map = tree.nodes.new(type="CompositorNodeMapValue")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        map.size = [0.2]
        map.use_min = True
        map.min = [0]
        map.use_max = True
        map.max = [255]
        links.new(rl.outputs['Depth'], map.inputs[0])

        self.fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
        # fileOutput.format.file_format = "OPEN_EXR"
        links.new(map.outputs[0], self.fileOutput.inputs[0])
        self.fileOutput.base_path = '/home/crl-5/Desktop/cloth_recon/tmp'
        if not os.path.exists(self.fileOutput.base_path):
            os.mkdir(self.fileOutput.base_path)

        self.generate_models()

    def generate_models(self):  
        # delect old objects
        bpy.ops.object.select_all(action='DESELECT')
        for object in bpy.context.scene.objects:
            object.select_set(True)
        bpy.ops.object.delete()
        
        # add Plane object with Collision Modifier
        bpy.ops.mesh.primitive_plane_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        bpy.context.object.scale = [4, 4, 1]
        bpy.context.object.collision.cloth_friction = 80
        
        bpy.context.object.hide_render = True
        bpy.ops.object.modifier_add(type='COLLISION')

        # add Cloth Material
        MaterialCloth = bpy.data.materials.new('MaterialCloth')
        MaterialCloth.diffuse_color = (0.4, 0.031, 0.4549, 1)

        # add Cloth object with Cloth Material
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=self.cloth_division, y_subdivisions=self.cloth_division, enter_editmode=False, align='WORLD',
                                        location=(0, 0, 0.05), scale=(1, 1, 1))
        bpy.context.object.name = 'Cloth'
        bpy.context.object.active_material = MaterialCloth
        # add VERTEX_WEIGHT_MIX, HOOK and CLOTH Modifiers
        bpy.ops.object.modifier_add(type='VERTEX_WEIGHT_MIX')
        bpy.ops.object.modifier_add(type='HOOK')
        bpy.context.object.modifiers['Hook'].name = "Hook1"
        # bpy.ops.object.modifier_add(type='HOOK')
        # bpy.context.object.modifiers['Hook'].name = "Hook2"
        bpy.ops.object.modifier_add(type='CLOTH')
        bpy.ops.object.shade_smooth()
        # set Cloth parameters
        bpy.context.object.modifiers['Cloth'].settings.mass = 0.5  # cloth vertex mass
        bpy.context.object.modifiers['Cloth'].settings.air_damping = 1  # air viscosity
        bpy.context.object.modifiers['Cloth'].settings.tension_stiffness = 10  # resistance to tension between nodes
        bpy.context.object.modifiers['Cloth'].settings.compression_stiffness = 10  # resistance to compression between nodes
        bpy.context.object.modifiers['Cloth'].settings.shear_stiffness = 40
        bpy.context.object.modifiers['Cloth'].settings.bending_stiffness = 40
        bpy.context.object.modifiers['Cloth'].settings.tension_damping = 10
        bpy.context.object.modifiers['Cloth'].settings.compression_damping = 10
        bpy.context.object.modifiers['Cloth'].settings.shear_damping = 2
        bpy.context.object.modifiers['Cloth'].settings.bending_damping = 2
        bpy.context.object.modifiers['Cloth'].collision_settings.use_self_collision = True
        bpy.context.object.modifiers['Cloth'].collision_settings.self_friction = 40
        bpy.context.object.modifiers['Cloth'].collision_settings.self_distance_min = 0.01
        # add Camera and Sun object
        bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 3), scale=(1, 1, 1))
        bpy.ops.object.camera_add(align='VIEW', location=(0, 0, 4.5), rotation=(0, 0, 0), scale=(1, 1, 1))
        camera = bpy.data.objects['Camera']
        # camera.data.lens = 15 #(focal length)
        bpy.context.scene.camera = camera


    # assign numpy mesh data to cloth.data.vertices
    def assign(self, cloth, np_mesh):
        #    print('assign np_mesh with shape', np_mesh.shape, 'to cloth.data.vertices of size', len(cloth.data.vertices))
        for i in range(len(cloth.data.vertices)):
            for c in range(3):
                cloth.data.vertices[i].co[c] = np_mesh[i, c]


    def cloth_simulation_reset(self, mesh):
        # initialize cloth object
        cloth = bpy.data.objects['Cloth']
        # clear cloth simulation data
        cloth.animation_data_clear()
        cloth.vertex_groups.clear()
        # assign cloth to active object
        bpy.context.view_layer.objects.active = cloth
        # assign init np_meshes vertices data to cloth.data.vertices
        self.assign(cloth, mesh)
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

    def render(self, mesh, name):
        image_name = name + '.png'
        bpy.data.scenes['Scene'].render.filepath = os.path.join(self.fileOutput.base_path, image_name)
        self.cloth_simulation_reset(mesh)
        self.fileOutput.file_slots[0].path = image_name + '#'
        bpy.ops.render.render(write_still=True)
        os.remove(os.path.join(self.fileOutput.base_path, image_name))

    def render_batch(self, mesh_batch, name):
        meshlis = [i for i in mesh_batch]
        namelis = [name for i in range(mesh_batch.shape[0])]
        # with multiprocessing.Pool(8) as pool:
            # pool.starmap(self.render, zip(meshlis, namelis))
        p = pp.ProcessPool(8)
        # TODO! not pickleable and so it can not use multiprocessing
        p.map(self.render, meshlis, namelis)
        # for i in range(mesh_batch.shape[0]):
        #     self.render(mesh_batch[i], name+'_%05d'%i)

if __name__ == '__main__':
    render = Render()
    # bz = int(sys.argv[-1])
    # gt_mesh = np.fromstring(sys.argv[-3], float).reshape((bz, 2601, 3))
    # pred_mesh = np.fromstring(sys.argv[-2], float).reshape((bz, 2601, 3))
    base = '/home/crl-5/Desktop/cloth_recon/tmp/'
    # with open(base+'gt.npy', 'rb') as f:
    #     gt_mesh = np.load(f)
    # f.close()
    # with open(base+'pred.npy', 'rb') as f:
    #     pred_mesh = np.load(f)
    # f.close()
    name = sys.argv[-1]
    mesh = np.loadtxt(base + name) 
    render.render(mesh, name.split('.')[0])
    # mesh = np.loadtxt('/home/crl-5/Desktop/50/test_label/04302.txt')
    # render.render_batch(gt_mesh, 'gt')
    # render.render_batch(pred_mesh, 'pred')
    # render.render(mesh, 'test')

