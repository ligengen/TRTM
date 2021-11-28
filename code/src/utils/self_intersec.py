import numpy as np
import bpy
import sys
import os

baseZ = 0.05
cloth_division = 50

# delect old objects
bpy.ops.object.select_all(action='DESELECT')
for object in bpy.context.scene.objects:
    object.select_set(True)
bpy.ops.object.delete()

MaterialCloth = bpy.data.materials.new('MaterialCloth')
bpy.ops.mesh.primitive_grid_add(x_subdivisions=cloth_division, y_subdivisions=cloth_division, enter_editmode=False, align='WORLD',
                                location=(0, 0, baseZ), scale=(1, 1, 1))
bpy.context.object.name = 'Cloth'
bpy.context.object.active_material = MaterialCloth
bpy.ops.object.shade_smooth()

bpy.ops.object.modifier_add(type='TRIANGULATE')
bpy.context.object.modifiers['Triangulate'].quad_method = "FIXED"
bpy.ops.object.modifier_add(type='VERTEX_WEIGHT_MIX')
bpy.ops.object.modifier_add(type='CLOTH')

# set Cloth parameters
bpy.context.object.modifiers['Cloth'].settings.mass = 0.5  # cloth vertex mass
bpy.context.object.modifiers['Cloth'].settings.air_damping = 1  # air viscosity

bpy.context.object.modifiers['Cloth'].settings.tension_stiffness = 10  # resistance to tension between nodes
bpy.context.object.modifiers['Cloth'].settings.compression_stiffness = 10  # resistance to compression between nodes
bpy.context.object.modifiers['Cloth'].settings.shear_stiffness = 40  # resistance to shear within rectangular
bpy.context.object.modifiers['Cloth'].settings.bending_stiffness = 40  # resistance to bend between rectangular

bpy.context.object.modifiers['Cloth'].settings.tension_damping = 10
bpy.context.object.modifiers['Cloth'].settings.compression_damping = 10
bpy.context.object.modifiers['Cloth'].settings.shear_damping = 2
bpy.context.object.modifiers['Cloth'].settings.bending_damping = 2

bpy.context.object.modifiers['Cloth'].collision_settings.use_self_collision = True
bpy.context.object.modifiers['Cloth'].collision_settings.self_friction = 40
bpy.context.object.modifiers['Cloth'].collision_settings.self_distance_min = 0.01

cloth = bpy.data.objects['Cloth']
cloth.animation_data_clear()
cloth.vertex_groups.clear()
bpy.context.view_layer.objects.active = cloth

file = sys.argv[-1]
mesh = np.loadtxt(file)
for i in range(len(cloth.data.vertices)):
    for c in range(3):
        cloth.data.vertices[i].co[c] = mesh[i, c]

basedir = '/home/crl-5/Desktop/cloth_recon/tmp/'
if not os.path.exists(basedir):
    os.mkdir(basedir)
cloth_obj = bpy.context.scene.objects[0]
bpy.ops.export_scene.obj(filepath=os.path.join(basedir + '%s.obj' % file.split('/')[-1]))
