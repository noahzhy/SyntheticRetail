import bpy
import mathutils

obj = bpy.context.active_object

if obj is None:
    raise RuntimeError("No active object selected")

if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

bbox_world = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
min_z = min(v.z for v in bbox_world)
center_x = sum(v.x for v in bbox_world) / 8
center_y = sum(v.y for v in bbox_world) / 8
bottom_center = mathutils.Vector((center_x, center_y, min_z))
bpy.context.scene.cursor.location = bottom_center
bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
print(f"Origin set to bottom center: {bottom_center}")
