import bpy
import bmesh
import random
import math
import mathutils
from mathutils import Vector, Quaternion

# ------------------------------------------------------------------------
# Core Operator
# ------------------------------------------------------------------------

class MESH_OT_GenerateBoxesOnFace(bpy.types.Operator):
    """Generate custom objects or boxes on the selected face"""
    bl_idname = "mesh.generate_boxes_on_face"
    bl_label = "Generate Objects"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH')

    def execute(self, context):
        props = context.scene.gen_props
        source_obj = props.source_object
        obj = context.edit_object
        
        # Safety check for zero scale
        if obj.matrix_world.determinant() == 0:
            self.report({'ERROR'}, "Active object has zero scale. Please apply scale.")
            return {'CANCELLED'}

        # Calculate size and local offset
        if source_obj:
            if source_obj.matrix_world.determinant() == 0:
                self.report({'ERROR'}, "Source object has zero scale.")
                return {'CANCELLED'}
            effective_size = max(source_obj.dimensions.x, source_obj.dimensions.y)
            final_scale = source_obj.scale.copy()
            bbox = [Vector(c) for c in source_obj.bound_box]
            source_local_offset = Vector((sum(v.x for v in bbox)/8, sum(v.y for v in bbox)/8, min(v.z for v in bbox)))
        else:
            effective_size = props.box_size
            final_scale = Vector((1, 1, 1))
            source_local_offset = Vector((0, 0, -effective_size / 2))

        spacing = props.min_distance + effective_size
        if spacing <= 0:
            self.report({'ERROR'}, "Spacing must be greater than zero")
            return {'CANCELLED'}
        
        bm = bmesh.from_edit_mesh(obj.data)
        selected_faces = [f for f in bm.faces if f.select]
        
        if len(selected_faces) != 1:
            self.report({'WARNING'}, "Select exactly 1 face")
            return {'CANCELLED'}
            
        face = selected_faces[0]
        if face.calc_area() <= 1e-7:
            self.report({'WARNING'}, "Selected face is too small or degenerate")
            return {'CANCELLED'}

        target_col = bpy.data.collections.get("synth") or bpy.data.collections.new("synth")
        if target_col.name not in context.scene.collection.children:
            context.scene.collection.children.link(target_col)

        mw = obj.matrix_world
        normal, center = face.normal.copy(), face.calc_center_median()
        
        # Coordinate system
        up = Vector((0,0,1)) if abs(normal.z) < 0.99 else Vector((1,0,0))
        tangent = normal.cross(up).normalized()
        bitangent = normal.cross(tangent).normalized()
        poly_verts = [Vector(((v.co - center).dot(tangent), (v.co - center).dot(bitangent))) for v in face.verts]

        proj_data = {'center': center, 'tangent': tangent, 'bitangent': bitangent, 'poly_verts': poly_verts}
        
        points = self.generate_random_points(
            face,
            props.box_count,
            spacing,
            effective_size,
            proj_data,
            allow_overhang=props.allow_overhang,
        ) if props.distribution_mode == 'RANDOM' else self.generate_grid_points(
            spacing,
            effective_size,
            proj_data,
            props.box_count,
            allow_overhang=props.allow_overhang,
        )

        if not points:
            self.report({'WARNING'}, "No space for objects")
            return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Ensure normal is valid for tracking
        if normal.length > 1e-6:
            align_quat = normal.to_track_quat('Z', 'Y')
        else:
            align_quat = Quaternion((1,0,0,0))
        
        mw_quat = mw.to_quaternion()
        
        for p in points:
            world_pos_on_face = mw @ p
            rot = (mw_quat @ align_quat).normalized()
            if props.random_rotation:
                rot @= Quaternion((0, 0, 1), random.uniform(0, 2 * math.pi))
            
            # Apply scale to offset
            scaled_offset = Vector((source_local_offset.x * final_scale.x, 
                                   source_local_offset.y * final_scale.y, 
                                   source_local_offset.z * final_scale.z))
            pos = world_pos_on_face - (rot @ scaled_offset)
            
            if source_obj:
                new_obj = source_obj.copy()
                new_obj.data = source_obj.data
                new_obj.location, new_obj.rotation_euler, new_obj.scale = pos, rot.to_euler(), final_scale
            else:
                bpy.ops.mesh.primitive_cube_add(size=effective_size, location=pos, rotation=rot.to_euler())
                new_obj = context.active_object
                [c.objects.unlink(new_obj) for c in new_obj.users_collection if c != target_col]
            
            if new_obj.name not in target_col.objects: target_col.objects.link(new_obj)

        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}
    def generate_random_points(self, face, count, min_dist, obj_size, proj_data, allow_overhang=False):
        tris = triangulate_face(face)
        if not tris: return []
        total_area, points = sum(t['area'] for t in tris), []
        center, tangent, bitangent, poly_verts = proj_data['center'], proj_data['tangent'], proj_data['bitangent'], proj_data['poly_verts']

        for _ in range(count * 100):
            if len(points) >= count: break
            r, acc = random.uniform(0, total_area), 0
            for t in tris:
                acc += t['area']
                if acc >= r:
                    p_3d = random_point_in_triangle(t['v1'], t['v2'], t['v3'])
                    if allow_overhang:
                        # Point is already sampled on the face; allow the object to overhang the face boundary.
                        if all((p_3d - ep).length >= min_dist for ep in points):
                            points.append(p_3d)
                    else:
                        pt_2d = Vector(((p_3d - center).dot(tangent), (p_3d - center).dot(bitangent)))
                        if is_box_fully_inside_2d(pt_2d, obj_size, poly_verts) and all((p_3d - ep).length >= min_dist for ep in points):
                            points.append(p_3d)
                    break
        return points

    def generate_grid_points(self, spacing, obj_size, proj_data, max_count=None, allow_overhang=False):
            points, poly_verts = [], proj_data['poly_verts']
            if not poly_verts: return []
            u_range = (min(p.x for p in poly_verts), max(p.x for p in poly_verts))
            v_range = (min(p.y for p in poly_verts), max(p.y for p in poly_verts))

            def accept(pt_2d: Vector) -> bool:
                if allow_overhang:
                    return is_point_inside_poly_2d(pt_2d, poly_verts)
                return is_box_fully_inside_2d(pt_2d, obj_size, poly_verts)

            # If spacing is larger than the face extent, a normal grid pass may yield zero.
            # In overhang mode, still try placing one object at the face center.
            if allow_overhang and (u_range[1] - u_range[0] <= spacing or v_range[1] - v_range[0] <= spacing):
                center_2d = Vector((0.0, 0.0))
                if accept(center_2d):
                    points.append(proj_data['center'])
                    return points if not max_count else points[:max_count]

            curr_v = v_range[1] - spacing / 2
            while curr_v > v_range[0]:
                curr_u = u_range[1] - spacing / 2
                while curr_u > u_range[0]:
                    if max_count and len(points) >= max_count: return points
                    pt_2d = Vector((curr_u, curr_v))
                    if accept(pt_2d):
                        points.append(proj_data['center'] + (curr_u * proj_data['tangent']) + (curr_v * proj_data['bitangent']))
                    curr_u -= spacing
                curr_v -= spacing
            return points


class MESH_OT_SubdivideFaceLongEdge(bpy.types.Operator):
    """Subdivide a quad along its long edge direction"""
    bl_idname = "mesh.subdivide_face_long_edge"
    bl_label = "Subdivide Long Edge"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def execute(self, context):
        props = context.scene.gen_props
        obj = context.edit_object
        bm = bmesh.from_edit_mesh(obj.data)

        selected_faces = [f for f in bm.faces if f.select]
        if len(selected_faces) != 1:
            self.report({'WARNING'}, "Select exactly 1 face")
            return {'CANCELLED'}

        face = selected_faces[0]

        target_faces = [face] if len(face.verts) == 4 else self.convert_to_quads(bm, face)
        if not target_faces:
            self.report({'WARNING'}, "Failed to convert face to quads")
            return {'CANCELLED'}

        long_edges = set()
        for f in target_faces:
            if len(f.verts) != 4:
                continue
            loops = f.loops
            edges = [l.edge for l in loops]
            lengths = [(e.verts[0].co - e.verts[1].co).length for e in edges]
            pair0 = lengths[0] + lengths[2]
            pair1 = lengths[1] + lengths[3]
            long_indices = (0, 2) if pair0 >= pair1 else (1, 3)
            long_edges.update({edges[long_indices[0]], edges[long_indices[1]]})

        if not long_edges:
            self.report({'WARNING'}, "No quad faces available after conversion")
            return {'CANCELLED'}

        segments = max(2, props.long_edge_segments)
        cuts = segments - 1

        bmesh.ops.subdivide_edges(bm, edges=list(long_edges), cuts=cuts, use_grid_fill=True)
        bmesh.update_edit_mesh(obj.data)

        self.report({'INFO'}, f"Subdivided {len(long_edges)//2} quad faces into {segments} segments along long edges")
        return {'FINISHED'}

    def convert_to_quads(self, bm, face):
        # Some Blender builds don't expose bmesh.ops.tris_convert_to_quads.
        # Prefer tris_convert_to_quads when available, otherwise fall back to join_triangles.
        face_center = face.calc_center_median()
        face_normal = face.normal.copy()
        src_verts = [v.co.copy() for v in face.verts]
        bb_min = Vector((min(v.x for v in src_verts), min(v.y for v in src_verts), min(v.z for v in src_verts)))
        bb_max = Vector((max(v.x for v in src_verts), max(v.y for v in src_verts), max(v.z for v in src_verts)))

        res_tri = bmesh.ops.triangulate(bm, faces=[face], quad_method='BEAUTY', ngon_method='BEAUTY')
        tri_faces = [f for f in res_tri.get('faces', []) if f.is_valid]
        if not tri_faces:
            return []

        face_threshold = math.radians(40)
        shape_threshold = math.radians(40)

        quad_faces = []
        if hasattr(bmesh.ops, "tris_convert_to_quads"):
            res_quads = bmesh.ops.tris_convert_to_quads(
                bm,
                faces=tri_faces,
                face_threshold=face_threshold,
                shape_threshold=shape_threshold
            )
            quad_faces = [f for f in res_quads.get('faces', []) if getattr(f, "is_valid", False) and len(f.verts) == 4]
        elif hasattr(bmesh.ops, "join_triangles"):
            # join_triangles merges adjacent triangles into quads when possible.
            # API names vary slightly across versions, so keep args conservative.
            res_quads = bmesh.ops.join_triangles(
                bm,
                faces=tri_faces,
                angle_face_threshold=face_threshold,
                angle_shape_threshold=shape_threshold,
            )

            # Try common result keys first.
            for key in ("faces", "new_faces", "result_faces"):
                faces = res_quads.get(key)
                if faces:
                    quad_faces = [f for f in faces if getattr(f, "is_valid", False) and hasattr(f, "verts") and len(f.verts) == 4]
                    if quad_faces:
                        break

        # Last-resort: pick quads produced in the same area (keeps behavior stable even if ops return no face lists).
        if not quad_faces:
            eps = max((bb_max - bb_min).length * 1e-4, 1e-6)
            bb_min2 = bb_min - Vector((eps, eps, eps))
            bb_max2 = bb_max + Vector((eps, eps, eps))

            def within_bbox(p):
                return (bb_min2.x <= p.x <= bb_max2.x and bb_min2.y <= p.y <= bb_max2.y and bb_min2.z <= p.z <= bb_max2.z)

            def normal_ok(n):
                return n.length > 1e-9 and face_normal.length > 1e-9 and n.normalized().dot(face_normal.normalized()) > 0.95

            quad_faces = [
                f for f in bm.faces
                if f.is_valid
                and len(f.verts) == 4
                and within_bbox(f.calc_center_median())
                and normal_ok(f.normal)
            ]

        return quad_faces


def is_point_inside_poly_2d(p, poly_verts):
    # Ray casting (even/odd) test in 2D.
    inside = False
    for i in range(len(poly_verts)):
        p1, p2 = poly_verts[i], poly_verts[(i + 1) % len(poly_verts)]
        if min(p1.y, p2.y) < p.y <= max(p1.y, p2.y) and p.x <= max(p1.x, p2.x):
            if p1.y != p2.y:
                xinters = (p.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
            if p1.x == p2.x or p.x <= xinters:
                inside = not inside
    return inside


def is_box_fully_inside_2d(center_pt, size, poly_verts):
    half = size / 2
    corners = [Vector((center_pt.x + h, center_pt.y + v)) for h in (half, -half) for v in (half, -half)]
    return all(is_point_inside_poly_2d(c, poly_verts) for c in corners)

def triangulate_face(face):
    verts = [v.co for v in face.verts]
    if len(verts) < 3: return []
    try:
        indices = mathutils.geometry.tessellate_polygon([[v.to_tuple() for v in verts]])
        return [{'v1': verts[i[0]], 'v2': verts[i[1]], 'v3': verts[i[2]], 
                 'area': 0.5 * ((verts[i[1]] - verts[i[0]]).cross(verts[i[2]] - verts[i[0]])).length} 
                for i in indices]
    except: return []

def random_point_in_triangle(v1, v2, v3):
    u, v = random.random(), random.random()
    if u + v > 1: u, v = 1 - u, 1 - v
    return u * v1 + v * v2 + (1 - u - v) * v3

# ------------------------------------------------------------------------
# UI Panel
# ------------------------------------------------------------------------

class GenProperties(bpy.types.PropertyGroup):
    distribution_mode: bpy.props.EnumProperty(
        name="Mode",
        items=[('RANDOM', "Random", ""), ('GRID', "Grid", "")],
        default='RANDOM'
    )
    source_object: bpy.props.PointerProperty(
        name="Source Object",
        type=bpy.types.Object,
        description="Select the object to generate. Leave empty to generate default cubes."
    )
    random_rotation: bpy.props.BoolProperty(
        name="Random Rotation",
        description="Randomly rotate the object's Z-axis",
        default=False
    )
    allow_overhang: bpy.props.BoolProperty(
        name="Allow Overhang",
        description="Allow generated objects to extend beyond the selected face boundary",
        default=False
    )
    long_edge_segments: bpy.props.IntProperty(
        name="Long Edge Segments",
        description="Number of segments along the long edge",
        default=3,
        min=2,
        max=100
    )
    
    box_count: bpy.props.IntProperty(name="Max Count", default=10, min=1, max=100)
    box_size: bpy.props.FloatProperty(name="Size", default=0.2, min=0.01)
    min_distance: bpy.props.FloatProperty(name="Gap", default=0.05, min=0.0)

class VIEW3D_PT_gen_panel(bpy.types.Panel):
    bl_space_type, bl_region_type, bl_category, bl_label = 'VIEW_3D', 'UI', "Synth Retail", "Generator Settings"

    def draw(self, context):
        layout, props = self.layout, context.scene.gen_props
        layout.prop(props, "distribution_mode", text="")
        layout.separator()
        
        row = layout.row()
        row.prop(props, "source_object", text="", icon='OBJECT_DATA')
        if props.source_object:
            layout.label(text=f"Size: {max(props.source_object.dimensions.x, props.source_object.dimensions.y):.2f}m", icon='CHECKMARK')
        else:
            layout.label(text="Default: Cube", icon='CUBE')

        layout.separator()
        col = layout.column(align=True)
        col.prop(props, "box_count")
        col.prop(props, "min_distance", text="Gap")
        if not props.source_object: col.prop(props, "box_size")
        col.prop(props, "random_rotation", toggle=True)
        col.prop(props, "allow_overhang", toggle=True)
        
        layout.separator()
        sub = layout.column(align=True)
        sub.prop(props, "long_edge_segments", text="Segments")
        sub.operator("mesh.subdivide_face_long_edge", icon='MOD_ARRAY')

        layout.separator()
        layout.scale_y = 1.0
        layout.operator("mesh.generate_boxes_on_face", text="Generate Objects" if props.source_object else "Generate Boxes", icon='PARTICLES')

# ------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------

classes = (MESH_OT_GenerateBoxesOnFace, MESH_OT_SubdivideFaceLongEdge, GenProperties, VIEW3D_PT_gen_panel)

def register():
    for cls in classes: bpy.utils.register_class(cls)
    bpy.types.Scene.gen_props = bpy.props.PointerProperty(type=GenProperties)

def unregister():
    for cls in reversed(classes): bpy.utils.unregister_class(cls)
    del bpy.types.Scene.gen_props


if __name__ == "__main__":
    register()