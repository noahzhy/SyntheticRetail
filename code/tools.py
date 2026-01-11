import bpy
import bmesh
import os
import json
import random
import math
import mathutils
from mathutils import Vector, Quaternion
from bpy_extras.object_utils import world_to_camera_view

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


class SCENE_OT_ExportSynthBBoxes(bpy.types.Operator):
    """Export 2D bounding boxes for all objects in the 'synth' collection.
    Bounding boxes are computed from frustum-clipped geometry.
    Occlusion is reported as a property, not used as a filter.
    """
    bl_idname = "scene.export_synth_bboxes"
    bl_label = "Export Synth BBoxes"
    bl_options = {'REGISTER', 'UNDO'}

    output_dir: bpy.props.StringProperty(
        name="Output Dir",
        description="Directory to save outputs (JSON + render JPG). Defaults to the blend folder.",
        default="//",
        subtype='DIR_PATH'
    )

    base_name: bpy.props.StringProperty(
        name="Base Name",
        description="Base filename (without extension) for outputs",
        default="synth",
    )

    output_path: bpy.props.StringProperty(
        name="Output JSON",
        description="Path to save bounding boxes JSON (defaults to blend folder)",
        default="//synth_bboxes.json",
        subtype='FILE_PATH'
    )

    export_render_jpg: bpy.props.BoolProperty(
        name="Export Render (JPG)",
        description="Also render the current scene and save as JPG into Output Dir",
        default=True,
    )

    min_visible_ratio: bpy.props.FloatProperty(
        name="Min Visible Ratio",
        description="Minimum visible proxy ratio based on clipped geometry",
        default=0.05,
        min=0.0,
        max=1.0,
    )

    min_bbox_area: bpy.props.FloatProperty(
        name="Min BBox Area",
        description="Discard extremely small boxes (normalized area)",
        default=1e-4,
        min=0.0,
        max=1.0,
    )

    @classmethod
    def poll(cls, context):
        return context.scene.camera is not None

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def execute(self, context):
        scene = context.scene
        cam = scene.camera
        depsgraph = context.evaluated_depsgraph_get()

        # Allow UI panel (Scene.export_props) to drive defaults.
        export_props = getattr(scene, "export_props", None)
        if export_props is not None:
            if self.output_dir in {"", "//"} and getattr(export_props, "output_dir", ""):
                self.output_dir = export_props.output_dir
            if self.base_name == "synth" and getattr(export_props, "base_name", ""):
                self.base_name = export_props.base_name
            if getattr(export_props, "export_render_jpg", None) is not None:
                self.export_render_jpg = export_props.export_render_jpg
            if getattr(export_props, "min_visible_ratio", None) is not None:
                self.min_visible_ratio = export_props.min_visible_ratio

        col = bpy.data.collections.get("synth")
        if not col:
            self.report({'WARNING'}, "Collection 'synth' not found")
            return {'CANCELLED'}

        results = []

        for obj in col.all_objects:
            if not self._is_mesh_renderable(obj):
                continue

            bbox = self._compute_visible_bbox(scene, cam, depsgraph, obj)
            if not bbox:
                continue

            xmin, ymin, xmax, ymax, visible_proxy, total_proxy, occluded = bbox

            # 只用 bbox 几何过滤
            area = (xmax - xmin) * (ymax - ymin)
            if area < self.min_bbox_area:
                continue

            results.append({
                "label": obj.name,
                "bbox": [xmin, ymin, xmax, ymax],
                "occluded": occluded,
                "visible_proxy": visible_proxy,
                "total_proxy": total_proxy,
                "visible_ratio": (
                    float(visible_proxy) / float(total_proxy)
                    if total_proxy > 0 else 0.0
                ),
            })

        if not results:
            self.report({'WARNING'}, "No valid visible objects in 'synth'")
            return {'CANCELLED'}

        out_dir = bpy.path.abspath(self.output_dir)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to create output dir: {exc}")
            return {'CANCELLED'}

        # Keep backward compatibility: if output_path is explicitly set to a non-default,
        # honor it; otherwise write to Output Dir with Base Name.
        default_out_path = bpy.path.abspath("//synth_bboxes.json")
        chosen_out_path = bpy.path.abspath(self.output_path)
        if chosen_out_path == default_out_path:
            out_path = os.path.join(out_dir, f"{self.base_name}.json")
        else:
            out_path = chosen_out_path

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to write JSON: {exc}")
            return {'CANCELLED'}

        # Render JPG to Output Dir
        jpg_path = os.path.join(out_dir, f"{self.base_name}.jpg")
        if self.export_render_jpg:
            prev_filepath = scene.render.filepath
            prev_format = scene.render.image_settings.file_format
            try:
                scene.render.filepath = jpg_path
                scene.render.image_settings.file_format = 'JPEG'
                bpy.ops.render.render(write_still=True)
            except Exception as exc:
                self.report({'ERROR'}, f"Failed to render JPG: {exc}")
                return {'CANCELLED'}
            finally:
                scene.render.filepath = prev_filepath
                scene.render.image_settings.file_format = prev_format

        if self.export_render_jpg:
            self.report({'INFO'}, f"Saved {len(results)} boxes to {out_path} and render to {jpg_path}")
        else:
            self.report({'INFO'}, f"Saved {len(results)} boxes to {out_path}")
        return {'FINISHED'}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _is_mesh_renderable(self, obj):
        return (
            obj.type == 'MESH'
            and obj.visible_get()
            and not obj.hide_render
        )

    # ------------------------------------------------------------------
    # Core bbox logic
    # ------------------------------------------------------------------

    def _compute_visible_bbox(self, scene, cam, depsgraph, obj):
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if not mesh:
            return None

        world_matrix = eval_obj.matrix_world
        cam_loc = cam.matrix_world.translation

        # --------------------------------------------------
        # Camera matrices
        # --------------------------------------------------
        render = scene.render
        res_x = max(1, int(round(render.resolution_x * render.resolution_percentage / 100)))
        res_y = max(1, int(round(render.resolution_y * render.resolution_percentage / 100)))

        proj_mat = cam.calc_matrix_camera(
            depsgraph,
            x=res_x,
            y=res_y,
            scale_x=render.pixel_aspect_x,
            scale_y=render.pixel_aspect_y,
        )
        view_mat = cam.matrix_world.inverted()
        clip_mat = proj_mat @ view_mat
        inv_clip_mat = clip_mat.inverted()

        # --------------------------------------------------
        # Frustum planes (clip space)
        # --------------------------------------------------
        planes = (
            Vector(( 1,  0,  0, 1)),
            Vector((-1,  0,  0, 1)),
            Vector(( 0,  1,  0, 1)),
            Vector(( 0, -1,  0, 1)),
            Vector(( 0,  0,  1, 1)),
            Vector(( 0,  0, -1, 1)),
        )

        def clip_poly(poly, plane):
            out = []
            prev = poly[-1]
            prev_d = plane.dot(prev)
            prev_in = prev_d >= 0
            for cur in poly:
                cur_d = plane.dot(cur)
                cur_in = cur_d >= 0
                if cur_in != prev_in:
                    t = prev_d / (prev_d - cur_d)
                    out.append(prev.lerp(cur, t))
                if cur_in:
                    out.append(cur)
                prev, prev_d, prev_in = cur, cur_d, cur_in
            return out

        def clip_frustum(poly):
            for pl in planes:
                poly = clip_poly(poly, pl)
                if not poly:
                    break
            return poly

        def is_point_visible(clip_pos):
            # Unproject clip space to world space
            world_homo = inv_clip_mat @ clip_pos
            if abs(world_homo.w) < 1e-6:
                return False
            world_pos = Vector((world_homo.x, world_homo.y, world_homo.z)) / world_homo.w
            
            ray_dir = world_pos - cam_loc
            ray_len = ray_dir.length
            if ray_len < 1e-4:
                return True

            hit, _, _, _, hit_obj, _ = scene.ray_cast(
                depsgraph,
                cam_loc,
                ray_dir.normalized(),
                distance=ray_len - 1e-4,
            )
            
            # If hit something that is NOT the object itself, it's occluded
            occluded = (
                hit and hit_obj and
                getattr(hit_obj, "original", hit_obj) != obj
            )
            return not occluded

        # --------------------------------------------------
        # Collect bbox + occlusion proxy
        # --------------------------------------------------
        frustum_pts = []
        visible_polys = 0
        occluded_polys = 0

        for poly in mesh.polygons:
            poly4 = []
            for vi in poly.vertices:
                wco = world_matrix @ mesh.vertices[vi].co
                poly4.append(clip_mat @ Vector((wco.x, wco.y, wco.z, 1.0)))

            clipped = clip_frustum(poly4)
            if not clipped:
                continue

            # ---------- face-center ray occlusion ----------
            center = Vector()
            count = 0
            for p4 in clipped:
                if abs(p4.w) < 1e-6:
                    continue
                ndc = Vector((p4.x / p4.w, p4.y / p4.w, p4.z / p4.w))
                center += ndc
                count += 1

            if count == 0:
                continue

            center /= count
            
            # Check center visibility
            center_clip = Vector((center.x, center.y, center.z, 1.0))
            if not is_point_visible(center_clip):
                occluded_polys += 1
            else:
                visible_polys += 1
                # Only collect bbox points from visible vertices of visible polygons
                for p4 in clipped:
                    if abs(p4.w) < 1e-6:
                        continue
                    
                    # Strict check: verify each vertex is visible
                    if is_point_visible(p4):
                        u = (p4.x / p4.w + 1.0) * 0.5
                        v = (p4.y / p4.w + 1.0) * 0.5
                        frustum_pts.append((u, v))

        total_polys = visible_polys + occluded_polys
        eval_obj.to_mesh_clear()

        if not frustum_pts or total_polys == 0:
            return None

        xs = [p[0] for p in frustum_pts]
        ys = [p[1] for p in frustum_pts]

        xmin = max(0.0, min(xs))
        xmax = min(1.0, max(xs))
        ymin_b = max(0.0, min(ys))
        ymax_b = min(1.0, max(ys))

        ymin = 1.0 - ymax_b
        ymax = 1.0 - ymin_b

        visible_ratio = visible_polys / total_polys
        occluded = visible_ratio < self.min_visible_ratio

        return xmin, ymin, xmax, ymax, visible_polys, total_polys, occluded


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


class ExportProperties(bpy.types.PropertyGroup):
    output_dir: bpy.props.StringProperty(
        name="Output Dir",
        description="Directory to save outputs (JSON + render JPG). Defaults to the blend folder.",
        default="//",
        subtype='DIR_PATH'
    )
    base_name: bpy.props.StringProperty(
        name="Base Name",
        description="Base filename (without extension) for outputs",
        default="synth",
    )
    export_render_jpg: bpy.props.BoolProperty(
        name="Export Render (JPG)",
        description="Also render the current scene and save as JPG into Output Dir",
        default=True,
    )

    min_visible_ratio: bpy.props.FloatProperty(
        name="Min Visible Ratio",
        description="Export only objects with visible_proxy/total_proxy >= this threshold",
        default=0.05,
        min=0.0,
        max=1.0,
    )


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

        layout.separator()
        export_props = context.scene.export_props
        box = layout.box()
        box.label(text="Export", icon='EXPORT')
        box.prop(export_props, "output_dir")
        box.prop(export_props, "base_name")
        box.prop(export_props, "export_render_jpg")
        box.prop(export_props, "min_visible_ratio")
        box.operator("scene.export_synth_bboxes", icon='IMAGE_DATA')

# ------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------

classes = (
    MESH_OT_GenerateBoxesOnFace,
    MESH_OT_SubdivideFaceLongEdge,
    SCENE_OT_ExportSynthBBoxes,
    GenProperties,
    ExportProperties,
    VIEW3D_PT_gen_panel,
)

def register():
    for cls in classes: bpy.utils.register_class(cls)
    bpy.types.Scene.gen_props = bpy.props.PointerProperty(type=GenProperties)
    bpy.types.Scene.export_props = bpy.props.PointerProperty(type=ExportProperties)

def unregister():
    for cls in reversed(classes): bpy.utils.unregister_class(cls)
    del bpy.types.Scene.gen_props
    del bpy.types.Scene.export_props


if __name__ == "__main__":
    register()