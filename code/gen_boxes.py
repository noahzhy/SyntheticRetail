import os
import json
import math
import random
import time
import bpy
from mathutils import Vector

# ------------------------------------------------------------------------
# Core Operator
# ------------------------------------------------------------------------

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
        target_col = None
        target_cols = []
        # 检查是否从UI调用（通过检查参数是否为默认值）
        called_from_ui = (
            self.output_dir == "//" and 
            self.base_name == "synth" and 
            self.output_path == "//synth_bboxes.json"
        )
        
        if export_props is not None:
            if self.output_dir in {"", "//"} and getattr(export_props, "output_dir", ""):
                self.output_dir = export_props.output_dir
            if self.base_name == "synth" and getattr(export_props, "base_name", ""):
                self.base_name = export_props.base_name
            # 只在从UI调用时才从export_props读取export_render_jpg
            if called_from_ui and getattr(export_props, "export_render_jpg", None) is not None:
                self.export_render_jpg = export_props.export_render_jpg
            if getattr(export_props, "min_visible_ratio", None) is not None:
                self.min_visible_ratio = export_props.min_visible_ratio
            target_col = getattr(export_props, "target_collection", None)
            if getattr(export_props, "target_collections", None):
                for item in export_props.target_collections:
                    col_item = getattr(item, "collection", None)
                    if col_item is not None:
                        target_cols.append(col_item)

        if not target_cols and target_col is not None:
            target_cols = [target_col]

        if not target_cols:
            # 默认使用所有以 SM_ 开头的集合
            target_cols = [col for col in bpy.data.collections if col.name.startswith("SM_")]

        if not target_cols:
            default_col = bpy.data.collections.get("synth")
            if default_col is not None:
                target_cols = [default_col]

        if not target_cols:
            self.report({'WARNING'}, "Target collection not found")
            return {'CANCELLED'}

        results = []

        # Front-row visibility filtering
        front_row_only = False
        front_row_margin = 0.1
        shelf_obj = None
        if export_props is not None:
            front_row_only = getattr(export_props, "front_row_only", False)
            front_row_margin = getattr(export_props, "front_row_margin", 0.1)
            shelf_obj = getattr(export_props, "shelf_object", None)

        candidate_obj_set = set()
        for col in target_cols:
            for obj in col.all_objects:
                if self._is_mesh_renderable(obj):
                    candidate_obj_set.add(obj)
        candidate_objs = list(candidate_obj_set)
        if not candidate_objs:
            col_names = ", ".join(col.name for col in target_cols)
            self.report({'WARNING'}, f"No renderable objects in '{col_names}'")
            return {'CANCELLED'}

        front_row_set = None
        if front_row_only:
            cam_loc = cam.matrix_world.translation

            if shelf_obj is not None:
                shelf_origin = shelf_obj.matrix_world.translation
                shelf_depth_axis = (shelf_obj.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()

                def is_probe_visible(probe_pos):
                    ray_dir = probe_pos - cam_loc
                    ray_len = ray_dir.length
                    if ray_len < 1e-4:
                        return True
                    hit, _, _, _, _, _ = scene.ray_cast(
                        depsgraph,
                        cam_loc,
                        ray_dir.normalized(),
                        distance=ray_len - 1e-4,
                    )
                    return not hit

                # Determine outer side by visibility test of two probes along shelf depth axis
                shelf_eval = shelf_obj.evaluated_get(depsgraph)
                shelf_bb = [shelf_eval.matrix_world @ Vector(corner) for corner in shelf_eval.bound_box]
                shelf_diag = max((shelf_bb[i] - shelf_bb[j]).length for i in range(len(shelf_bb)) for j in range(i + 1, len(shelf_bb)))
                probe_dist = max(shelf_diag * 0.25, 0.05)
                probe_pos_a = shelf_origin + shelf_depth_axis * probe_dist
                probe_pos_b = shelf_origin - shelf_depth_axis * probe_dist
                vis_a = is_probe_visible(probe_pos_a)
                vis_b = is_probe_visible(probe_pos_b)

                if vis_a and not vis_b:
                    shelf_outer_dir = shelf_depth_axis
                elif vis_b and not vis_a:
                    shelf_outer_dir = -shelf_depth_axis
                else:
                    # Fallback: choose the side more towards the camera
                    to_cam = cam_loc - shelf_origin
                    shelf_outer_dir = shelf_depth_axis if to_cam.dot(shelf_depth_axis) >= 0.0 else -shelf_depth_axis

                def min_depth_for_obj(o):
                    eval_o = o.evaluated_get(depsgraph)
                    wm = eval_o.matrix_world
                    bb = [wm @ Vector(corner) for corner in eval_o.bound_box]
                    return min((p - shelf_origin).dot(shelf_outer_dir) for p in bb)

                def is_facing_camera(o):
                    obj_forward = (o.matrix_world.to_3x3() @ Vector((0.0, -1.0, 0.0))).normalized()
                    to_cam = (cam_loc - o.matrix_world.translation).normalized()
                    return obj_forward.dot(to_cam) > 0.0

                depths = [(obj, min_depth_for_obj(obj)) for obj in candidate_objs if is_facing_camera(obj)]
                if depths:
                    min_depth = min(d for _, d in depths)
                    threshold = min_depth + max(front_row_margin, 0.0)
                    front_row_set = {obj for obj, d in depths if d <= threshold}
                else:
                    front_row_set = set()
            else:
                view_dir = (cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()

                def min_depth_for_obj(o):
                    eval_o = o.evaluated_get(depsgraph)
                    wm = eval_o.matrix_world
                    bb = [wm @ Vector(corner) for corner in eval_o.bound_box]
                    return min((p - cam_loc).dot(view_dir) for p in bb)

                depths = [(obj, min_depth_for_obj(obj)) for obj in candidate_objs]
                min_depth = min(d for _, d in depths)
                threshold = min_depth + max(front_row_margin, 0.0)
                front_row_set = {obj for obj, d in depths if d <= threshold}

        for obj in candidate_objs:
            if front_row_set is not None and obj not in front_row_set:
                continue

            # 快速视锥剔除检查，跳过相机完全看不见的物体
            if not self._is_in_camera_frustum(scene, cam, depsgraph, obj):
                continue

            bbox = self._compute_visible_bbox(scene, cam, depsgraph, obj)
            if not bbox:
                continue

            xmin, ymin, xmax, ymax, visible_proxy, total_proxy, occluded = bbox

            # 只用 bbox 几何过滤
            area = (xmax - xmin) * (ymax - ymin)
            if area < self.min_bbox_area:
                continue

            # 跳过被遮挡的物体
            if occluded:
                continue

            results.append({
                "label": obj.name,
                "bbox": [xmin, ymin, xmax, ymax],
                "visible_proxy": visible_proxy,
                "total_proxy": total_proxy,
                "visible_ratio": (
                    float(visible_proxy) / float(total_proxy)
                    if total_proxy > 0 else 0.0
                ),
            })

        if not results:
            col_names = ", ".join(col.name for col in target_cols)
            self.report({'WARNING'}, f"No valid visible objects in '{col_names}'")
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
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if chosen_out_path == default_out_path:
            out_path = os.path.join(out_dir, f"{self.base_name}_{timestamp}.json")
        else:
            out_path = chosen_out_path

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to write JSON: {exc}")
            return {'CANCELLED'}

        # Render JPG to Output Dir
        jpg_path = os.path.join(out_dir, f"{self.base_name}_{timestamp}.jpg")
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

    def _is_in_camera_frustum(self, scene, cam, depsgraph, obj):
        """快速检查物体是否在相机视锥内（粗略检查）"""
        eval_obj = obj.evaluated_get(depsgraph)
        if not eval_obj:
            return False
        
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
        
        world_matrix = eval_obj.matrix_world
        
        # 检查物体的包围盒角点是否有任何一个在视锥内或附近
        # 使用稍大的范围以避免边缘情况的误判
        tolerance = 0.1
        in_frustum = False
        
        for corner in eval_obj.bound_box:
            world_co = world_matrix @ Vector(corner)
            clip_co = clip_mat @ Vector((world_co.x, world_co.y, world_co.z, 1.0))
            
            if abs(clip_co.w) < 1e-6:
                continue
            
            # NDC坐标
            ndc_x = clip_co.x / clip_co.w
            ndc_y = clip_co.y / clip_co.w
            ndc_z = clip_co.z / clip_co.w
            
            # 检查是否在视锥内（含容差）
            if (-1.0 - tolerance <= ndc_x <= 1.0 + tolerance and
                -1.0 - tolerance <= ndc_y <= 1.0 + tolerance and
                -1.0 <= ndc_z <= 1.0):
                in_frustum = True
                break
        
        return in_frustum

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

        def is_front_facing(poly, normal_mat):
            world_center = world_matrix @ poly.center
            world_normal = (normal_mat @ poly.normal).normalized()
            view_dir = (cam_loc - world_center)
            if view_dir.length <= 1e-6:
                return False
            return world_normal.dot(view_dir) > 0.0

        def get_clipped_poly(poly):
            poly4 = []
            for vi in poly.vertices:
                wco = world_matrix @ mesh.vertices[vi].co
                poly4.append(clip_mat @ Vector((wco.x, wco.y, wco.z, 1.0)))
            return clip_frustum(poly4)

        def poly_center_clip(clipped):
            center = Vector()
            count = 0
            for p4 in clipped:
                if abs(p4.w) < 1e-6:
                    continue
                center += Vector((p4.x / p4.w, p4.y / p4.w, p4.z / p4.w))
                count += 1
            if count == 0:
                return None
            center /= count
            return Vector((center.x, center.y, center.z, 1.0))

        def collect_bbox_points(clipped, out_pts):
            for p4 in clipped:
                if abs(p4.w) < 1e-6:
                    continue
                out_pts.append(((p4.x / p4.w + 1.0) * 0.5, (p4.y / p4.w + 1.0) * 0.5))

        def poly_area_ndc(clipped):
            pts = []
            for p4 in clipped:
                if abs(p4.w) < 1e-6:
                    continue
                pts.append((p4.x / p4.w, p4.y / p4.w))
            if len(pts) < 3:
                return 0.0
            area = 0.0
            for i in range(len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % len(pts)]
                area += x1 * y2 - x2 * y1
            return abs(area) * 0.5

        # --------------------------------------------------
        # Collect bbox + occlusion proxy
        # --------------------------------------------------
        world_normal_mat = world_matrix.to_3x3()

        def compute_bbox(include_backfaces=False):
            frustum_pts = []
            visible_polys = 0
            occluded_polys = 0
            front_polys = 0
            front_area = 0.0
            visible_area = 0.0

            for poly in mesh.polygons:
                if not include_backfaces and not is_front_facing(poly, world_normal_mat):
                    continue
                front_polys += 1

                clipped = get_clipped_poly(poly)
                if not clipped:
                    continue

                poly_area = poly_area_ndc(clipped)
                if poly_area <= 0.0:
                    continue
                front_area += poly_area

                center_clip = poly_center_clip(clipped)
                if center_clip is None:
                    continue

                if not is_point_visible(center_clip):
                    occluded_polys += 1
                    continue

                visible_polys += 1
                visible_area += poly_area
                collect_bbox_points(clipped, frustum_pts)

            total_polys = max(1, visible_polys + occluded_polys)

            if not frustum_pts or front_polys == 0 or front_area <= 0.0:
                return None

            xs = [p[0] for p in frustum_pts]
            ys = [p[1] for p in frustum_pts]

            xmin = max(0.0, min(xs))
            xmax = min(1.0, max(xs))
            ymin_b = max(0.0, min(ys))
            ymax_b = min(1.0, max(ys))

            ymin = 1.0 - ymax_b
            ymax = 1.0 - ymin_b

            visible_ratio = visible_area / front_area
            occluded = visible_ratio < self.min_visible_ratio

            return xmin, ymin, xmax, ymax, visible_polys, total_polys, occluded

        result = compute_bbox(include_backfaces=False)
        if result is None:
            # Fallback for single-sided meshes facing away from the camera
            result = compute_bbox(include_backfaces=True)

        eval_obj.to_mesh_clear()
        return result


class SCENE_OT_CreateShelfCameraSweep(bpy.types.Operator):
    """Create a W/U/S sweep animation facing the shelf front"""
    bl_idname = "scene.create_shelf_camera_sweep"
    bl_label = "Create Shelf Camera Sweep"
    bl_options = {'REGISTER', 'UNDO'}

    def _compute_shelf_outer_dir(self, scene, depsgraph, shelf_obj, cam_loc):
        shelf_origin = shelf_obj.matrix_world.translation
        shelf_depth_axis = (shelf_obj.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()

        def is_probe_visible(probe_pos):
            ray_dir = probe_pos - cam_loc
            ray_len = ray_dir.length
            if ray_len < 1e-4:
                return True
            hit, _, _, _, _, _ = scene.ray_cast(
                depsgraph,
                cam_loc,
                ray_dir.normalized(),
                distance=ray_len - 1e-4,
            )
            return not hit

        shelf_eval = shelf_obj.evaluated_get(depsgraph)
        shelf_bb = [shelf_eval.matrix_world @ Vector(corner) for corner in shelf_eval.bound_box]
        shelf_diag = max((shelf_bb[i] - shelf_bb[j]).length for i in range(len(shelf_bb)) for j in range(i + 1, len(shelf_bb)))
        probe_dist = max(shelf_diag * 0.25, 0.05)
        probe_pos_a = shelf_origin + shelf_depth_axis * probe_dist
        probe_pos_b = shelf_origin - shelf_depth_axis * probe_dist
        vis_a = is_probe_visible(probe_pos_a)
        vis_b = is_probe_visible(probe_pos_b)

        if vis_a and not vis_b:
            return shelf_depth_axis
        if vis_b and not vis_a:
            return -shelf_depth_axis
        to_cam = cam_loc - shelf_origin
        return shelf_depth_axis if to_cam.dot(shelf_depth_axis) >= 0.0 else -shelf_depth_axis

    def execute(self, context):
        scene = context.scene
        export_props = getattr(scene, "export_props", None)
        if export_props is None:
            self.report({'ERROR'}, "Export properties not found")
            return {'CANCELLED'}

        cam = export_props.sweep_camera or scene.camera
        shelf_obj = getattr(export_props, "shelf_object", None)
        if cam is None or cam.type != 'CAMERA':
            self.report({'ERROR'}, "请选择相机")
            return {'CANCELLED'}
        if shelf_obj is None:
            self.report({'ERROR'}, "请选择货架对象")
            return {'CANCELLED'}

        frames = max(2, int(export_props.sweep_frames))
        start_frame = scene.frame_start
        end_frame = start_frame + frames - 1

        depsgraph = context.evaluated_depsgraph_get()
        center = shelf_obj.matrix_world.translation
        cam_loc = cam.matrix_world.translation

        # 获取货架上的商品列表（从SM_开头的集合）
        product_objs = []
        for col_name in bpy.data.collections:
            if col_name.name.startswith("SM_"):
                for obj in col_name.all_objects:
                    if obj.type == 'MESH' and obj.visible_get():
                        product_objs.append(obj)
        
        # 如果有商品，按用户设置的间隔随机选择一个作为目标
        target_product_frames = {}
        if product_objs:
            frame_interval = max(1, int(export_props.sweep_target_interval))
            for i in range(0, frames, frame_interval):
                if random.random() > 0.3:  # 70%的几率选择商品
                    target_product_frames[start_frame + i] = random.choice(product_objs)

        shelf_matrix_3x3 = shelf_obj.matrix_world.to_3x3()
        local_x_world = shelf_matrix_3x3 @ Vector((1, 0, 0))
        local_y_world = shelf_matrix_3x3 @ Vector((0, 1, 0))
        local_x_world.z = 0
        local_y_world.z = 0
        local_x_world = local_x_world.normalized()
        local_y_world = local_y_world.normalized()
        is_local_x_long = shelf_obj.dimensions.x >= shelf_obj.dimensions.y
        length_axis = local_x_world if is_local_x_long else local_y_world

        outer_dir = self._compute_shelf_outer_dir(scene, depsgraph, shelf_obj, cam_loc)
        outer_dir.z = 0
        if outer_dir.length < 1e-6:
            outer_dir = length_axis.cross(Vector((0, 0, 1)))
        outer_dir.normalize()

        shelf_bb = [shelf_obj.matrix_world @ Vector(corner) for corner in shelf_obj.bound_box]
        shelf_top_z = max(p.z for p in shelf_bb)
        shelf_bottom_z = min(p.z for p in shelf_bb)
        shelf_height = shelf_top_z - shelf_bottom_z
        shelf_mid_z = (shelf_top_z + shelf_bottom_z) * 0.5

        height = cam_loc.z - center.z
        if abs(height) < 1e-4:
            height = max(0.1, shelf_height * 0.5)

        min_dist = max(0.05, float(export_props.sweep_min_distance))
        max_dist = max(min_dist, float(export_props.sweep_max_distance))

        span = max(shelf_obj.dimensions.x, shelf_obj.dimensions.y) * 0.6
        pattern = export_props.sweep_pattern
        span = max(shelf_obj.dimensions.x, shelf_obj.dimensions.y) * 0.6

        # Ensure camera tracks a moving target on shelf front
        target_name = "SM_CamSweepTarget"
        target = bpy.data.objects.get(target_name)
        if target is None:
            target = bpy.data.objects.new(target_name, None)
            target.empty_display_type = 'PLAIN_AXES'
            scene.collection.objects.link(target)

        track = cam.constraints.get("SM_TrackToSweepTarget")
        if track is None:
            track = cam.constraints.new(type='TRACK_TO')
            track.name = "SM_TrackToSweepTarget"
        track.target = target
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'

        for i in range(frames):
            frame = start_frame + i
            t = 0.0 if frames <= 1 else i / (frames - 1)

            if pattern == 'W':
                wave = math.sin(4.0 * math.pi * t - math.pi / 2.0)
            elif pattern == 'S':
                wave = math.sin(2.0 * math.pi * t - math.pi / 2.0)
            else:  # 'U'
                wave = math.cos(2.0 * math.pi * t)

            # 根据垂直扫掠开关选择不同路径
            if export_props.sweep_vertical:
                # 垂直扫掠：8字形轨迹，使用Lissajous曲线模拟自然扫描货架商品
                # 水平方向: sin(t)
                lateral = math.sin(2.0 * math.pi * t) * span
                
                # 垂直方向: sin(2*t) - 创造8字形
                vertical_wave = math.sin(4.0 * math.pi * t)
                cam_z = shelf_mid_z + (vertical_wave * shelf_height * 0.4)
                target_z = cam_z
                
                # 前后距离：使用波形在min_dist和max_dist之间变化
                dist = min_dist + (max_dist - min_dist) * (wave * 0.5 + 0.5)
                
                # 相机位置：8字形移动 + 前后距离变化
                loc = center + length_axis * lateral + outer_dir * dist + Vector((0, 0, cam_z))
                
                # 目标点：在货架正面，跟随8字形移动
                target_loc = Vector((
                    center.x + length_axis.x * lateral,
                    center.y + length_axis.y * lateral,
                    target_z,
                ))
            else:
                # 水平扫掠：左右移动 + 前后距离变化（W/U/S波形）
                # 水平位置：t从0到1，从左到右扫过
                lateral = (t * 2.0 - 1.0) * span
                
                # 前后距离：使用波形在min_dist和max_dist之间变化
                dist = min_dist + (max_dist - min_dist) * (wave * 0.5 + 0.5)
                
                # 相机位置：在货架正面，随左右移动，前后距离变化
                loc = center + length_axis * lateral + outer_dir * dist + Vector((0, 0, height))
                
                # 目标点：在货架正面，随左右移动同步
                target_loc = Vector((
                    center.x + length_axis.x * lateral,
                    center.y + length_axis.y * lateral,
                    shelf_mid_z,
                ))
            cam.location = loc
            cam.keyframe_insert(data_path="location", frame=frame)
            
            # 如果当前帧有指定的目标商品，使用商品位置作为目标点
            if frame in target_product_frames:
                product = target_product_frames[frame]
                target_loc = product.matrix_world.translation.copy()
            
            target.location = target_loc
            target.keyframe_insert(data_path="location", frame=frame)

        scene.camera = cam
        scene.frame_end = max(scene.frame_end, end_frame)
        self.report({'INFO'}, f"已生成相机扫掠轨迹: {frames} 帧")
        return {'FINISHED'}


class SCENE_OT_RenderAnimationSequence(bpy.types.Operator):
    """Render the entire animation sequence and export bboxes for each frame"""
    bl_idname = "scene.render_animation_sequence"
    bl_label = "Render Animation Sequence"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        export_props = getattr(scene, "export_props", None)
        if export_props is None:
            self.report({'ERROR'}, "Export properties not found")
            return {'CANCELLED'}

        start_frame = scene.frame_start
        end_frame = scene.frame_end
        total_frames = end_frame - start_frame + 1

        if total_frames <= 0:
            self.report({'ERROR'}, "Invalid frame range")
            return {'CANCELLED'}

        # 保存原始设置
        original_frame = scene.frame_current
        original_render_path = scene.render.filepath
        original_render_format = scene.render.image_settings.file_format

        out_dir = bpy.path.abspath(export_props.output_dir or "//")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to create output dir: {exc}")
            return {'CANCELLED'}

        # 为这次序列渲染创建一个带时间戳的子目录
        sequence_timestamp = time.strftime("%Y%m%d_%H%M%S")
        sequence_dir = os.path.join(out_dir, f"sequence_{sequence_timestamp}")
        try:
            os.makedirs(sequence_dir, exist_ok=True)
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to create sequence dir: {exc}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"开始渲染动画序列: {total_frames} 帧")

        # 逐帧渲染和导出
        success_count = 0
        for frame in range(start_frame, end_frame + 1):
            scene.frame_set(frame)
            
            # 生成当前帧的文件名（图片保留时间戳）
            frame_basename = f"frame_{frame:04d}_{sequence_timestamp}"
            frame_jpg = os.path.join(sequence_dir, f"{frame_basename}.jpg")
            frame_json = os.path.join(sequence_dir, f"{frame_basename}.json")

            # 渲染当前帧
            try:
                scene.render.filepath = frame_jpg
                scene.render.image_settings.file_format = 'JPEG'
                bpy.ops.render.render(write_still=True)
            except Exception as exc:
                self.report({'WARNING'}, f"Failed to render frame {frame}: {exc}")
                continue

            # 导出当前帧的bbox
            try:
                # 获取目标集合
                target_cols = []
                target_col = getattr(export_props, "target_collection", None)
                if getattr(export_props, "target_collections", None):
                    for item in export_props.target_collections:
                        col_item = getattr(item, "collection", None)
                        if col_item is not None:
                            target_cols.append(col_item)
                
                if not target_cols and target_col is not None:
                    target_cols = [target_col]
                
                if not target_cols:
                    # 默认使用所有以 SM_ 开头的集合
                    target_cols = [col for col in bpy.data.collections if col.name.startswith("SM_")]
                
                if not target_cols:
                    default_col = bpy.data.collections.get("synth")
                    if default_col is not None:
                        target_cols = [default_col]
                
                if target_cols:
                    # 使用operator执行导出（通过bpy.ops调用）
                    bpy.ops.scene.export_synth_bboxes(
                        output_dir=sequence_dir,
                        base_name=frame_basename,
                        output_path=frame_json,
                        export_render_jpg=False,
                    )
                    success_count += 1
                else:
                    self.report({'WARNING'}, f"No target collections found for frame {frame}")
            except Exception as exc:
                self.report({'WARNING'}, f"Failed to export bbox for frame {frame}: {exc}")

        # 恢复原始设置
        scene.frame_set(original_frame)
        scene.render.filepath = original_render_path
        scene.render.image_settings.file_format = original_render_format

        self.report({'INFO'}, f"动画序列渲染完成: {success_count}/{total_frames} 帧成功")
        return {'FINISHED'}


class CollectionListItem(bpy.types.PropertyGroup):
    collection: bpy.props.PointerProperty(
        name="Collection",
        type=bpy.types.Collection,
        description="Collection to export"
    )


class ExportProperties(bpy.types.PropertyGroup):
    target_collection: bpy.props.PointerProperty(
        name="Collection",
        type=bpy.types.Collection,
        description="Select a collection to export. Defaults to 'synth' if empty."
    )
    target_collections: bpy.props.CollectionProperty(
        name="Collections",
        type=CollectionListItem,
        description="Collections to export"
    )
    target_collections_index: bpy.props.IntProperty(
        name="Collection Index",
        default=0
    )
    shelf_object: bpy.props.PointerProperty(
        name="Shelf",
        type=bpy.types.Object,
        description="Optional shelf object to define front-row direction (local -Y)"
    )
    front_row_only: bpy.props.BoolProperty(
        name="Front Row Only",
        description="Only export objects in the outer front row",
        default=False,
    )
    front_row_margin: bpy.props.FloatProperty(
        name="Front Row Margin",
        description="Depth tolerance (meters) for front-row selection",
        default=0.1,
        min=0.0,
    )
    sweep_camera: bpy.props.PointerProperty(
        name="Sweep Camera",
        type=bpy.types.Object,
        description="Camera used for shelf sweep animation"
    )
    sweep_frames: bpy.props.IntProperty(
        name="Sweep Frames",
        description="Total frames for the sweep animation",
        default=120,
        min=2,
        max=10000,
    )
    sweep_pattern: bpy.props.EnumProperty(
        name="Sweep Pattern",
        description="W / U / S sweep shape",
        items=[
            ('W', "W", "Double wave sweep"),
            ('U', "U", "Single valley sweep"),
            ('S', "S", "Single wave sweep"),
        ],
        default='W',
    )
    sweep_vertical: bpy.props.BoolProperty(
        name="Vertical Sweep",
        description="If enabled, sweeps vertically (top to bottom). If disabled, sweeps horizontally (left to right)",
        default=False,
    )
    sweep_min_distance: bpy.props.FloatProperty(
        name="Min Distance",
        description="Minimum distance to shelf front",
        default=0.6,
        min=0.01,
        max=100.0,
    )
    sweep_max_distance: bpy.props.FloatProperty(
        name="Max Distance",
        description="Maximum distance to shelf front",
        default=1.5,
        min=0.01,
        max=100.0,
    )
    sweep_target_interval: bpy.props.IntProperty(
        name="Target Product Interval",
        description="Interval (in frames) to randomly select a product as camera target",
        default=10,
        min=1,
        max=100,
    )
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


class SCENE_UL_CollectionList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "collection", text="", emboss=False, icon='OUTLINER_COLLECTION')
        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            layout.label(text="", icon='OUTLINER_COLLECTION')


class SCENE_OT_CollectionListAdd(bpy.types.Operator):
    bl_idname = "scene.collection_list_add"
    bl_label = "Add Collection"

    def execute(self, context):
        export_props = context.scene.export_props
        item = export_props.target_collections.add()
        if context.collection is not None:
            item.collection = context.collection
        export_props.target_collections_index = len(export_props.target_collections) - 1
        return {'FINISHED'}


class SCENE_OT_CollectionListRemove(bpy.types.Operator):
    bl_idname = "scene.collection_list_remove"
    bl_label = "Remove Collection"

    def execute(self, context):
        export_props = context.scene.export_props
        idx = export_props.target_collections_index
        if 0 <= idx < len(export_props.target_collections):
            export_props.target_collections.remove(idx)
            export_props.target_collections_index = max(0, idx - 1)
        return {'FINISHED'}


class VIEW3D_PT_gen_panel(bpy.types.Panel):
    bl_space_type, bl_region_type, bl_category, bl_label = 'VIEW_3D', 'UI', "Synth Retail", "Generator Settings"

    def draw(self, context):
        layout = self.layout
        export_props = context.scene.export_props
        box = layout.box()
        box.label(text="Export", icon='EXPORT')
        box.label(text="Collections")
        row = box.row()
        row.template_list(
            "SCENE_UL_CollectionList",
            "",
            export_props,
            "target_collections",
            export_props,
            "target_collections_index",
            rows=3,
        )
        col = row.column(align=True)
        col.operator("scene.collection_list_add", icon='ADD', text="")
        col.operator("scene.collection_list_remove", icon='REMOVE', text="")
        box.prop(export_props, "target_collection")
        box.prop(export_props, "shelf_object")
        box.prop(export_props, "front_row_only")
        row = box.row()
        row.enabled = export_props.front_row_only
        row.prop(export_props, "front_row_margin")
        box.prop(export_props, "output_dir")
        box.prop(export_props, "base_name")
        box.prop(export_props, "export_render_jpg")
        box.prop(export_props, "min_visible_ratio")
        box.operator("scene.export_synth_bboxes", icon='IMAGE_DATA')

        cam_box = layout.box()
        cam_box.label(text="Camera Sweep", icon='CAMERA_DATA')
        cam_box.prop(export_props, "sweep_camera")
        cam_box.prop(export_props, "sweep_frames")
        cam_box.prop(export_props, "sweep_vertical")
        cam_box.prop(export_props, "sweep_pattern")
        row = cam_box.row(align=True)
        row.prop(export_props, "sweep_min_distance")
        row.prop(export_props, "sweep_max_distance")
        cam_box.prop(export_props, "sweep_target_interval")
        cam_box.operator("scene.create_shelf_camera_sweep", icon='CON_TRACKTO')

        render_box = layout.box()
        render_box.label(text="Animation Sequence", icon='RENDER_ANIMATION')
        render_box.operator("scene.render_animation_sequence", icon='PLAY')

# ------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------

classes = (
    SCENE_OT_ExportSynthBBoxes,
    SCENE_OT_CreateShelfCameraSweep,
    SCENE_OT_RenderAnimationSequence,
    CollectionListItem,
    ExportProperties,
    SCENE_UL_CollectionList,
    SCENE_OT_CollectionListAdd,
    SCENE_OT_CollectionListRemove,
    VIEW3D_PT_gen_panel,
)

def register():
    for cls in classes: bpy.utils.register_class(cls)
    bpy.types.Scene.export_props = bpy.props.PointerProperty(type=ExportProperties)

def unregister():
    for cls in reversed(classes): bpy.utils.unregister_class(cls)
    del bpy.types.Scene.export_props


if __name__ == "__main__":
    register()