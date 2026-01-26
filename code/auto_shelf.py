bl_info = {
    "name": "货架大师 (Shelf Master v3.7)", 
    "author": "Gemini Assistant",
    "version": (3, 7),
    "blender": (3, 0, 0),
    "location": "View3D > N-Panel > 货架大师",
    "description": "v3.7: 添加可折叠UI面板，边界控制和边缘优先摆放",
    "category": "Object",
}

import bpy
import random
import time
import math
from mathutils import Vector
from bpy.props import FloatProperty, IntProperty, BoolProperty, PointerProperty, EnumProperty
from bpy.types import Operator, Panel, PropertyGroup

COLLECTION_NAME = "SM_Stocked_Products_Final"

class SM_Properties_v34(PropertyGroup):
    shape_mode: EnumProperty(
        name="物体形状",
        items=[
            ('CYLINDER', "圆柱体 (瓶/罐)", "紧密排列", 'MESH_CYLINDER', 0),
            ('BOX', "立方体 (盒/书)", "防撞角穿模", 'MESH_CUBE', 1),
        ],
        default='CYLINDER'
    )
    gap: FloatProperty(name="基础间距", default=0.015, min=0.001, max=1.0)
    vacancy_rate: FloatProperty(name="销售程度", default=0.1, min=0.0, max=1.0, subtype='FACTOR')
    invert_front: BoolProperty(name="翻转前后方向", default=False)
    jitter_rot: FloatProperty(name="旋转抖动", default=0.1, min=0.0, max=3.14)
    jitter_pos: FloatProperty(name="位置抖动", default=0.01, min=0.0, max=0.2)
    edge_margin_x: FloatProperty(name="左右边界留空", default=0.02, min=0.0, max=0.5, description="沿货架长度方向的边界留空")
    edge_margin_y: FloatProperty(name="前后边界留空", default=0.02, min=0.0, max=0.5, description="沿货架深度方向的边界留空")
    prefer_edges: BoolProperty(name="优先边缘摆放", default=False, description="商品尽量远离货架中心")
    safety_margin: FloatProperty(name="防穿模系数", default=1.05, min=1.0, max=2.0)
    # UI折叠控制
    ui_fold_stacking: BoolProperty(name="", default=True)
    ui_fold_grouping: BoolProperty(name="", default=True)
    ui_fold_spacing: BoolProperty(name="", default=True)
    ui_fold_edges: BoolProperty(name="", default=True)
    use_grouping: BoolProperty(name="启用分组", default=True)
    facing_min: IntProperty(name="排面Min", default=2, min=1)
    facing_max: IntProperty(name="排面Max", default=5, min=1)
    use_stacking: BoolProperty(name="允许垂直堆叠", default=False)
    max_stack: IntProperty(name="最大堆叠数", default=3, min=1, max=10) 
    check_height: BoolProperty(name="启用高度检测", default=True)
    top_gap_ratio: FloatProperty(name="顶部留空", default=0.05, min=0.0, max=0.5)
    check_isolated: BoolProperty(name="过滤孤立排", default=True, description="跳过货架顶部或过窄的单行位置")
    seed: IntProperty(name="随机种子", default=42, min=-1)

class OT_ClearShelfItems_v34(Operator):
    bl_idname = "object.sm_clear_items_v34"
    bl_label = "清空货架"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        targets = [COLLECTION_NAME, "SM_Stocked_Products_Final"]
        for t_name in targets:
            if t_name in bpy.data.collections:
                col = bpy.data.collections[t_name]
                for obj in col.objects:
                    try: bpy.data.objects.remove(obj, do_unlink=True)
                    except: pass
                bpy.data.collections.remove(col)
        self.report({'INFO'}, "已清空。")
        return {'FINISHED'}

class OT_GenerateShelfItems_v34(Operator):
    bl_idname = "object.sm_generate_items_v34"
    bl_label = "生成布局"
    bl_options = {'REGISTER', 'UNDO'}

    def get_obj_radius(self, obj, mode):
        width, depth = obj.dimensions.x, obj.dimensions.y
        return max(width, depth) / 2.0 if mode == 'CYLINDER' else math.sqrt(width**2 + depth**2) / 2.0

    def get_obj_dims(self, obj):
        return obj.dimensions

    def get_z_offset(self, obj):
        bbox = [Vector(v) for v in obj.bound_box]
        min_z_local = min([v.z for v in bbox])
        return abs(min_z_local) * obj.scale.z

    def place_item_simple(self, stock_col, target, base_loc, item_height, item_z_offset, target_radius, placed_items, props, rng, max_attempts=6):
        # 简化注释，保留必要说明
        # 放置商品，避免穿模
        effective_radius = target_radius
        EPSILON = 0.001
        for _ in range(max_attempts):
            jx = rng.uniform(-props.jitter_pos, props.jitter_pos)
            jy = rng.uniform(-props.jitter_pos, props.jitter_pos)
            bottom_pos = Vector((base_loc.x + jx, base_loc.y + jy, base_loc.z))
            has_collision = False
            # 预计算新物体的上下边界
            z_bottom_new = bottom_pos.z + EPSILON
            z_top_new = bottom_pos.z + item_height - EPSILON
            for pp, pr, ph in placed_items:
                # 现有物体的上下边界
                z_bottom_old = pp.z
                z_top_old = pp.z + ph
                # Z轴碰撞检测（AABB重叠）
                is_z_overlap = (z_bottom_new < z_top_old) and (z_top_new > z_bottom_old)
                if is_z_overlap:
                    dist_2d = (Vector((pp.x, pp.y)) - Vector((bottom_pos.x, bottom_pos.y))).length
                    # 距离 >= (新半径 + 旧半径) * 安全系数
                    min_safe_dist = (effective_radius + pr) * props.safety_margin
                    if dist_2d < min_safe_dist:
                        has_collision = True
                        break
            if has_collision:
                continue
            new_o = target.copy()
            new_o.data = target.data
            stock_col.objects.link(new_o)
            new_o.location = Vector((bottom_pos.x, bottom_pos.y, bottom_pos.z + item_z_offset))
            new_o.rotation_euler.z = target.rotation_euler.z + rng.uniform(-props.jitter_rot, props.jitter_rot)
            placed_items.append((bottom_pos, effective_radius, item_height))
            return True
        return False

    def generate_global_plan(self, products, plan_start, plan_end, grid_step, props, rng):
        if not products: return [], []
        product_queue = products[:]
        rng.shuffle(product_queue)
        shelf_plan = []
        segment_stock_levels = []
        current_p = plan_start
        while current_p < plan_end:
            if not product_queue: 
                product_queue = products[:]
                rng.shuffle(product_queue)
            prod = product_queue.pop(0)
            facings = rng.randint(props.facing_min, props.facing_max)
            segment_len = facings * grid_step
            end_p = current_p + segment_len
            shelf_plan.append({'end': end_p, 'obj': prod})
            base_stock = 1.0 - props.vacancy_rate
            if props.vacancy_rate <= 0.001: stock_level = 1.0
            elif props.vacancy_rate >= 0.999: stock_level = 0.0
            else:
                variation = rng.uniform(-0.1, 0.1)
                stock_level = max(0.0, min(1.0, base_stock + variation))
            segment_stock_levels.append(stock_level)
            current_p = end_p
        return shelf_plan, segment_stock_levels

    def check_headroom_robust(self, scene, depsgraph, center_pos, radius, shelf_obj):
        # 检查从center_pos向上到上层货架板的实际空间
        # 向上射线检测，找到最近的遮挡物（包括货架的上层板）
        offsets = [Vector((0,0,0)), Vector((radius*0.5,0,0)), Vector((-radius*0.5,0,0)), Vector((0,radius*0.5,0)), Vector((0,-radius*0.5,0))]
        min_headroom = 999.0
        for off in offsets:
            start_p = center_pos + off + Vector((0, 0, 0.02))
            res, loc, _, _, obj, _ = scene.ray_cast(depsgraph, start_p, Vector((0, 0, 1)))
            if res:
                dist = loc.z - center_pos.z
                if dist > 0.03 and dist < min_headroom:
                    min_headroom = dist
        if min_headroom > 900:
            shelf_top_z = max([(shelf_obj.matrix_world @ Vector(c)).z for c in shelf_obj.bound_box])
            min_headroom = shelf_top_z - center_pos.z
        return min_headroom

    def check_floor_robust(self, scene, depsgraph, center_pos, radius, shelf_obj):
        # 检查商品底部是否有有效的货架表面支撑
        check_radius = min(radius * 0.5, 0.02)
        offsets = [
            Vector((0, 0, 0)),
            Vector((check_radius, 0, 0)), Vector((-check_radius, 0, 0)),
            Vector((0, check_radius, 0)), Vector((0, -check_radius, 0))
        ]
        ray_start_lift = 0.05
        z_start = center_pos.z + ray_start_lift
        MAX_FLOAT_TOLERANCE = 0.01
        ray_length = ray_start_lift + MAX_FLOAT_TOLERANCE + 0.005
        success_count = 0
        target_original = shelf_obj.original if hasattr(shelf_obj, "original") else shelf_obj
        for off in offsets:
            start_p = Vector((center_pos.x, center_pos.y, z_start)) + off
            res, loc, norm, _, obj, _ = scene.ray_cast(
                depsgraph, 
                start_p, 
                Vector((0, 0, -1)), 
                distance=ray_length
            )
            if res:
                hit_obj = obj.original if hasattr(obj, "original") else obj
                if hit_obj != target_original:
                    continue
                if abs(norm.z) < 0.8: 
                    continue
                if not self.is_top_surface_at(scene, depsgraph, loc, shelf_obj):
                    continue
                diff = center_pos.z - loc.z 
                MAX_EMBED_TOLERANCE = 0.03
                if -MAX_EMBED_TOLERANCE <= diff <= MAX_FLOAT_TOLERANCE:
                    success_count += 1
        return success_count >= 3

    def is_top_surface_at(self, scene, depsgraph, hit_loc, shelf_obj, max_thickness=0.02):
        # 判定当前命中的货架面是否为层板上表面
        shelf_original = shelf_obj.original if hasattr(shelf_obj, "original") else shelf_obj
        start_p = hit_loc + Vector((0, 0, 0.002))
        res, loc, _, _, obj, _ = scene.ray_cast(
            depsgraph,
            start_p,
            Vector((0, 0, 1)),
            distance=max_thickness
        )
        if not res:
            return True
        hit_obj = obj.original if hasattr(obj, "original") else obj
        return hit_obj != shelf_original

    def is_valid_shelf_at(self, scene, depsgraph, pos, shelf_obj, target_z):
        # 检查指定位置是否是有效的货架表面
        ray_orig = Vector((pos.x, pos.y, target_z + 0.1))
        success, loc, norm, _, obj, _ = scene.ray_cast(depsgraph, ray_orig, Vector((0, 0, -1)))
        if not success:
            return False
        hit_obj = obj.original if hasattr(obj, "original") else obj
        shelf_original = shelf_obj.original if hasattr(shelf_obj, "original") else shelf_obj
        return (
            hit_obj == shelf_original
            and abs(loc.z - target_z) < 0.02
            and abs(norm.z) > 0.8
            and self.is_top_surface_at(scene, depsgraph, loc, shelf_obj)
        )

    def get_depth_direction(self, shelf):
        # 获取货架的深度方向向量（垂直于长边的方向）
        # 使用局部尺寸判断长短轴，而非世界坐标包围盒
        local_dims = shelf.dimensions
        if local_dims.x >= local_dims.y:
            depth_vec_local = Vector((0, 1, 0))
        else:
            depth_vec_local = Vector((1, 0, 0))
        depth_vec = shelf.matrix_world.to_3x3() @ depth_vec_local
        depth_vec.z = 0
        return depth_vec.normalized()

    def execute(self, context):
        try: return self.safe_execute(context)
        except Exception as e:
            self.report({'ERROR'}, f"错误: {str(e)}")
            return {'CANCELLED'}

    def safe_execute(self, context):
        # 主执行逻辑
        scene = context.scene
        props = scene.sm_props_v34
        selection = context.selected_objects
        active_obj = context.active_object
        if not selection or not active_obj or len(selection) < 2:
            self.report({'ERROR'}, "需选择商品和货架")
            return {'CANCELLED'}
        shelf = active_obj
        all_products = [o for o in selection if o != shelf]
        actual_seed = props.seed if props.seed != -1 else int(time.time() * 1000) % 1000000
        product_heights = {p.name: p.dimensions.z for p in all_products}
        product_z_offsets = {p.name: self.get_z_offset(p) for p in all_products}
        max_safe_radius = max(self.get_obj_radius(p, props.shape_mode) for p in all_products)
        grid_step = (max_safe_radius * 2) + props.gap
        if grid_step <= 0.0001: return {'CANCELLED'}
        corners = [shelf.matrix_world @ Vector(c) for c in shelf.bound_box]
        min_x, max_x = min([v.x for v in corners]), max([v.x for v in corners])
        min_y, max_y = min([v.y for v in corners]), max([v.y for v in corners])
        max_z, min_z_bound = max([v.z for v in corners]), min([v.z for v in corners])
        len_x, len_y = max_x - min_x, max_y - min_y
        local_dims = shelf.dimensions
        is_local_x_long = (local_dims.x >= local_dims.y)
        shelf_center = shelf.matrix_world.to_translation()
        shelf_matrix_3x3 = shelf.matrix_world.to_3x3()
        local_x_world = shelf_matrix_3x3 @ Vector((1, 0, 0))
        local_y_world = shelf_matrix_3x3 @ Vector((0, 1, 0))
        local_x_world.z = 0
        local_y_world.z = 0
        local_x_world = local_x_world.normalized()
        local_y_world = local_y_world.normalized()
        if is_local_x_long:
            length_axis = local_x_world
            depth_axis = local_y_world
            max_length = local_dims.x
            max_depth = local_dims.y
        else:
            length_axis = local_y_world
            depth_axis = local_x_world
            max_length = local_dims.y
            max_depth = local_dims.x
        edge_x = max(0.0, props.edge_margin_x)
        edge_y = max(0.0, props.edge_margin_y)
        min_x += edge_x
        max_x -= edge_x
        min_y += edge_y
        max_y -= edge_y
        len_x = max(0.0, max_x - min_x)
        len_y = max(0.0, max_y - min_y)
        cols = int(len_x / grid_step)
        rows = int(len_y / grid_step)
        depth_direction = depth_axis if props.check_isolated else None
        if COLLECTION_NAME not in bpy.data.collections:
            stock_col = bpy.data.collections.new(COLLECTION_NAME)
            context.scene.collection.children.link(stock_col)
        else:
            stock_col = bpy.data.collections[COLLECTION_NAME]
        depsgraph = context.evaluated_depsgraph_get()
        length_margin = props.edge_margin_x if is_local_x_long else props.edge_margin_y
        usable_length = max(0.0, max_length - (length_margin * 2.0))
        plan_start, plan_end = -usable_length / 2.0, usable_length / 2.0
        ray_start_z = max_z + 1.0
        total_count, placed_items, level_plans = 0, [], {}
        
        # 生成所有格子坐标
        grid_cells = []
        for r_idx in range(rows):
            for c_idx in range(cols):
                base_x = min_x + (c_idx * grid_step) + (grid_step / 2)
                base_y = min_y + (r_idx * grid_step) + (grid_step / 2)
                grid_cells.append((r_idx, c_idx, base_x, base_y))
        
        # 如果启用边缘优先，按距离中心由远到近排序
        if props.prefer_edges:
            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            grid_cells.sort(key=lambda cell: -((cell[2] - center_x)**2 + (cell[3] - center_y)**2)**0.5)
        
        # 遍历格子进行摆放
        for r_idx, c_idx, base_x, base_y in grid_cells:
                world_pos = Vector((base_x, base_y, 0))
                rel_vec = world_pos - Vector((shelf_center.x, shelf_center.y, 0))
                check_coord = rel_vec.dot(length_axis)
                depth_local = rel_vec.dot(depth_axis)
                depth_ratio = (depth_local + max_depth / 2.0) / max_depth if max_depth > 0.001 else 0.5
                depth_ratio = max(0.0, min(1.0, depth_ratio))
                if props.invert_front: 
                    depth_ratio = 1.0 - depth_ratio
                cur_ray_z = ray_start_z
                loop_safety = 50 
                while loop_safety > 0:
                    loop_safety -= 1
                    if cur_ray_z < min_z_bound: break
                    ray_orig = Vector((base_x, base_y, cur_ray_z))
                    success, loc, norm, idx, obj, mtx = scene.ray_cast(depsgraph, ray_orig, Vector((0,0,-1)))
                    if not success: break
                    next_ray_z = loc.z - 0.01 
                    if obj.name == shelf.name and abs(norm.z) > 0.8 and self.is_top_surface_at(scene, depsgraph, loc, shelf):
                        if props.check_isolated and depth_direction:
                            neighbor_f = Vector((base_x, base_y, loc.z)) + depth_direction * grid_step
                            neighbor_b = Vector((base_x, base_y, loc.z)) - depth_direction * grid_step
                            has_neighbor = (self.is_valid_shelf_at(scene, depsgraph, neighbor_f, shelf, loc.z) or
                                            self.is_valid_shelf_at(scene, depsgraph, neighbor_b, shelf, loc.z))
                            if not has_neighbor:
                                cur_ray_z = next_ray_z
                                continue
                        level_key = round(loc.z / 0.05) * 0.05
                        if level_key not in level_plans:
                            lvl_rng = random.Random(actual_seed + int(level_key * 100))
                            level_plans[level_key] = self.generate_global_plan(all_products, plan_start, plan_end, grid_step, props, lvl_rng)
                        current_plan, current_stock = level_plans[level_key]
                        target, allow_placement, seg_idx = None, True, -1
                        if props.use_grouping and current_plan:
                            for i, seg in enumerate(current_plan):
                                if check_coord <= seg['end'] + 0.001:
                                    target, seg_idx = seg['obj'], i
                                    break
                            if target is None:
                                target, seg_idx = current_plan[-1]['obj'], len(current_plan)-1
                            stock_val = current_stock[seg_idx]
                            allow_placement = stock_val > 0.0 and depth_ratio <= stock_val + 0.001
                        else:
                            col_rng = random.Random(actual_seed + int(level_key * 1000) + int(check_coord * 100))
                            target = col_rng.choice(all_products)
                            stock_val = 1.0 - props.vacancy_rate
                            allow_placement = stock_val > 0.0 and depth_ratio <= stock_val + 0.001
                        if not target or not allow_placement:
                            cur_ray_z = next_ray_z
                            continue
                        target_radius_init = self.get_obj_radius(target, props.shape_mode)
                        headroom = self.check_headroom_robust(scene, depsgraph, loc, target_radius_init, shelf)
                        safe_limit = headroom * (1.0 - props.top_gap_ratio)
                        item_height = product_heights[target.name]
                        item_z_offset = product_z_offsets[target.name]
                        required_height = item_height + item_z_offset
                        if required_height > safe_limit:
                            candidates = [p for p in all_products if (product_heights[p.name] + product_z_offsets[p.name]) <= safe_limit]
                            if not candidates:
                                cur_ray_z = next_ray_z
                                continue
                            sub_rng = random.Random(actual_seed + int(level_key * 500) + seg_idx)
                            target = sub_rng.choice(candidates)
                            item_height = product_heights[target.name]
                            item_z_offset = product_z_offsets[target.name]
                            required_height = item_height + item_z_offset
                        target_radius = self.get_obj_radius(target, props.shape_mode)
                        floor_ok = self.check_floor_robust(scene, depsgraph, loc, target_radius, shelf)
                        if not floor_ok:
                            cur_ray_z = next_ray_z
                            continue
                        stack_count = 1
                        if props.use_stacking:
                            usable_height = max(0.0, safe_limit - item_z_offset)
                            stack_count = min(max(1, int(usable_height / item_height)), props.max_stack)
                        local_rng = random.Random(actual_seed + r_idx * 100 + c_idx + int(level_key * 50))
                        for k in range(stack_count):
                            base_loc = Vector((loc.x, loc.y, loc.z + (k * item_height)))
                            placed = self.place_item_simple(
                                stock_col,
                                target,
                                base_loc,
                                item_height,
                                item_z_offset,
                                target_radius,
                                placed_items,
                                props,
                                local_rng,
                            )
                            if placed:
                                total_count += 1
                            else:
                                break
                    cur_ray_z = next_ray_z
        self.report({'INFO'}, f"生成完毕: {total_count} 个")
        return {'FINISHED'}

class PT_ShelfMasterPanel_v34(Panel):
    # UI 面板
    bl_label = "货架大师 v3.7"
    bl_idname = "VIEW3D_PT_shelf_master_v34"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "货架大师"

    @classmethod
    def poll(cls, context): return context.mode == 'OBJECT'

    def draw(self, context):
        layout = self.layout
        props = context.scene.sm_props_v34
        box = layout.box()
        box.label(text="1. 先选商品  2. Shift+加选货架", icon='INFO')
        layout.prop(props, "shape_mode", expand=True)
        
        # 堆叠与高度 (可折叠)
        row = layout.row()
        row.prop(props, "ui_fold_stacking", icon="TRIA_DOWN" if props.ui_fold_stacking else "TRIA_RIGHT", emboss=False)
        row.label(text="堆叠与高度:", icon='ALIGN_TOP')
        if props.ui_fold_stacking:
            box = layout.box()
            box.prop(props, "use_stacking")
            if props.use_stacking: box.prop(props, "max_stack")
            box.prop(props, "top_gap_ratio", slider=True, text="顶部留空")
            box.prop(props, "check_isolated")
        
        # 排面与销售 (可折叠)
        row = layout.row()
        row.prop(props, "ui_fold_grouping", icon="TRIA_DOWN" if props.ui_fold_grouping else "TRIA_RIGHT", emboss=False)
        row.label(text="排面与销售:", icon='GRID')
        if props.ui_fold_grouping:
            box = layout.box()
            box.prop(props, "use_grouping")
            if props.use_grouping:
                row = box.row(align=True)
                row.prop(props, "facing_min")
                row.prop(props, "facing_max")
            box.prop(props, "vacancy_rate", slider=True, text="销售程度 (0全满, 1全空)")
            box.prop(props, "invert_front")
        
        # 间距与随机 (可折叠)
        row = layout.row()
        row.prop(props, "ui_fold_spacing", icon="TRIA_DOWN" if props.ui_fold_spacing else "TRIA_RIGHT", emboss=False)
        row.label(text="间距与随机:", icon='PREFERENCES')
        if props.ui_fold_spacing:
            col = layout.column(align=True)
            col.prop(props, "gap"); col.prop(props, "safety_margin", slider=True)
            col.prop(props, "jitter_rot"); col.prop(props, "jitter_pos")
            col.prop(props, "seed")
        
        # 边界控制 (可折叠)
        row = layout.row()
        row.prop(props, "ui_fold_edges", icon="TRIA_DOWN" if props.ui_fold_edges else "TRIA_RIGHT", emboss=False)
        row.label(text="边界控制:", icon='DRIVER_DISTANCE')
        if props.ui_fold_edges:
            box = layout.box()
            box.prop(props, "edge_margin_x")
            box.prop(props, "edge_margin_y")
            box.prop(props, "prefer_edges")

        col = layout.column(align=True)
        col.operator("object.sm_generate_items_v34", icon='PLAY', text="生成布局")
        col.operator("object.sm_clear_items_v34", icon='TRASH', text="一键清空")

classes = (SM_Properties_v34, OT_GenerateShelfItems_v34, OT_ClearShelfItems_v34, PT_ShelfMasterPanel_v34)

def register():
    for cls in classes: bpy.utils.register_class(cls)
    bpy.types.Scene.sm_props_v34 = PointerProperty(type=SM_Properties_v34)

def unregister():
    for cls in reversed(classes): bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, "sm_props_v34"): del bpy.types.Scene.sm_props_v34

if __name__ == "__main__":
    register()
