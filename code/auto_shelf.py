bl_info = {
    "name": "货架大师 (Shelf Master CN)", 
    "author": "Gemini Assistant",
    "version": (2, 3),
    "blender": (3, 0, 0),
    "location": "3D视图 > N面板 > 货架大师",
    "description": "自动理货插件：真实库存逻辑 (不重复、自动留白)",
    "category": "Object",
}

import bpy
import random
import time
import math
from mathutils import Vector
from bpy.props import FloatProperty, IntProperty, BoolProperty, PointerProperty, EnumProperty
from bpy.types import Operator, Panel, PropertyGroup

COLLECTION_NAME = "SM_Stocked_Products" 

# --- 1. 属性定义 ---
class SM_Properties(PropertyGroup):
    shape_mode: EnumProperty(
        name="物体形状",
        items=[
            ('CYLINDER', "圆柱体 (瓶/罐)", "紧密排列", 'MESH_CYLINDER', 0),
            ('BOX', "立方体 (盒/书)", "防撞角穿模", 'MESH_CUBE', 1),
        ],
        default='CYLINDER'
    )
    
    gap: FloatProperty(name="基础间距", default=0.005, min=0.001, max=1.0, subtype='DISTANCE')
    
    # 销售模拟
    vacancy_rate: FloatProperty(
        name="销售程度 (缺货率)", 
        description="数值越高，外侧空的越多",
        default=0.2, min=0.0, max=1.0, subtype='FACTOR'
    )
    invert_front: BoolProperty(name="翻转前后方向", default=False)
    
    jitter_rot: FloatProperty(name="随机旋转幅度", default=0.1, min=0.0, max=3.14)
    jitter_pos: FloatProperty(name="随机位置抖动", default=0.01, min=0.0, max=0.2, subtype='DISTANCE')
    safety_margin: FloatProperty(name="防穿模系数", default=1.05, min=1.0, max=2.0)
    
    use_grouping: BoolProperty(name="启用自动分组", default=True)
    facing_min: IntProperty(name="排面宽度 (列数Min)", default=2, min=1)
    facing_max: IntProperty(name="排面宽度 (列数Max)", default=5, min=1)

    seed: IntProperty(name="随机种子 (-1=随机)", default=42, min=-1)
    check_height: BoolProperty(name="检测层高", description="防止穿顶", default=True)

# --- 2. 操作符 (逻辑核心) ---
class OT_ClearShelfItems(Operator):
    bl_idname = "object.sm_clear_items"
    bl_label = "清空所有生成的物品"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if COLLECTION_NAME in bpy.data.collections:
            col = bpy.data.collections[COLLECTION_NAME]
            objs = [o for o in col.objects]
            for obj in objs:
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(col)
            self.report({'INFO'}, "已清空货架。")
        else:
            self.report({'WARNING'}, "无需清空。")
        return {'FINISHED'}

class OT_GenerateShelfItems(Operator):
    bl_idname = "object.sm_generate_items"
    bl_label = "开始生成布局"
    bl_options = {'REGISTER', 'UNDO'}

    def get_obj_safe_radius(self, obj, mode):
        width = obj.dimensions.x
        depth = obj.dimensions.y
        if mode == 'CYLINDER':
            return max(width, depth) / 2.0
        else:
            return math.sqrt(width**2 + depth**2) / 2.0
            
    def get_obj_height(self, obj):
        return obj.dimensions.z

    def get_z_offset(self, obj):
        bbox = [Vector(v) for v in obj.bound_box]
        min_z_local = min([v.z for v in bbox])
        return abs(min_z_local) * obj.scale.z

    def execute(self, context):
        scene = context.scene
        props = scene.sm_props
        
        selection = context.selected_objects
        active_obj = context.active_object
        
        if not selection or not active_obj or len(selection) < 2:
            self.report({'ERROR'}, "请先框选商品，最后Shift加选货架。")
            return {'CANCELLED'}
        
        shelf = active_obj
        products = [o for o in selection if o != shelf]
        
        if not products:
            self.report({'ERROR'}, "未找到商品。")
            return {'CANCELLED'}

        actual_seed = props.seed
        if actual_seed == -1:
            actual_seed = int(time.time() * 1000) % 1000000
        random.seed(actual_seed)

        max_safe_radius = 0
        for p in products:
            r = self.get_obj_safe_radius(p, props.shape_mode)
            if r > max_safe_radius: max_safe_radius = r
        
        grid_step = (max_safe_radius * 2) + props.gap
        collision_threshold_dist = (max_safe_radius * 2) * props.safety_margin

        if grid_step <= 0.001: return {'CANCELLED'}
        
        corners = [shelf.matrix_world @ Vector(c) for c in shelf.bound_box]
        min_x = min([v.x for v in corners])
        max_x = max([v.x for v in corners])
        min_y = min([v.y for v in corners])
        max_y = max([v.y for v in corners])
        max_z = max([v.z for v in corners])
        min_z_bound = min([v.z for v in corners])
        
        # 智能长边判定
        len_x = max_x - min_x
        len_y = max_y - min_y
        is_long_axis_x = True
        if len_y > len_x: is_long_axis_x = False
            
        if COLLECTION_NAME not in bpy.data.collections:
            stock_col = bpy.data.collections.new(COLLECTION_NAME)
            context.scene.collection.children.link(stock_col)
        else:
            stock_col = bpy.data.collections[COLLECTION_NAME]
        
        depsgraph = context.evaluated_depsgraph_get()
        
        # --- 规划排面 (v2.3 不重复逻辑) ---
        shelf_plan = []
        plan_start = min_x if is_long_axis_x else min_y
        plan_end = max_x if is_long_axis_x else max_y
        current_p = plan_start
        segment_stock_levels = []
        
        # 1. 创建待摆放队列 (复制一份并打乱)
        product_queue = products[:]
        random.shuffle(product_queue) # 随机排序
        
        if props.use_grouping:
            # 2. 遍历队列，摆完为止
            for prod in product_queue:
                # 如果货架满了，停止摆放
                if current_p >= plan_end:
                    break
                
                facings = random.randint(props.facing_min, props.facing_max)
                segment_len = facings * grid_step
                
                end_p = current_p + segment_len
                
                # 可选：如果这一组超出了货架边缘，是否截断？
                # 目前逻辑是允许最后这组超出一点点（扫描时会自动裁剪），或者只要起点在范围内就摆
                
                shelf_plan.append({'end': end_p, 'obj': prod})
                
                # 随机库存
                random_loss = random.uniform(0.0, props.vacancy_rate * 2.0)
                stock_percent = max(0.0, 1.0 - random_loss)
                segment_stock_levels.append(stock_percent)
                
                current_p = end_p
                
            # 循环结束后，如果 current_p 还不到 plan_end，剩下的部分自然就是空的
        
        # 网格扫描
        x_range = len_x
        y_range = len_y
        cols = int(x_range / grid_step)
        rows = int(y_range / grid_step)
        max_depth_steps = rows if is_long_axis_x else cols
        
        ray_start_z_limit = max_z + 1.0
        total_count = 0
        placed_items_list = [] 

        for r in range(rows):
            for c in range(cols):
                
                base_x = min_x + (c * grid_step) + (grid_step / 2)
                base_y = min_y + (r * grid_step) + (grid_step / 2)
                check_coord = base_x if is_long_axis_x else base_y
                
                # 深度计算
                if is_long_axis_x:
                    current_depth_idx = r
                else:
                    current_depth_idx = c
                depth_ratio = current_depth_idx / max(1, max_depth_steps)
                if props.invert_front: depth_ratio = 1.0 - depth_ratio

                target_product = None
                allow_placement = True
                
                if props.use_grouping:
                    seg_idx = -1
                    # 查表
                    for i, segment in enumerate(shelf_plan):
                        if check_coord <= segment['end']:
                            target_product = segment['obj']
                            seg_idx = i
                            break
                    
                    # 重点修改：如果超出了所有plan的范围，就不摆了 (None)
                    # 之前的版本是 target_product = shelf_plan[-1]，现在删掉这行
                    
                    if target_product is None:
                        continue # 这里空着
                    
                    if seg_idx >= 0:
                        stock_limit = segment_stock_levels[seg_idx]
                        if depth_ratio > stock_limit:
                            allow_placement = False
                else:
                    # 不分组模式下，依然保持随机选择（不分组就意味着杂乱，重复无所谓）
                    # 如果你也希望不分组模式下也不重复，逻辑会变得很奇怪（变成完全随机的马赛克）
                    # 所以不分组模式维持原样，只在分组模式应用“不重复”
                    target_product = random.choice(products)
                    indiv_stock = 1.0 - (random.uniform(0.0, props.vacancy_rate * 2.0))
                    if depth_ratio > indiv_stock:
                        allow_placement = False
                
                if target_product is None or not allow_placement: 
                    continue

                # --- 垂直扫描 ---
                current_ray_z = ray_start_z_limit
                layer_safety_limit = 20 
                
                while layer_safety_limit > 0:
                    layer_safety_limit -= 1
                    if current_ray_z < min_z_bound: break

                    ray_orig = Vector((base_x, base_y, current_ray_z))
                    success, loc, norm, idx, obj, mtx = scene.ray_cast(depsgraph, ray_orig, Vector((0,0,-1)))
                    
                    if not success: break
                    next_ray_z = loc.z - 0.01 

                    if obj.name == shelf.name and norm.z > 0.5:
                        has_headroom = True
                        if props.check_height:
                            item_height = self.get_obj_height(target_product)
                            up_ray_orig = Vector((loc.x, loc.y, loc.z + 0.01))
                            up_success, up_loc, _, _, _, _ = scene.ray_cast(depsgraph, up_ray_orig, Vector((0,0,1)))
                            if up_success:
                                headroom = up_loc.z - loc.z
                                if headroom < item_height * 1.01:
                                    has_headroom = False
                        
                        if has_headroom:
                            best_pos = None
                            for _ in range(10):
                                jx = random.uniform(-props.jitter_pos, props.jitter_pos)
                                jy = random.uniform(-props.jitter_pos, props.jitter_pos)
                                tentative = Vector((loc.x + jx, loc.y + jy, loc.z))
                                collision = False
                                if props.jitter_pos > 0.0001 or props.safety_margin > 1.0:
                                    for p_pos, p_rad in placed_items_list:
                                        if abs(p_pos.z - tentative.z) < 0.1: 
                                            dist_2d = (Vector((p_pos.x, p_pos.y)) - Vector((tentative.x, tentative.y))).length
                                            if dist_2d < collision_threshold_dist:
                                                collision = True
                                                break
                                if not collision:
                                    best_pos = tentative
                                    break
                            
                            if best_pos:
                                new_obj = target_product.copy()
                                new_obj.data = target_product.data
                                stock_col.objects.link(new_obj)
                                new_obj.location = best_pos
                                new_obj.location.z += self.get_z_offset(target_product)
                                base_rot = target_product.rotation_euler.z
                                rand_rot = random.uniform(-props.jitter_rot, props.jitter_rot)
                                new_obj.rotation_euler.z = base_rot + rand_rot
                                r_safe = self.get_obj_safe_radius(target_product, props.shape_mode)
                                placed_items_list.append((new_obj.location, r_safe))
                                total_count += 1

                    current_ray_z = next_ray_z

        self.report({'INFO'}, f"完成: 生成 {total_count} 个物品")
        return {'FINISHED'}

# --- 3. UI 面板 ---
class PT_ShelfMasterPanel(Panel):
    bl_label = "货架大师 v2.3 (不重复版)"
    bl_idname = "VIEW3D_PT_shelf_master_cn"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "货架大师"

    def draw(self, context):
        layout = self.layout
        props = context.scene.sm_props

        box = layout.box()
        box.label(text="1. 选商品  2. Shift+选货架", icon='INFO')
        
        layout.separator()
        
        layout.label(text="核心模式:", icon='MODIFIER')
        layout.prop(props, "shape_mode", expand=True)
        layout.prop(props, "check_height")

        layout.separator()
        
        box = layout.box()
        box.label(text="智能分组 (Auto Grouping):", icon='GRID')
        box.prop(props, "use_grouping")
        
        if props.use_grouping:
            row = box.row(align=True)
            row.label(text="排面宽度 (列数):")
            row.prop(props, "facing_min", text="Min")
            row.prop(props, "facing_max", text="Max")
            box.label(text="* 商品不重复，摆完即止", icon='RESTRICT_SELECT_OFF')

        layout.separator()

        layout.label(text="拟真销售:", icon='GRAPH')
        col = layout.column()
        col.prop(props, "vacancy_rate", slider=True, text="销售程度")
        row = col.row()
        row.prop(props, "invert_front", text="翻转前后方向", icon='FILE_REFRESH')

        layout.separator()

        layout.label(text="参数微调:", icon='PREFERENCES')
        col = layout.column()
        col.prop(props, "gap")
        col.prop(props, "safety_margin", slider=True)
        col.prop(props, "jitter_rot")
        col.prop(props, "jitter_pos")
        col.prop(props, "seed")

        layout.separator()
        
        col = layout.column(align=True)
        col.scale_y = 1.4
        col.operator("object.sm_generate_items", icon='PLAY')
        col.operator("object.sm_clear_items", icon='TRASH')

# --- 4. 注册 ---
classes = (
    SM_Properties,
    OT_GenerateShelfItems,
    OT_ClearShelfItems,
    PT_ShelfMasterPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.sm_props = PointerProperty(type=SM_Properties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.sm_props

if __name__ == "__main__":
    register()