import bpy
import random
from . import geometry, collision, utils

def generate_planogram(context, props):
    random_seed = props.random_seed
    if props.random_seed == -1:
        random_seed = random.randint(0, 1000)
    random.seed(random_seed)
    product_coll = bpy.data.collections.get(props.product_collection)
    shelf_coll = bpy.data.collections.get(props.shelf_collection)
    if not product_coll or not shelf_coll:
        utils.report_error("Invalid collection selection")
        return

    # Create new SM Collection
    target_coll_name = f"SM_Planogram"
    target_coll = bpy.data.collections.new(target_coll_name)
    context.scene.collection.children.link(target_coll)

    # 1. 解析货架层级（传入context以启用射线检测模式）
    shelf_levels = geometry.detect_shelf_levels(shelf_coll, context)
    # 2. 获取SKU列表
    skus = [obj for obj in product_coll.objects if obj.type == 'MESH']
    # 3. 按优先级/体积排序（可扩展）
    skus = sorted(skus, key=lambda o: o.name)
    # 4. 分配每层货架
    fill_order = shelf_levels if props.fill_order == 'TOP_DOWN' else list(reversed(shelf_levels))
    for level in fill_order:
        bounds = geometry.get_shelf_bounds(level)
        # 5. SKU分段布局
        segments = utils.segment_skus(skus, props)
        x_cursor = bounds['xmin'] + props.edge_margin  # 从左边缘加上margin开始
        
        for sku, seg_cfg in segments:
            sku_width = geometry.get_obj_width(sku)
            
            # 检查是否至少能放一个（考虑右边缘margin）
            if x_cursor + sku_width > bounds['xmax'] - props.edge_margin:
                break  # 这层已满，停止放置
            
            # 6. 计算排面数量和深度
            facing = random.randint(props.min_facing, props.max_facing)
            depth = random.randint(props.min_depth, props.max_depth)
            
            # 7. 计算可用空间并调整facing（减去两侧edge_margin）
            available_width = bounds['xmax'] - x_cursor - props.edge_margin
            max_possible_facing = int(available_width // (sku_width + props.horizontal_spacing))
            if max_possible_facing < 1:
                # 尝试不带间距放一个
                if available_width >= sku_width:
                    max_possible_facing = 1
                else:
                    break  # 放不下了
            
            # 限制facing不超过可放置数量
            if props.allow_partial:
                facing = min(facing, max_possible_facing)
            elif facing > max_possible_facing:
                break  # 不允许部分放置，跳过
            
            # 重新计算实际段宽度
            seg_width = geometry.compute_segment_width(sku, facing, props.horizontal_spacing)
            
            # 8. SKU排面布局
            for i in range(facing):
                for d in range(depth):
                    pos = geometry.compute_position(x_cursor, bounds, sku, i, d, props)
                    # 额外边界检查（考虑edge_margin）
                    if pos[0] - sku_width/2 < bounds['xmin'] + props.edge_margin or pos[0] + sku_width/2 > bounds['xmax'] - props.edge_margin:
                        continue  # 跳过超出边界的位置
                    if not collision.check_collision(sku, pos, context):
                        new_obj = utils.create_sku_instance(sku, target_coll)
                        geometry.place_object(new_obj, pos)
            
            x_cursor += seg_width + props.segment_spacing
    # 9. 可扩展：优先级、品牌分区等


def register():
    pass


def unregister():
    pass
