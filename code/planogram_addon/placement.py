import bpy
import random
from . import geometry, collision, utils

def generate_planogram(context, props):
    random.seed(props.random_seed)
    product_coll = bpy.data.collections.get(props.product_collection)
    shelf_coll = bpy.data.collections.get(props.shelf_collection)
    if not product_coll or not shelf_coll:
        utils.report_error("Invalid collection selection")
        return

    # 1. 解析货架层级
    shelf_levels = geometry.detect_shelf_levels(shelf_coll)
    # 2. 获取SKU列表
    skus = [obj for obj in product_coll.objects if obj.type == 'MESH']
    # 3. 按优先级/体积排序（可扩展）
    skus = sorted(skus, key=lambda o: o.name)
    # 4. 分配每层货架
    fill_order = shelf_levels if props.fill_order == 'TOP_DOWN' else reversed(shelf_levels)
    for level in fill_order:
        bounds = geometry.get_shelf_bounds(level)
        # 5. SKU分段布局
        segments = utils.segment_skus(skus, props)
        x_cursor = bounds['xmin']
        for sku, seg_cfg in segments:
            # 6. 计算排面数量和深度
            facing = random.randint(props.min_facing, props.max_facing)
            depth = random.randint(props.min_depth, props.max_depth)
            # 7. 计算SKU段宽度
            seg_width = geometry.compute_segment_width(sku, facing, props.horizontal_spacing)
            if x_cursor + seg_width > bounds['xmax']:
                if props.allow_partial:
                    facing = max(1, int((bounds['xmax'] - x_cursor) // (geometry.get_obj_width(sku) + props.horizontal_spacing)))
                    if facing < 1:
                        break
                else:
                    break
            # 8. SKU排面布局
            for i in range(facing):
                for d in range(depth):
                    pos = geometry.compute_position(x_cursor, bounds, sku, i, d, props)
                    if not collision.check_collision(sku, pos, context):
                        new_obj = utils.create_sku_instance(sku, product_coll)
                        geometry.place_object(new_obj, pos)
            x_cursor += seg_width + props.segment_spacing
    # 9. 可扩展：优先级、品牌分区等


def register():
    pass


def unregister():
    pass
