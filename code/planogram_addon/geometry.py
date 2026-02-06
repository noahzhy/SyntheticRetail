import bpy
import mathutils

def detect_shelf_levels(shelf_coll):
    # 假设每个shelf为一个mesh对象，按Z排序
    shelves = [obj for obj in shelf_coll.objects if obj.type == 'MESH']
    return sorted(shelves, key=lambda o: o.location.z, reverse=True)

def get_shelf_bounds(shelf_obj):
    # 返回shelf表面可用区域的xmin/xmax/y/z等
    bbox = [shelf_obj.matrix_world @ mathutils.Vector(corner) for corner in shelf_obj.bound_box]
    xs = [v.x for v in bbox]
    ys = [v.y for v in bbox]
    z = max([v.z for v in bbox])
    return {'xmin': min(xs), 'xmax': max(xs), 'ymin': min(ys), 'ymax': max(ys), 'z': z}

def get_obj_width(obj):
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    return max([v.x for v in bbox]) - min([v.x for v in bbox])

def compute_segment_width(sku, facing, spacing):
    return facing * get_obj_width(sku) + (facing - 1) * spacing

def compute_position(x_cursor, bounds, sku, i, d, props):
    # 计算每个SKU排面的位置
    width = get_obj_width(sku)
    x = x_cursor + i * (width + props.horizontal_spacing)
    y = (bounds['ymin'] + bounds['ymax']) / 2
    z = bounds['z']
    return (x, y, z)

def place_object(obj, pos):
    obj.location = pos


def register():
    pass


def unregister():
    pass
