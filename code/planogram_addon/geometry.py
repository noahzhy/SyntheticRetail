import bpy
from mathutils import Vector


def get_z_offset(obj):
    """
    计算物体底部相对于原点的Z轴偏移量
    返回值为正数，表示物体底部在原点下方多远
    """
    bbox = [Vector(v) for v in obj.bound_box]
    min_z_local = min([v.z for v in bbox])
    return abs(min_z_local) * obj.scale.z


def detect_shelf_levels(shelf_coll, context=None):
    """
    检测货架层板的可放置表面
    简化版：将每个mesh对象视为一个层板，使用其包围盒顶面
    """
    levels = []
    shelves = [obj for obj in shelf_coll.objects if obj.type == 'MESH']
    
    for obj in shelves:
        # 获取世界坐标包围盒
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        
        xs = [v.x for v in corners]
        ys = [v.y for v in corners]
        zs = [v.z for v in corners]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        top_z = max(zs)
        
        # 过滤太小的表面
        if (max_x - min_x) > 0.05 and (max_y - min_y) > 0.05:
            levels.append({
                'z': top_z,
                'xmin': min_x, 'xmax': max_x,
                'ymin': min_y, 'ymax': max_y,
                'obj': obj
            })
    
    # 按Z从高到低排序
    return sorted(levels, key=lambda l: l['z'], reverse=True)


def get_shelf_bounds(shelf_obj):
    """获取层板边界，支持dict或object"""
    if isinstance(shelf_obj, dict):
        return shelf_obj
    bbox = [shelf_obj.matrix_world @ Vector(corner) for corner in shelf_obj.bound_box]
    xs = [v.x for v in bbox]
    ys = [v.y for v in bbox]
    z = max([v.z for v in bbox])
    return {'xmin': min(xs), 'xmax': max(xs), 'ymin': min(ys), 'ymax': max(ys), 'z': z}


def get_obj_width(obj):
    """获取物体在X方向的宽度"""
    bbox = [Vector(v) for v in obj.bound_box]
    width = (max(v.x for v in bbox) - min(v.x for v in bbox)) * obj.scale.x
    return width


def get_obj_depth(obj):
    """获取物体在Y方向的深度"""
    bbox = [Vector(v) for v in obj.bound_box]
    depth = (max(v.y for v in bbox) - min(v.y for v in bbox)) * obj.scale.y
    return depth


def compute_segment_width(sku, facing, spacing):
    return facing * get_obj_width(sku) + (facing - 1) * spacing


def compute_position(x_cursor, bounds, sku, i, d, props):
    """计算SKU放置位置"""
    width = get_obj_width(sku)
    depth = get_obj_depth(sku)
    
    # X位置：光标 + 半宽 + 索引*(宽度+间距)
    x = x_cursor + width / 2.0 + i * (width + props.horizontal_spacing)
    
    # Y位置：从后往前排列，考虑edge_margin
    shelf_depth = bounds['ymax'] - bounds['ymin']
    # 从ymax（后方）减去edge_margin开始往前放
    y = bounds['ymax'] - props.edge_margin - depth / 2.0 - d * depth
    
    z = bounds['z']
    return (x, y, z)


def place_object(obj, pos):
    """
    放置物体，确保底部与表面对齐
    pos = (x, y, surface_z) 其中surface_z是层板上表面高度
    """
    z_offset = get_z_offset(obj)
    
    obj.location.x = pos[0]
    obj.location.y = pos[1]
    # 物体底部放在surface_z上，所以原点位置 = surface_z + z_offset
    obj.location.z = pos[2] + z_offset


def is_top_surface(scene, depsgraph, hit_loc, shelf_obj, max_thickness=0.02):
    """
    判断命中点是否为层板上表面（而非底面）
    
    通过从命中点向上发射短距离射线，检查是否有同一货架的遮挡：
    - 如果上方无遮挡，则为顶面
    - 如果上方有同一货架遮挡（在max_thickness范围内），则为底面
    
    Args:
        scene: Blender场景
        depsgraph: 评估后的依赖图
        hit_loc: 射线命中位置
        shelf_obj: 货架对象
        max_thickness: 层板最大厚度（米）
    
    Returns:
        bool: True表示是顶面，False表示是底面
    """
    shelf_original = shelf_obj.original if hasattr(shelf_obj, "original") else shelf_obj
    start_p = hit_loc + Vector((0, 0, 0.002))
    res, loc, _, _, obj, _ = scene.ray_cast(
        depsgraph, start_p, Vector((0, 0, 1)), distance=max_thickness
    )
    if not res:
        return True
    hit_obj = obj.original if hasattr(obj, "original") else obj
    return hit_obj != shelf_original


def detect_shelf_levels_by_ray(shelf_obj, context, sample_density=5):
    """
    通过射线检测货架的多个层板
    
    从货架顶部向下发射垂直射线，循环检测每一层层板表面。
    参考 auto_shelf.py 中的实现逻辑。
    
    Args:
        shelf_obj: 货架对象
        context: Blender context
        sample_density: 每个方向的采样点数量
    
    Returns:
        list[dict]: 每层的信息，按Z从高到低排序
            - 'z': 层板顶面Z坐标
            - 'xmin', 'xmax': X方向边界
            - 'ymin', 'ymax': Y方向边界
            - 'obj': 货架对象引用
    """
    scene = context.scene
    depsgraph = context.evaluated_depsgraph_get()
    
    # 获取货架边界
    corners = [shelf_obj.matrix_world @ Vector(c) for c in shelf_obj.bound_box]
    min_x = min(v.x for v in corners)
    max_x = max(v.x for v in corners)
    min_y = min(v.y for v in corners)
    max_y = max(v.y for v in corners)
    min_z = min(v.z for v in corners)
    max_z = max(v.z for v in corners)
    
    shelf_original = shelf_obj.original if hasattr(shelf_obj, "original") else shelf_obj
    
    # 在货架表面采样多个点，收集每层的命中信息
    levels = {}  # z_key -> {'z': z, 'x_hits': [], 'y_hits': []}
    
    step_x = (max_x - min_x) / sample_density
    step_y = (max_y - min_y) / sample_density
    
    for i in range(sample_density):
        for j in range(sample_density):
            base_x = min_x + step_x * (i + 0.5)
            base_y = min_y + step_y * (j + 0.5)
            
            # 从顶部向下扫描
            cur_z = max_z + 0.5
            safety = 20  # 防止无限循环
            
            while safety > 0 and cur_z > min_z:
                safety -= 1
                ray_orig = Vector((base_x, base_y, cur_z))
                success, loc, norm, _, obj, _ = scene.ray_cast(
                    depsgraph, ray_orig, Vector((0, 0, -1))
                )
                
                if not success:
                    break
                
                hit_obj = obj.original if hasattr(obj, "original") else obj
                
                # 检查是否命中货架且法线朝上（水平表面）
                if hit_obj == shelf_original and abs(norm.z) > 0.8:
                    # 检查是否为顶面（非底面）
                    if is_top_surface(scene, depsgraph, loc, shelf_obj):
                        # 使用Z坐标分组（0.03米精度）
                        z_key = round(loc.z / 0.03) * 0.03
                        if z_key not in levels:
                            levels[z_key] = {
                                'z': loc.z,
                                'x_hits': [],
                                'y_hits': []
                            }
                        levels[z_key]['x_hits'].append(base_x)
                        levels[z_key]['y_hits'].append(base_y)
                
                # 继续向下检测下一层
                cur_z = loc.z - 0.02
    
    # 转换为结果列表
    result = []
    for z_key, data in levels.items():
        if len(data['x_hits']) > 0:
            result.append({
                'z': data['z'],
                'xmin': min(data['x_hits']),
                'xmax': max(data['x_hits']),
                'ymin': min(data['y_hits']),
                'ymax': max(data['y_hits']),
                'obj': shelf_obj
            })
    
    # 按Z从高到低排序
    return sorted(result, key=lambda l: l['z'], reverse=True)


def register():
    pass


def unregister():
    pass
