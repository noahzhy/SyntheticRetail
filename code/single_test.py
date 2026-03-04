"""
Planogram Addon - 单文件测试版
整合所有功能到一个文件，方便测试
"""

bl_info = {
    "name": "Planogram Test (Single File)",
    "author": "Gemini Assistant",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > N-Panel > Planogram",
    "description": "单文件测试版：包含几何计算、放置逻辑和UI",
    "category": "Object",
}

import bpy
import random
import math
from mathutils import Vector, Euler
from bpy.props import PointerProperty, IntProperty, FloatProperty, EnumProperty, BoolProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup

# ============================================================================
# 工具函数 (utils.py)
# ============================================================================

def report_error(msg):
    print(f"[ERROR] {msg}")


def segment_skus(skus, props, rng=None):
    """
    随机选择商品成组摆放
    
    支持两种模式：
    1. 重复抽取（替换采样）：商品用完后会重新补满，可无限抽取。
    2. 不重复抽取（不替换采样）：商品抽完后即停止。

    Args:
        skus: 商品列表
        props: 属性，包含 sample_with_replacement 选项
        rng: 随机数生成器（可选）
    
    Returns:
        list: [(sku, config), ...] 商品和配置的列表
    """
    if not skus:
        return []
    
    if rng is None:
        _seed = random.randint(0, 10000)
        rng = random.Random(_seed)
    
    # 创建洗牌袋
    shuffle_bag = skus[:]
    rng.shuffle(shuffle_bag)
    
    # 如果是不替换采样，直接返回打乱后的列表
    if not props.sample_with_replacement:
        return [(sku, {}) for sku in shuffle_bag]

    # 如果是替换采样，则生成一个更长的列表
    result = []
    # 生成足够多的商品组（假设最多需要100组）
    for _ in range(100):
        if not shuffle_bag:
            # 袋子空了，重新洗牌
            shuffle_bag = skus[:]
            rng.shuffle(shuffle_bag)
        
        # 从袋中随机抽取一个商品
        sku = shuffle_bag.pop()
        result.append((sku, {}))
    
    return result


def create_sku_instance(source_obj, target_collection):
    """创建SKU实例"""
    new_obj = source_obj.copy()
    if source_obj.data:
        new_obj.data = source_obj.data
    new_obj.hide_render = False
    new_obj.hide_viewport = False
    target_collection.objects.link(new_obj)
    return new_obj


# ============================================================================
# 碰撞检测 (collision.py)
# ============================================================================

def check_floor(scene, depsgraph, center_pos, shelf_obj):
    """
    检测商品位置下方是否有有效的货架表面
    
    通过向下发射射线，验证商品是否放置在货架层板上，避免：
    - 商品超出货架边界
    - 商品与货架背板穿模
    - 商品悬空
    
    Args:
        scene: Blender场景
        depsgraph: 评估后的依赖图
        center_pos: 商品中心位置
        shelf_obj: 货架对象
    
    Returns:
        bool: True表示位置有效，False表示无效
    """
    ray_start_lift = 0.05
    z_start = center_pos.z + ray_start_lift
    MAX_FLOAT_TOLERANCE = 0.01
    ray_length = ray_start_lift + MAX_FLOAT_TOLERANCE + 0.005
    
    shelf_original = shelf_obj.original if hasattr(shelf_obj, "original") else shelf_obj
    
    start_p = Vector((center_pos.x, center_pos.y, z_start))
    res, loc, norm, _, obj, _ = scene.ray_cast(
        depsgraph, start_p, Vector((0, 0, -1)), distance=ray_length
    )
    
    if not res:
        return False
    
    hit_obj = obj.original if hasattr(obj, "original") else obj
    if hit_obj != shelf_original:
        return False
    
    # 检查法线是否朝上（水平表面）
    if abs(norm.z) < 0.8:
        return False
    
    # 检查距离是否在合理范围内
    diff = center_pos.z - loc.z
    return -0.03 <= diff <= MAX_FLOAT_TOLERANCE


# ============================================================================
# 几何计算 (geometry.py)
# ============================================================================

def get_z_offset(obj):
    """
    计算物体底部相对于原点的Z轴偏移量
    返回值为正数，表示物体底部在原点下方多远
    """
    bbox = [Vector(v) for v in obj.bound_box]
    min_z_local = min([v.z for v in bbox])
    return abs(min_z_local) * obj.scale.z


def get_shelf_axes(shelf_obj):
    """
    确定货架的长度轴和深度轴（世界坐标系）
    
    基于局部尺寸判断哪个轴更长，然后转换到世界坐标系。
    这样可以正确处理任意旋转的货架。
    
    Args:
        shelf_obj: 货架对象
    
    Returns:
        tuple: (length_axis, depth_axis, max_length, max_depth, center)
            - length_axis: 长度方向的单位向量（世界坐标）
            - depth_axis: 深度方向的单位向量（世界坐标）
            - max_length: 货架长度
            - max_depth: 货架深度
            - center: 货架中心点（世界坐标）
    """
    # 获取局部尺寸
    local_dims = shelf_obj.dimensions
    
    # 判断哪个局部轴更长（X vs Y）
    is_local_x_long = (local_dims.x >= local_dims.y)
    
    # 获取货架中心点
    center = shelf_obj.matrix_world.to_translation()
    
    # 将局部轴转换到世界坐标系
    shelf_matrix_3x3 = shelf_obj.matrix_world.to_3x3()
    local_x_world = shelf_matrix_3x3 @ Vector((1, 0, 0))
    local_y_world = shelf_matrix_3x3 @ Vector((0, 1, 0))
    
    # 投影到水平面（忽略Z分量）
    local_x_world.z = 0
    local_y_world.z = 0
    local_x_world = local_x_world.normalized()
    local_y_world = local_y_world.normalized()
    
    # 根据长短轴分配长度和深度
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
    
    return length_axis, depth_axis, max_length, max_depth, center


def detect_shelf_front(shelf_obj, context, depth_axis, center, max_depth):
    """
    自动检测货架的前方（开口方向）
    
    通过向深度轴两个方向发射【水平】射线，检测哪一侧更早碰到垂直面（背板）。
    背板命中距离更短的方向 = 背板方向，翻转后使depth_axis指向开口前方。
    
    原先使用垂直向下射线，层板水平面在正负两侧均会命中，导致判断失效。
    改用水平射线后，背板（垂直面，法线水平）和开口（无命中/远距）有明显区别。
    
    Args:
        shelf_obj: 货架对象
        context: Blender context
        depth_axis: 初始深度轴方向
        center: 货架中心点
        max_depth: 货架深度
    
    Returns:
        Vector: 修正后的深度轴方向（指向前方）
    """
    scene = context.scene
    depsgraph = context.evaluated_depsgraph_get()
    shelf_original = shelf_obj.original if hasattr(shelf_obj, "original") else shelf_obj
    
    # 在货架高度范围内多个高度采样，提高鲁棒性
    corners = [shelf_obj.matrix_world @ Vector(c) for c in shelf_obj.bound_box]
    z_min = min(v.z for v in corners)
    z_max = max(v.z for v in corners)
    sample_heights = [z_min + (z_max - z_min) * t for t in (0.3, 0.5, 0.7)]
    
    max_ray_dist = max_depth * 1.5
    
    def hit_dist_horizontal(direction, z):
        """从中心以给定水平方向发射射线，返回命中垂直面的距离（未命中返回999）"""
        origin = Vector((center.x, center.y, z))
        res, loc, norm, _, obj, _ = scene.ray_cast(
            depsgraph, origin, direction, distance=max_ray_dist
        )
        if not res:
            return 999.0
        hit_obj = obj.original if hasattr(obj, "original") else obj
        if hit_obj != shelf_original:
            return 999.0
        # 判断命中面是否为垂直面（背板法线接近水平，abs(norm.z) < 0.5）
        if abs(norm.z) < 0.5:
            return (Vector(loc) - origin).length
        return 999.0
    
    # 统计各高度层在正/负方向命中背板的距离
    pos_dists = [hit_dist_horizontal(depth_axis, z) for z in sample_heights]
    neg_dists = [hit_dist_horizontal(-depth_axis, z) for z in sample_heights]
    
    # 取中位距离进行比较，距离更小的方向 = 背板方向
    pos_dists.sort()
    neg_dists.sort()
    pos_median = pos_dists[len(pos_dists) // 2]
    neg_median = neg_dists[len(neg_dists) // 2]
    
    # 正方向更近 = 正方向是背板 → 翻转，使depth_axis指向前方（开口）
    if pos_median < neg_median:
        return -depth_axis
    
    return depth_axis


def detect_shelf_levels(shelf_coll, context=None):
    """
    检测货架层板的可放置表面
    简化版：将每个mesh对象视为一个层板，使用其包围盒顶面
    """
    levels = []
    shelves = [obj for obj in shelf_coll.objects if obj.type == 'MESH']
    
    for obj in shelves:
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        
        xs = [v.x for v in corners]
        ys = [v.y for v in corners]
        zs = [v.z for v in corners]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        top_z = max(zs)
        
        if (max_x - min_x) > 0.05 and (max_y - min_y) > 0.05:
            levels.append({
                'z': top_z,
                'xmin': min_x, 'xmax': max_x,
                'ymin': min_y, 'ymax': max_y,
                'obj': obj
            })
    
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


def is_cylinder(obj, vert_threshold=12, height_tolerance_ratio=0.01):
    """
    判断一个物体是否为圆柱体 (优化版)

    通过快速分析其顶部和底部的顶点分布来判断。
    1. 找出物体的最高和最低点 (Z轴)。
    2. 识别出所有位于顶部和底部的顶点。
    3. 检查是否存在一个完全由这些顶部或底部顶点组成的面。
    4. 如果这个面的顶点数超过阈值，则认为是圆柱体。

    这个方法比逐面计算法线更高效，尤其对于高精度模型。

    Args:
        obj: 要检查的物体
        vert_threshold: 构成圆形顶/底面的最小顶点数。降低此值可放宽标准。
        height_tolerance_ratio: 用于确定“顶部”/“底部”顶点的容差范围，基于物体总高度的比例。

    Returns:
        bool: 如果是圆柱体则为True，否则为False
    """
    if not obj or obj.type != 'MESH':
        return False

    mesh = obj.data
    if not mesh.vertices or not mesh.polygons:
        return False

    # 1. 找出本地坐标中的最高和最低Z值
    # 使用numpy可以更快，但为了避免依赖，使用常规Python
    verts_co = [v.co for v in mesh.vertices]
    if not verts_co:
        return False
        
    min_z = min(v.z for v in verts_co)
    max_z = max(v.z for v in verts_co)
    
    height = max_z - min_z
    if height < 0.0001:
        return False # A flat plane

    # 2. 识别顶部和底部的顶点
    # 使用一个小的容差来处理不完全平坦的表面
    tolerance = height * height_tolerance_ratio
    top_vert_indices = {i for i, v in enumerate(verts_co) if (max_z - v.z) < tolerance}
    bottom_vert_indices = {i for i, v in enumerate(verts_co) if (v.z - min_z) < tolerance}

    if not top_vert_indices and not bottom_vert_indices:
        return False

    # 3. 检查是否有由顶部或底部顶点组成的多边形
    # 这比检查所有面的法线要快
    for poly in mesh.polygons:
        poly_verts = set(poly.vertices)
        
        # 检查是否为顶面或底面
        is_top_face = poly_verts.issubset(top_vert_indices)
        is_bottom_face = poly_verts.issubset(bottom_vert_indices)

        if is_top_face or is_bottom_face:
            # 4. 如果面的顶点数足够多，则认为是圆柱体
            if len(poly.vertices) > vert_threshold:
                return True

    return False


def compute_segment_width(sku, facing, spacing):
    """计算segment宽度"""
    return facing * get_obj_width(sku) + (facing - 1) * spacing


def compute_position(x_cursor_local, bounds, sku, i, d, props):
    """
    计算SKU放置位置（使用货架坐标系统）
    
    Args:
        x_cursor_local: 当前位置沿货架长度轴的局部坐标
        bounds: 货架层板边界信息（包含方向轴）
        sku: 商品对象
        i: 排面索引（沿长度方向）
        d: 深度索引（沿深度方向）
        props: 属性
    
    Returns:
        tuple: (x, y, z) 世界坐标位置
    """
    width = get_obj_width(sku)
    depth = get_obj_depth(sku)
    
    # 获取货架方向信息
    length_axis = bounds.get('length_axis')
    depth_axis = bounds.get('depth_axis')
    center = bounds.get('center')
    max_depth = bounds.get('max_depth', 0)
    
    # 如果有方向信息，使用货架坐标系统
    if length_axis is not None and depth_axis is not None and center is not None:
        # 计算沿长度轴的偏移（从货架中心开始）
        length_offset = x_cursor_local + width / 2.0 + i * (width + props.horizontal_spacing)
        
        # 计算沿深度轴的偏移（从前边缘向后）
        depth_offset = max_depth / 2.0 - props.edge_margin - depth / 2.0 - d * depth
        
        # 转换到世界坐标
        world_pos = (
            center +
            length_axis * length_offset +
            depth_axis * depth_offset
        )
        
        return (world_pos.x, world_pos.y, bounds['z'])
    
    # 回退到旧方法（如果缺少方向信息）
    x = x_cursor_local + width / 2.0 + i * (width + props.horizontal_spacing)
    shelf_depth = bounds['ymax'] - bounds['ymin']
    y = bounds['ymax'] - props.edge_margin - depth / 2.0 - d * depth
    z = bounds['z']
    
    return (x, y, z)


def place_object(obj, pos, rotation_z=0.0):
    """
    放置物体，确保底部与表面对齐
    
    Args:
        obj: 要放置的物体
        pos: 位置 (x, y, z)
        rotation_z: Z轴旋转角度（弧度）
    """
    z_offset = get_z_offset(obj)
    obj.location.x = pos[0]
    obj.location.y = pos[1]
    obj.location.z = pos[2] + z_offset
    obj.rotation_euler.z = rotation_z

def create_segment_label(text, position, depth_axis, level_z, props, target_collection, seg_center_local, max_length):
    """
    创建segment标签（矩形价格签），垂直于层板外侧
    
    Args:
        text: 标签名称（用于对象命名）
        position: 标签位置（世界坐标，segment中心）
        depth_axis: 货架深度轴方向（指向前方）
        level_z: 层板Z坐标
        props: 属性
        target_collection: 目标集合
        seg_center_local: segment中心沿货架长度轴的局部坐标
        max_length: 货架最大长度
    
    Returns:
        创建的标签对象
    """
    # 创建矩形网格（价格签）
    width = props.label_width
    height = props.label_height
    
    # 创建平面网格数据
    mesh = bpy.data.meshes.new(f"PriceTag_{text}")
    
    # 定义矩形顶点（在XZ平面，Y=0）
    # 标签中心在原点，宽度沿X轴，高度沿Z轴
    hw = width / 2.0  # 半宽
    verts = [
        (-hw, 0, 0),        # 左下
        (hw, 0, 0),         # 右下
        (hw, 0, height),    # 右上
        (-hw, 0, height),   # 左上
    ]
    faces = [(0, 1, 2, 3)]
    
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    
    # 创建对象
    label_obj = bpy.data.objects.new(f"PriceTag_{text}", mesh)
    target_collection.objects.link(label_obj)
    
    # 计算标签位置（在层板前边缘外侧）
    # 添加水平随机偏移
    horizontal_offset_dist = 0.0
    if props.label_random_horizontal_offset > 0.0:
        horizontal_offset_dist = random.uniform(
            -props.label_random_horizontal_offset, 
            props.label_random_horizontal_offset
        )
        
        # 限制偏移不超过货架左右边界
        # 货架的局部 X 范围是 [-max_length/2, max_length/2]
        # 标签的最终坐标将会是 seg_center_local + horizontal_offset_dist
        tag_local_x = seg_center_local + horizontal_offset_dist
        hw = width / 2.0
        
        # 考虑边缘边距(edge_margin) 和 标签自身半宽(hw)
        min_allowed_x = -max_length / 2.0 + props.edge_margin + hw
        max_allowed_x = max_length / 2.0 - props.edge_margin - hw
        
        # Clamp (约束)
        if tag_local_x < min_allowed_x:
            horizontal_offset_dist = min_allowed_x - seg_center_local
        elif tag_local_x > max_allowed_x:
            horizontal_offset_dist = max_allowed_x - seg_center_local
    
    # 根据 depth_axis (向前) 和 货架的UP(Z)，计算水平向右的方向 (length_axis_right)
    # depth_axis cross Z = 右向向量
    right_axis = Vector((depth_axis.y, -depth_axis.x, 0.0)).normalized() # 顺时针旋转90度得到右向向量
    
    label_pos = Vector(position) + depth_axis * props.label_offset_depth + right_axis * horizontal_offset_dist
    label_pos.z = level_z + props.label_offset_updown
    label_obj.location = label_pos
    
    # 计算旋转，使标签垂直于层板，面向外侧
    # 标签平面法线初始指向+Y，需要旋转使其指向depth_axis方向
    angle_z = math.atan2(depth_axis.y, depth_axis.x)
    
    # 设置旋转：绕Z轴对准深度方向（+90度因为平面法线初始在+Y）
    label_obj.rotation_euler = Euler((0, 0, angle_z + math.radians(90)), 'XYZ')
    
    # 创建或获取价签材质
    mat_name = "PriceTag_Material"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # 设置为白色
            bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    
    # 应用材质
    if label_obj.data.materials:
        label_obj.data.materials[0] = mat
    else:
        label_obj.data.materials.append(mat)
    
    return label_obj


def is_top_surface(scene, depsgraph, hit_loc, shelf_obj, max_thickness=0.06):
    """
    判断命中点是否为层板上表面（而非底面）
    
    通过从命中点向上发射短距离射线，检查是否有同一货架的遮挡。
    max_thickness 设为 0.06m，可处理最厚约 6cm 的层板。
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


def check_headroom(scene, depsgraph, center_pos, shelf_obj, headroom_cache=None):
    """
    检测商品上方的可用空间（headroom）
    
    从商品位置向上发射射线，检测到上层层板或货架顶部的距离
    
    Args:
        scene: Blender场景
        depsgraph: 评估后的依赖图
        center_pos: 商品中心位置
        shelf_obj: 货架对象
        headroom_cache: 可选的缓存字典，用于加速重复检测
    
    Returns:
        float: 可用高度（米），如果未检测到上方遮挡则返回999.0
    """
    # 使用缓存加速：同一层板的headroom相似
    cache_key = round(center_pos.z / 0.05) * 0.05
    if headroom_cache is not None and cache_key in headroom_cache:
        return headroom_cache[cache_key]
    
    shelf_original = shelf_obj.original if hasattr(shelf_obj, "original") else shelf_obj
    start_p = center_pos + Vector((0, 0, 0.02))
    ray_dir = Vector((0, 0, 1))
    
    min_headroom = 999.0
    
    for _ in range(3):  # 最多重试3次
        res, loc, norm, _, obj, _ = scene.ray_cast(depsgraph, start_p, ray_dir)
        if not res:
            break
        
        hit_obj = obj.original if hasattr(obj, "original") else obj
        
        # 跳过非货架对象
        if hit_obj != shelf_original:
            start_p = loc + Vector((0, 0, 0.002))
            continue
        
        # 跳过非水平表面
        if abs(norm.z) < 0.8:
            start_p = loc + Vector((0, 0, 0.002))
            continue
        
        dist = loc.z - center_pos.z
        if dist > 0.03 and dist < min_headroom:
            min_headroom = dist
        break
    
    # 如果没有检测到上层，检查货架顶部
    if min_headroom > 900:
        shelf_top_z = max([(shelf_obj.matrix_world @ Vector(c)).z for c in shelf_obj.bound_box])
        min_headroom = max(0.0, shelf_top_z - center_pos.z)
    
    # 缓存结果
    if headroom_cache is not None:
        headroom_cache[cache_key] = min_headroom
    
    return min_headroom


def detect_shelf_levels_by_ray(shelf_obj, context, sample_density=5):
    """
    通过射线检测货架的多个层板
    
    从货架顶部向下发射垂直射线，循环检测每一层层板表面。
    
    Args:
        shelf_obj: 货架对象
        context: Blender context
        sample_density: 每个方向的采样点数量
    
    Returns:
        list[dict]: 每层的信息，按Z从高到低排序
    """
    scene = context.scene
    depsgraph = context.evaluated_depsgraph_get()
    
    # 获取货架包围盒边界（用于返回准确的 xmin/xmax/ymin/ymax）
    corners = [shelf_obj.matrix_world @ Vector(c) for c in shelf_obj.bound_box]
    bbox_min_x = min(v.x for v in corners)
    bbox_max_x = max(v.x for v in corners)
    bbox_min_y = min(v.y for v in corners)
    bbox_max_y = max(v.y for v in corners)
    min_z = min(v.z for v in corners)
    max_z = max(v.z for v in corners)
    
    # 获取货架方向轴信息
    length_axis, depth_axis, max_length, max_depth, center = get_shelf_axes(shelf_obj)
    
    # 自动检测并修正深度轴方向（确保指向前方）
    depth_axis = detect_shelf_front(shelf_obj, context, depth_axis, center, max_depth)
    
    shelf_original = shelf_obj.original if hasattr(shelf_obj, "original") else shelf_obj
    
    levels = {}  # z_key -> {'z': z, 'hit_count': int}
    
    # 在包围盒范围内采样
    step_x = (bbox_max_x - bbox_min_x) / sample_density
    step_y = (bbox_max_y - bbox_min_y) / sample_density
    
    for i in range(sample_density):
        for j in range(sample_density):
            base_x = bbox_min_x + step_x * (i + 0.5)
            base_y = bbox_min_y + step_y * (j + 0.5)
            
            cur_z = max_z + 0.5
            safety = 20
            
            while safety > 0 and cur_z > min_z:
                safety -= 1
                ray_orig = Vector((base_x, base_y, cur_z))
                success, loc, norm, _, obj, _ = scene.ray_cast(
                    depsgraph, ray_orig, Vector((0, 0, -1))
                )
                
                if not success:
                    break
                
                hit_obj = obj.original if hasattr(obj, "original") else obj
                
                if hit_obj == shelf_original and abs(norm.z) > 0.8:
                    if is_top_surface(scene, depsgraph, loc, shelf_obj):
                        # 用 0.05m 精度分组，避免同一层板因浮点误差被拆成多层
                        z_key = round(loc.z / 0.05) * 0.05
                        if z_key not in levels:
                            levels[z_key] = {
                                'z': loc.z,
                                'hit_count': 0
                            }
                        levels[z_key]['hit_count'] += 1
                        # 命中上表面后，跳过整块层板厚度再继续向下
                        cur_z = loc.z - 0.10
                        continue
                
                # 未命中层板上表面时，小步下移继续搜索
                cur_z = loc.z - 0.02
    
    # 转换为结果列表，使用包围盒边界而不是采样点范围
    # 最少命中次数阈值：要求至少 3 条射线命中，过滤侧板/背板等噪声
    min_hit_count = max(3, int(sample_density * sample_density * 0.15))
    result = []
    for z_key, data in levels.items():
        if data['hit_count'] >= min_hit_count:
            result.append({
                'z': data['z'],
                'xmin': bbox_min_x,
                'xmax': bbox_max_x,
                'ymin': bbox_min_y,
                'ymax': bbox_max_y,
                'obj': shelf_obj,  # 保留货架对象引用
                # 添加方向信息
                'length_axis': length_axis,
                'depth_axis': depth_axis,
                'max_length': max_length,
                'max_depth': max_depth,
                'center': center
            })
    
    return sorted(result, key=lambda l: l['z'], reverse=True)


# ============================================================================
# 放置逻辑 (placement.py)
# ============================================================================

def generate_planogram(context, props):
    """生成陈列布局"""
    random_seed = props.random_seed
    if props.random_seed == -1:
        random_seed = random.randint(0, 1000)
    random.seed(random_seed)
    
    product_coll = bpy.data.collections.get(props.product_collection)
    shelf_coll = bpy.data.collections.get(props.shelf_collection)
    
    if not product_coll or not shelf_coll:
        report_error("Invalid collection selection")
        return
    
    # ----------------------------------------------------------------
    # scene.ray_cast() 不会命中视口隐藏的物体。
    # 在整个生成过程（层板检测 + 商品摆放）中临时强制显示所有货架对象，
    # 函数结束后统一还原，确保两个阶段的射线检测都能正常工作。
    # ----------------------------------------------------------------
    shelves_all = [obj for obj in shelf_coll.objects if obj.type == 'MESH']
    visibility_backup = {}  # obj -> (hide_viewport, hide_get)
    coll_hide_backup = shelf_coll.hide_viewport
    shelf_coll.hide_viewport = False
    for _s in shelves_all:
        visibility_backup[_s] = (_s.hide_viewport, _s.hide_get())
        _s.hide_viewport = False
        _s.hide_set(False)
    context.view_layer.update()
    
    def _restore_visibility():
        shelf_coll.hide_viewport = coll_hide_backup
        for _s, (_hv, _hg) in visibility_backup.items():
            _s.hide_viewport = _hv
            _s.hide_set(_hg)
        context.view_layer.update()
    
    # 清理已存在的SM_开头的集合
    sm_collections = [coll for coll in bpy.data.collections if coll.name.startswith("SM_")]
    if sm_collections:
        print(f"\n清理 {len(sm_collections)} 个已存在的SM_集合...")
        for coll in sm_collections:
            # 删除集合中的所有对象
            for obj in list(coll.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            # 从场景中移除集合
            if coll.name in context.scene.collection.children:
                context.scene.collection.children.unlink(coll)
            # 删除集合
            bpy.data.collections.remove(coll)
        print("清理完成")
    
    # 创建目标集合
    target_coll_name = "SM_Planogram"
    target_coll = bpy.data.collections.new(target_coll_name)
    context.scene.collection.children.link(target_coll)
    
    # 使用射线检测获取所有货架的所有层板
    shelf_levels = []
    shelves = [obj for obj in shelf_coll.objects if obj.type == 'MESH']
    
    print(f"\n开始检测 {len(shelves)} 个货架的层板...")
    for shelf in shelves:
        levels = detect_shelf_levels_by_ray(shelf, context, sample_density=5)
        print(f"  货架 {shelf.name}: 检测到 {len(levels)} 个层板")
        shelf_levels.extend(levels)
    
    print(f"总计检测到 {len(shelf_levels)} 个层板\n")
    
    if not shelf_levels:
        report_error("未检测到任何可用层板")
        _restore_visibility()
        return
    
    # 获取SKU列表
    skus = [obj for obj in product_coll.objects if obj.type == 'MESH']
    skus = sorted(skus, key=lambda o: o.name)
    
    if not skus:
        report_error("商品集合中没有可用商品")
        _restore_visibility()
        return
    
    # 按填充顺序处理每层
    fill_order = shelf_levels if props.fill_order == 'TOP_DOWN' else list(reversed(shelf_levels))
    
    # 创建 headroom 缓存以提升性能
    headroom_cache = {}
    depsgraph = context.evaluated_depsgraph_get()
    
    total_placed = 0
    total_labels = 0
    
    for level_idx, level in enumerate(fill_order):
        bounds = get_shelf_bounds(level)
        segments = segment_skus(skus, props)
        
        # 获取货架长度，使用局部坐标系统
        max_length = bounds.get('max_length', bounds['xmax'] - bounds['xmin'])
        
        # 获取方向信息（用于标签和商品旋转）
        length_axis = bounds.get('length_axis')
        depth_axis = bounds.get('depth_axis')
        center = bounds.get('center')
        max_depth = bounds.get('max_depth', 0)
        
        # 计算货架的Z轴旋转角度（根据深度轴方向 + 90度）
        shelf_rotation_z = 0.0
        if depth_axis is not None:
            shelf_rotation_z = math.atan2(depth_axis.y, depth_axis.x) + math.radians(90)
        
        # 从货架中心开始的局部坐标光标
        x_cursor_local = -max_length / 2.0 + props.edge_margin
        
        level_placed = 0
        level_segments = []  # 记录本层的segment信息用于生成标签
        
        for sku, seg_cfg in segments:
            sku_width = get_obj_width(sku)
            
            # 检查是否超出货架长度
            if x_cursor_local + sku_width > max_length / 2.0 - props.edge_margin:
                break
            
            facing = random.randint(props.min_facing, props.max_facing)
            depth = random.randint(props.min_depth, props.max_depth)
            
            # 计算可用长度（局部坐标）
            available_length = max_length / 2.0 - props.edge_margin - x_cursor_local
            max_possible_facing = int(available_length // (sku_width + props.horizontal_spacing))
            
            if max_possible_facing < 1:
                if available_length >= sku_width:
                    max_possible_facing = 1
                else:
                    break
            
            if props.allow_partial:
                facing = min(facing, max_possible_facing)
            elif facing > max_possible_facing:
                break
            
            seg_width = compute_segment_width(sku, facing, props.horizontal_spacing)
            
            # 记录segment起始位置（用于标签）
            seg_start_local = x_cursor_local
            segment_has_items = False
            
            for i in range(facing):
                for d in range(depth):
                    pos_raw = compute_position(x_cursor_local, bounds, sku, i, d, props)
                    
                    # 添加随机位置偏移
                    if props.position_noise > 0.0:
                        offset_x = random.uniform(-props.position_noise, props.position_noise)
                        offset_y = random.uniform(-props.position_noise, props.position_noise)
                        
                        if length_axis is not None and depth_axis is not None:
                            offset_vec = length_axis * offset_x + depth_axis * offset_y
                            pos = (pos_raw[0] + offset_vec.x, pos_raw[1] + offset_vec.y, pos_raw[2])
                        else:
                            pos = (pos_raw[0] + offset_x, pos_raw[1] + offset_y, pos_raw[2])
                    else:
                        pos = pos_raw
                    
                    # 边界检查
                    if pos[0] - sku_width/2 < bounds['xmin'] + props.edge_margin:
                        continue
                    if pos[0] + sku_width/2 > bounds['xmax'] - props.edge_margin:
                        continue
                    
                    # 获取商品尺寸信息
                    sku_height = sku.dimensions.z
                    sku_z_offset = get_z_offset(sku)
                    
                    # 使用射线检测验证位置有效性（避免穿模和超出边界）
                    center_pos = Vector((pos[0], pos[1], pos[2] + sku_z_offset))
                    shelf_obj = bounds.get('obj')
                    
                    if not shelf_obj:
                        continue
                    
                    # 检测下方是否有有效的货架表面
                    if not check_floor(context.scene, depsgraph, center_pos, shelf_obj):
                        continue
                    
                    segment_has_items = True
                    
                    # 计算堆叠数量
                    stack_count = 1
                    if props.use_stacking:
                        headroom = check_headroom(
                            context.scene, depsgraph, center_pos, shelf_obj, headroom_cache
                        )
                        
                        # 应用顶部留空比例
                        safe_limit = headroom * (1.0 - props.top_gap_ratio)
                        usable_height = max(0.0, safe_limit - sku_z_offset)
                        
                        # 计算可堆叠数量
                        if sku_height > 0:
                            stack_count = min(
                                max(1, int(usable_height / sku_height)),
                                props.max_stack
                            )
                    
                    # 垂直堆叠放置
                    for k in range(stack_count):
                        stack_pos = (pos[0], pos[1], pos[2] + k * sku_height)
                        new_obj = create_sku_instance(sku, target_coll)
                        
                        # 默认旋转等于货架旋转
                        rotation_z = shelf_rotation_z
                        
                        # 如果是圆柱体，则应用随机旋转
                        if is_cylinder(sku):
                            random_rotation = random.uniform(-props.rotation_z_range, props.rotation_z_range)
                            rotation_z += random_rotation
                        
                        place_object(new_obj, stack_pos, rotation_z)
                        level_placed += 1
                        total_placed += 1
            
            # 记录segment信息用于生成标签
            if segment_has_items and props.use_labels and length_axis is not None and depth_axis is not None and center is not None:
                # 计算segment中心位置（世界坐标）
                seg_center_local = seg_start_local + seg_width / 2.0
                seg_center_world = (
                    center +
                    length_axis * seg_center_local +
                    depth_axis * (max_depth / 2.0)  # 在前边缘
                )
                level_segments.append({
                    'sku_name': sku.name,
                    'center': seg_center_world,
                    'depth_axis': depth_axis,
                    'seg_center_local': seg_center_local
                })
            
            # 更新光标位置（局部坐标）
            x_cursor_local += seg_width + props.segment_spacing
        
        # 为本层的每个segment生成标签
        if props.use_labels and level_segments:
            for seg_info in level_segments:
                create_segment_label(
                    text=seg_info['sku_name'],
                    position=seg_info['center'],
                    depth_axis=seg_info['depth_axis'],
                    level_z=bounds['z'],
                    props=props,
                    target_collection=target_coll,
                    seg_center_local=seg_info['seg_center_local'],
                    max_length=max_length
                )
                total_labels += 1
        
        print(f"层板 {level_idx + 1} (Z={bounds['z']:.3f}m): 放置了 {level_placed} 个商品, {len(level_segments)} 个标签")
    
    print(f"\n总计放置了 {total_placed} 个商品, {total_labels} 个标签")
    
    # 还原货架可见性
    _restore_visibility()


# ============================================================================
# UI (ui.py)
# ============================================================================

class PlanogramProperties(PropertyGroup):
    product_collection: StringProperty(name="Product Collection")
    shelf_collection: StringProperty(name="Shelf Collection")
    min_facing: IntProperty(name="Min Facing", default=2, min=1)
    max_facing: IntProperty(name="Max Facing", default=3, min=1)
    min_depth: IntProperty(name="Min Depth", default=1, min=1)
    max_depth: IntProperty(name="Max Depth", default=3, min=1)
    horizontal_spacing: FloatProperty(name="组内空隙", default=0.001, min=0.0)
    segment_spacing: FloatProperty(name="组间空隙", default=0.001, min=0.0)
    edge_margin: FloatProperty(name="边距", default=0.02, min=0.0)
    align_mode: EnumProperty(
        name="对齐方式",
        items=[('LEFT', "Left", ""), ('CENTER', "Center", ""), ('RIGHT', "Right", "")]
    )
    fill_order: EnumProperty(
        name="填充顺序",
        items=[('TOP_DOWN', "从上到下", ""), ('BOTTOM_UP', "从下到上", "")]
    )
    random_seed: IntProperty(name="随机种子", default=42)
    allow_partial: BoolProperty(name="允许可部分填充（补满排面）", default=True)
    sample_with_replacement: BoolProperty(name="重复抽取商品", default=True, description="允许在商品清单用完后重复抽取（替换采样）")
    use_stacking: BoolProperty(name="允许垂直堆叠", default=False)
    max_stack: IntProperty(name="最大堆叠数", default=3, min=1, max=10)
    top_gap_ratio: FloatProperty(name="顶部留空比例", default=0.05, min=0.0, max=0.5, description="上方预留空间占headroom的比例")
    rotation_z_range: FloatProperty(name="Z轴旋转范围", default=180.0, min=0.0, max=180.0, subtype='ANGLE', description="商品Z轴随机旋转的最大角度（实际旋转范围为±此值）")
    position_noise: FloatProperty(name="位置随机轻微偏移", default=0.0, min=0.0, max=0.1, description="商品位置水平产生的随机轻微偏移范围（最大距离）")
    
    # 标签设置
    use_labels: BoolProperty(name="生成标签", default=False, description="为每个segment生成价格标签")
    label_width: FloatProperty(name="标签宽度", default=0.05, min=0.01, max=0.3, description="价格标签的宽度")
    label_height: FloatProperty(name="标签高度", default=0.03, min=0.01, max=0.2, description="价格标签的高度")
    label_offset_depth: FloatProperty(name="标签前后偏移", default=0.0, min=0.0, max=0.5, description="标签距离层板前边缘的距离")
    label_offset_updown: FloatProperty(name="标签上下偏移", default=0.0, min=-0.5, max=0.5, description="标签距离层板前边缘的距离")
    label_random_horizontal_offset: FloatProperty(name="标签水平随机偏移", default=0.0, min=0.0, max=0.5, description="标签在水平方向随机偏移的最大范围（±该值）")

    # UI折叠状态
    show_basic: BoolProperty(name="基础设置", default=True)
    show_layout: BoolProperty(name="布局参数", default=False)
    show_stacking: BoolProperty(name="堆叠设置", default=False)
    show_rotation: BoolProperty(name="随机变换设置", default=False)
    show_labels: BoolProperty(name="标签设置", default=False)
    show_debug: BoolProperty(name="调试工具", default=False)
    
    # 清理设置
    auto_clean_unused: BoolProperty(name="清空后自动清理未使用数据", default=True, description="执行清空操作后，自动清理场景中未使用的冗余数据块(如材质、网格等)")


class PLANOGRAM_PT_panel(Panel):
    bl_label = "Planogram Test"
    bl_idname = "PLANOGRAM_PT_panel_test"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Planogram"

    def draw(self, context):
        props = context.scene.planogram_props_test
        layout = self.layout
        
        # 版本信息
        box = layout.box()
        box.label(text=f"版本: v{bl_info['version'][0]}.{bl_info['version'][1]}.{bl_info['version'][2]}", icon='INFO')
        
        # 基础设置（可折叠）
        box = layout.box()
        row = box.row()
        row.prop(props, "show_basic", icon='TRIA_DOWN' if props.show_basic else 'TRIA_RIGHT', emboss=False)
        if props.show_basic:
            box.prop_search(props, "product_collection", bpy.data, "collections", text="商品集合")
            box.prop_search(props, "shelf_collection", bpy.data, "collections", text="货架集合")
            box.prop(props, "random_seed")
            box.prop(props, "fill_order")
            box.prop(props, "allow_partial")
            box.prop(props, "sample_with_replacement")
        
        # 布局参数（可折叠）
        box = layout.box()
        row = box.row()
        row.prop(props, "show_layout", icon='TRIA_DOWN' if props.show_layout else 'TRIA_RIGHT', emboss=False)
        if props.show_layout:
            col = box.column(align=True)
            col.label(text="排面和深度:", icon='ALIGN_JUSTIFY')
            col.prop(props, "min_facing")
            col.prop(props, "max_facing")
            col.prop(props, "min_depth")
            col.prop(props, "max_depth")
            
            box.separator()
            col = box.column(align=True)
            col.label(text="间距设置:", icon='ARROW_LEFTRIGHT')
            col.prop(props, "horizontal_spacing")
            col.prop(props, "segment_spacing")
            col.prop(props, "edge_margin")
        
        # 堆叠设置（可折叠）
        box = layout.box()
        row = box.row()
        row.prop(props, "show_stacking", icon='TRIA_DOWN' if props.show_stacking else 'TRIA_RIGHT', emboss=False)
        if props.show_stacking:
            box.prop(props, "use_stacking")
            if props.use_stacking:
                col = box.column(align=True)
                col.prop(props, "max_stack")
                col.prop(props, "top_gap_ratio")
        
        # 随机变换设置（可折叠）
        box = layout.box()
        row = box.row()
        row.prop(props, "show_rotation", icon='TRIA_DOWN' if props.show_rotation else 'TRIA_RIGHT', emboss=False)
        if props.show_rotation:
            box.prop(props, "rotation_z_range")
            box.prop(props, "position_noise")
        
        # 标签设置（可折叠）
        box = layout.box()
        row = box.row()
        row.prop(props, "show_labels", icon='TRIA_DOWN' if props.show_labels else 'TRIA_RIGHT', emboss=False)
        if props.show_labels:
            box.prop(props, "use_labels")
            if props.use_labels:
                col = box.column(align=True)
                col.prop(props, "label_width")
                col.prop(props, "label_height")
                col.prop(props, "label_offset_depth")
                col.prop(props, "label_offset_updown")
                col.prop(props, "label_random_horizontal_offset")
        
        # 操作按钮
        layout.separator()
        layout.operator("planogram_test.layout", text="生成陈列布局", icon='PLAY')
        
        row = layout.row(align=True)
        row.operator("planogram_test.clear", text="清空陈列布局", icon='TRASH')
        row.prop(props, "auto_clean_unused", text="", icon='BRUSH_DATA')
        
        
        # 调试工具（可折叠）
        box = layout.box()
        row = box.row()
        row.prop(props, "show_debug", icon='TRIA_DOWN' if props.show_debug else 'TRIA_RIGHT', emboss=False)
        if props.show_debug:
            box.operator("planogram_test.detect_levels", text="检测选中货架层板", icon='OUTLINER_OB_MESH')
            box.operator("planogram_test.detect_collection_levels", text="检测集合所有层板", icon='COLLECTION_COLOR_01')
            box.operator("planogram_test.check_cylinder", text="检测是否为圆柱体", icon='MESH_CYLINDER')


class PLANOGRAM_OT_clear(Operator):
    bl_idname = "planogram_test.clear"
    bl_label = "Clear SM Collections"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        collections_to_remove = [c for c in bpy.data.collections if c.name.startswith("SM_")]
        
        for col in collections_to_remove:
            objs_to_remove = [o for o in col.objects]
            if objs_to_remove:
                bpy.data.batch_remove(objs_to_remove)
        
        if collections_to_remove:
            bpy.data.batch_remove(collections_to_remove)
            
        props = context.scene.planogram_props_test
        clean_msg = ""
        
        if props.auto_clean_unused:
            # 多次执行orphans_purge直到没有多余孤立数据
            purge_count = 0
            while True:
                result = bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
                # orphans_purge 实际上没有好的返回值告诉你清理了多少，但在API中它会在不再有东西可清理时停止
                # 为了安全且有效，我们循环3次通常就足以清除级联的孤立数据(比如先删除对象，后删除网格，再材质)
                purge_count += 1
                if purge_count >= 3:
                    break
            clean_msg = " 并已清理未使用数据块"
        
        self.report({'INFO'}, f"已清空 {len(collections_to_remove)} 个集合{clean_msg}")
        return {'FINISHED'}


class PLANOGRAM_OT_layout(Operator):
    bl_idname = "planogram_test.layout"
    bl_label = "Generate Planogram"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.planogram_props_test
        generate_planogram(context, props)
        self.report({'INFO'}, "陈列布局生成完成")
        return {'FINISHED'}


class PLANOGRAM_OT_detect_levels(Operator):
    """检测当前选中货架的层板"""
    bl_idname = "planogram_test.detect_levels"
    bl_label = "Detect Shelf Levels"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        shelf = context.active_object
        if not shelf or shelf.type != 'MESH':
            self.report({'ERROR'}, "请选择一个货架对象")
            return {'CANCELLED'}
        
        levels = detect_shelf_levels_by_ray(shelf, context, sample_density=5)
        
        if not levels:
            self.report({'WARNING'}, "未检测到任何层板")
            return {'CANCELLED'}
        
        msg = f"检测到 {len(levels)} 个层板:\n"
        for i, level in enumerate(levels):
            msg += f"  层 {i+1}: Z={level['z']:.3f}m\n"
        
        print(msg)
        self.report({'INFO'}, f"检测到 {len(levels)} 个层板，详情见控制台")
        return {'FINISHED'}


class PLANOGRAM_OT_detect_collection_levels(Operator):
    """检测货架集合中所有货架的层板"""
    bl_idname = "planogram_test.detect_collection_levels"
    bl_label = "Detect Collection Shelf Levels"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.planogram_props_test
        shelf_coll = bpy.data.collections.get(props.shelf_collection)
        
        if not shelf_coll:
            self.report({'ERROR'}, "请先选择货架集合")
            return {'CANCELLED'}
        
        shelves = [obj for obj in shelf_coll.objects if obj.type == 'MESH']
        
        if not shelves:
            self.report({'WARNING'}, "货架集合中没有网格对象")
            return {'CANCELLED'}
        
        print("\n" + "="*60)
        print(f"货架集合: {shelf_coll.name}")
        print("="*60)
        
        total_levels = 0
        for shelf in shelves:
            levels = detect_shelf_levels_by_ray(shelf, context, sample_density=5)
            total_levels += len(levels)
            
            print(f"\n货架: {shelf.name}")
            print(f"  检测到 {len(levels)} 个层板:")
            for i, level in enumerate(levels):
                print(f"    层 {i+1}: Z={level['z']:.3f}m, "
                      f"X=[{level['xmin']:.2f}, {level['xmax']:.2f}], "
                      f"Y=[{level['ymin']:.2f}, {level['ymax']:.2f}]")
        
        print("\n" + "="*60)
        print(f"总计: {len(shelves)} 个货架, {total_levels} 个层板")
        print("="*60 + "\n")
        
        self.report({'INFO'}, f"检测完成: {len(shelves)} 个货架, {total_levels} 个层板")
        return {'FINISHED'}


class PLANOGRAM_OT_check_cylinder(Operator):
    """检测选中物体是否为圆柱体"""
    bl_idname = "planogram_test.check_cylinder"
    bl_label = "Check if Cylinder"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj:
            self.report({'ERROR'}, "请先选择一个物体")
            return {'CANCELLED'}

        if is_cylinder(obj):
            self.report({'INFO'}, f"物体 '{obj.name}' 是一个圆柱体。")
        else:
            self.report({'INFO'}, f"物体 '{obj.name}' 不是一个圆柱体。")
        
        return {'FINISHED'}


# ============================================================================
# 注册
# ============================================================================

classes = [
    PlanogramProperties,
    PLANOGRAM_PT_panel,
    PLANOGRAM_OT_clear,
    PLANOGRAM_OT_layout,
    PLANOGRAM_OT_detect_levels,
    PLANOGRAM_OT_detect_collection_levels,
    PLANOGRAM_OT_check_cylinder,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.planogram_props_test = PointerProperty(type=PlanogramProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.planogram_props_test


if __name__ == "__main__":
    register()
