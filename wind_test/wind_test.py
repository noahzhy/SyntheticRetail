import bpy
import math
import numpy as np
from mathutils import Vector, Euler, Matrix
from mathutils.bvhtree import BVHTree

bl_info = {
    "name": "Wind Turbine Blade Inspection",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Blade Inspection",
    "description": "Automated drone inspection path generator for wind turbine blades",
    "category": "Animation",
}


class BladeInspectionProperties(bpy.types.PropertyGroup):
    """属性组用于存储叶片和相机选择"""
    blade1: bpy.props.PointerProperty(
        name="Blade 1",
        type=bpy.types.Object,
        description="Select first blade object"
    )
    blade2: bpy.props.PointerProperty(
        name="Blade 2",
        type=bpy.types.Object,
        description="Select second blade object"
    )
    blade3: bpy.props.PointerProperty(
        name="Blade 3",
        type=bpy.types.Object,
        description="Select third blade object"
    )
    camera: bpy.props.PointerProperty(
        name="Camera",
        type=bpy.types.Object,
        description="Select camera for inspection"
    )
    frames_per_face: bpy.props.IntProperty(
        name="Frames per Face",
        description="Number of frames for scanning one face",
        default=60,
        min=10,
        max=300
    )
    distance_from_blade: bpy.props.FloatProperty(
        name="Distance from Blade",
        description="Camera distance from blade surface (meters)",
        default=25.0,
        min=0.5,
        max=100.0
    )
    extension_ratio: bpy.props.FloatProperty(
        name="Extension Ratio",
        description="Extend scanning path beyond blade tips (0.1 = 10% extension)",
        default=0.15,
        min=0.0,
        max=0.5
    )
    transition_frames: bpy.props.IntProperty(
        name="Transition Frames",
        description="Number of frames for smooth transition between faces and blades",
        default=10,
        min=0,
        max=60
    )
    create_path_visualization: bpy.props.BoolProperty(
        name="Visualize Path",
        description="Create curve objects to visualize inspection paths",
        default=True
    )


class BLADE_OT_GenerateInspectionPath(bpy.types.Operator):
    """生成叶片巡检路径"""
    bl_idname = "blade.generate_inspection_path"
    bl_label = "Generate Inspection Path"
    bl_description = "Generate camera animation path for blade inspection"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.blade_inspection_props
        
        # 检查必要的对象是否已选择
        if not props.camera:
            self.report({'ERROR'}, "Please select a camera")
            return {'CANCELLED'}
        
        blades = [props.blade1, props.blade2, props.blade3]
        if not all(blades):
            self.report({'ERROR'}, "Please select all three blades")
            return {'CANCELLED'}
        
        camera = props.camera
        
        # 清除相机的现有动画
        if camera.animation_data:
            camera.animation_data_clear()
        
        # 删除之前的可视化路径
        if props.create_path_visualization:
            self.cleanup_old_paths()
        
        # 设置场景帧率和起始帧
        current_frame = 1
        bpy.context.scene.frame_start = 1
        
        # 存储所有路径点用于可视化
        all_path_points = []
        
        # 计算轮毂中心（三个叶片的公共旋转中心）
        hub_center = self.find_hub_center(blades)
        if hub_center:
            self.report({'INFO'}, f"Hub center found at: ({hub_center.x:.2f}, {hub_center.y:.2f}, {hub_center.z:.2f})")
        
        # 为每个叶片生成巡检路径
        last_blade_end_pos = None
        last_blade_end_quat = None
        
        for blade_idx, blade in enumerate(blades):
            if blade is None:
                continue
            
            # 叶片间过渡（除了第一个叶片）
            if blade_idx > 0 and last_blade_end_pos and props.transition_frames > 0:
                # 获取新叶片的起始位置
                next_root, next_tip, next_axis, _ = self.get_blade_axis_and_bounds(blade, hub_center)
                if next_root:
                    perp1, perp2 = self.get_perpendicular_vectors(next_axis, blade, hub_center)
                    next_start_pos = next_root + perp1 * props.distance_from_blade
                    next_target = next_root
                    
                    # 生成叶片间过渡
                    for trans_i in range(props.transition_frames):
                        t = (trans_i + 1) / (props.transition_frames + 1)
                        
                        trans_pos = last_blade_end_pos.lerp(next_start_pos, t)
                        camera.location = trans_pos
                        camera.keyframe_insert(data_path="location", frame=current_frame + trans_i)
                        
                        next_dir = next_target - next_start_pos
                        if next_dir.length > 0.001:
                            next_quat = next_dir.to_track_quat('-Z', 'Y')
                            trans_quat = last_blade_end_quat.slerp(next_quat, t)
                            camera.rotation_quaternion = trans_quat
                            camera.keyframe_insert(data_path="rotation_quaternion", frame=current_frame + trans_i)
                    
                    current_frame += props.transition_frames
                
            self.report({'INFO'}, f"Processing blade {blade_idx + 1}: {blade.name}")
            current_frame, blade_paths, end_pos, end_quat = self.generate_blade_inspection(
                context, blade, camera, current_frame, props, hub_center
            )
            if blade_paths:
                all_path_points.extend(blade_paths)
            
            # 保存这个叶片结束时的位置和旋转
            last_blade_end_pos = end_pos
            last_blade_end_quat = end_quat
        
        # 设置场景结束帧
        bpy.context.scene.frame_end = current_frame - 1
        bpy.context.scene.frame_set(1)
        
        # 创建路径可视化
        if props.create_path_visualization and all_path_points:
            self.create_path_curves(all_path_points)
        
        self.report({'INFO'}, f"Inspection path generated successfully! Total frames: {current_frame - 1}")
        return {'FINISHED'}
    
    def cleanup_old_paths(self):
        """清理之前创建的路径可视化对象和集合"""
        # 清理InspectionPaths集合
        collection_name = "InspectionPaths"
        if collection_name in bpy.data.collections:
            collection = bpy.data.collections[collection_name]
            # 删除集合中的所有对象
            for obj in list(collection.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            # 从场景中取消链接集合
            bpy.context.scene.collection.children.unlink(collection)
            # 删除集合
            bpy.data.collections.remove(collection)
        
        # 清理任何遗留的InspectionPath对象
        for obj in list(bpy.data.objects):
            if obj.name.startswith("InspectionPath_"):
                bpy.data.objects.remove(obj, do_unlink=True)
        
        # 清理未使用的curve数据
        for curve in bpy.data.curves:
            if curve.name.startswith("InspectionPath_") and curve.users == 0:
                bpy.data.curves.remove(curve)
    
    def create_path_curves(self, path_groups):
        """创建curve对象可视化巡检路径，并放入专门的集合中"""
        # 创建或获取InspectionPaths集合
        collection_name = "InspectionPaths"
        if collection_name in bpy.data.collections:
            collection = bpy.data.collections[collection_name]
        else:
            collection = bpy.data.collections.new(collection_name)
            # 将集合链接到场景
            bpy.context.scene.collection.children.link(collection)
        
        for idx, points in enumerate(path_groups):
            if len(points) < 2:
                continue
            
            # 创建curve数据
            curve_data = bpy.data.curves.new(f"InspectionPath_{idx}", type='CURVE')
            curve_data.dimensions = '3D'
            curve_data.resolution_u = 2
            curve_data.bevel_depth = 0.02  # 让路径有一定厚度
            
            # 创建spline
            polyline = curve_data.splines.new('POLY')
            polyline.points.add(len(points) - 1)
            
            # 设置点的坐标
            for i, point in enumerate(points):
                polyline.points[i].co = (point.x, point.y, point.z, 1)
            
            # 创建curve对象
            curve_obj = bpy.data.objects.new(f"InspectionPath_{idx}", curve_data)
            # 将对象链接到专门的集合
            collection.objects.link(curve_obj)
            
            # 设置渲染时不可见
            curve_obj.hide_render = True
            
            # 设置材质颜色（不同的路径不同颜色）
            mat = bpy.data.materials.new(name=f"PathMat_{idx}")
            mat.use_nodes = True
            if mat.node_tree:
                # 使用不同的颜色区分不同的路径
                colors = [
                    (1, 0, 0, 1),    # 红色
                    (0, 1, 0, 1),    # 绿色
                    (0, 0, 1, 1),    # 蓝色
                    (1, 1, 0, 1),    # 黄色
                ]
                color = colors[idx % len(colors)]
                
                # 清除默认节点并创建自定义节点
                nodes = mat.node_tree.nodes
                nodes.clear()
                
                # 创建Emission节点
                emission = nodes.new(type='ShaderNodeEmission')
                emission.inputs['Color'].default_value = color
                emission.inputs['Strength'].default_value = 2.0
                emission.location = (0, 0)
                
                # 创建输出节点
                output = nodes.new(type='ShaderNodeOutputMaterial')
                output.location = (300, 0)
                
                # 连接节点
                links = mat.node_tree.links
                links.new(emission.outputs['Emission'], output.inputs['Surface'])
            
            curve_obj.data.materials.append(mat)
    
    def find_hub_center(self, blades):
        """通过分析三个叶片的根部位置找到轮毂中心"""
        root_positions = []
        
        for blade in blades:
            if blade and blade.type == 'MESH':
                # 获取应用了修改器后的mesh
                depsgraph = bpy.context.evaluated_depsgraph_get()
                eval_obj = blade.evaluated_get(depsgraph)
                mesh = eval_obj.to_mesh()
                
                if mesh and len(mesh.vertices) > 0:
                    # 转换到世界空间
                    verts_world = [blade.matrix_world @ v.co for v in mesh.vertices]
                    # 使用简单的质心作为初始估计
                    centroid = sum(verts_world, Vector()) / len(verts_world)
                    root_positions.append(centroid)
                
                # 清理临时mesh
                eval_obj.to_mesh_clear()
        
        if len(root_positions) >= 2:
            # 轮毂中心应该接近所有叶片根部的中心点
            hub = sum(root_positions, Vector()) / len(root_positions)
            return hub
        
        return None
    
    def get_blade_axis_and_bounds(self, blade, hub_center=None):
        """通过分析mesh获取叶片的主轴方向和边界（使用应用了修改器后的mesh）"""
        # 确保对象是mesh类型
        if blade.type != 'MESH':
            return None, None, None, None
        
        # 获取应用了所有修改器（包括几何节点）后的mesh
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = blade.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        
        if not mesh or len(mesh.vertices) < 2:
            eval_obj.to_mesh_clear()
            return None, None, None, None
        
        # 获取世界空间中的所有顶点
        verts_world = [blade.matrix_world @ v.co for v in mesh.vertices]
        
        # 使用PCA找到叶片的主轴方向（最长的方向）
        verts_array = np.array([(v.x, v.y, v.z) for v in verts_world])
        centroid = np.mean(verts_array, axis=0)
        centered = verts_array - centroid
        
        # 计算协方差矩阵
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 主轴是最大特征值对应的特征向量
        main_axis_idx = np.argmax(eigenvalues)
        main_axis = Vector(eigenvectors[:, main_axis_idx])
        main_axis.normalize()
        
        # 计算沿主轴的投影来找根部和叶尖
        projections = [v.dot(main_axis) for v in verts_world]
        min_proj = min(projections)
        max_proj = max(projections)
        
        # 根部和叶尖位置
        root_pos = Vector(centroid) + main_axis * min_proj
        tip_pos = Vector(centroid) + main_axis * max_proj
        
        # 如果提供了轮毂中心，修正主轴方向和根部位置
        if hub_center:
            # 找到距离轮毂最近的顶点作为真实根部
            min_dist = float('inf')
            closest_vert = root_pos
            for v in verts_world:
                dist = (v - hub_center).length
                if dist < min_dist:
                    min_dist = dist
                    closest_vert = v
            
            # 使用轮毂中心作为根部参考点
            root_pos = hub_center
            
            # 重新计算主轴：从轮毂中心到叶片质心，然后到最远点
            blade_centroid = Vector(centroid)
            temp_axis = (blade_centroid - hub_center).normalized()
            
            # 找到沿这个方向最远的点作为叶尖
            max_dist = 0
            for v in verts_world:
                proj = (v - hub_center).dot(temp_axis)
                if proj > max_dist:
                    max_dist = proj
                    tip_pos = v
            
            # 最终主轴从轮毂指向叶尖
            main_axis = (tip_pos - root_pos).normalized()
        
        # 计算叶片的平均宽度（垂直于主轴的最大距离）
        perp_distances = []
        for v in verts_world:
            # 点到主轴的距离
            v_proj = (v - root_pos).dot(main_axis)
            closest_point = root_pos + main_axis * v_proj
            dist = (v - closest_point).length
            perp_distances.append(dist)
        
        blade_width = max(perp_distances) * 2 if perp_distances else 1.0
        
        # 清理临时mesh
        eval_obj.to_mesh_clear()
        
        return root_pos, tip_pos, main_axis, blade_width
    
    def create_bvh_tree(self, obj):
        """为对象创建BVH树用于ray casting"""
        if obj.type != 'MESH':
            return None
        
        # 获取求值后的mesh（应用了所有修改器）
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        
        # 转换到世界空间
        mesh.transform(obj.matrix_world)
        
        # 创建BVH树
        bvh = BVHTree.FromPolygons(
            [v.co for v in mesh.vertices],
            [p.vertices for p in mesh.polygons]
        )
        
        eval_obj.to_mesh_clear()
        return bvh
    
    def get_perpendicular_vectors(self, main_axis, blade=None, hub_center=None):
        """获取垂直于主轴的两个正交向量，使用叶片自身坐标系保证一致性"""
        # 如果提供了叶片和轮毂中心，使用径向方向作为参考
        if blade is not None and hub_center is not None:
            # 获取叶片原点（对象中心）
            blade_origin = blade.matrix_world.translation
            
            # 从轮毂中心指向叶片原点的径向方向
            radial_dir = (blade_origin - hub_center).normalized()
            
            # 将径向方向投影到垂直于主轴的平面上
            # 移除径向方向在主轴上的分量
            radial_proj = radial_dir - main_axis * radial_dir.dot(main_axis)
            
            # 如果投影后的向量太小，说明径向方向接近平行于主轴
            if radial_proj.length > 0.01:
                radial_proj.normalize()
                # perp1 指向远离轮毂的径向方向（叶片外侧）
                perp1 = radial_proj
                # perp2 垂直于主轴和perp1
                perp2 = main_axis.cross(perp1)
                perp2.normalize()
                return perp1, perp2
        
        # 回退方案：使用世界坐标系
        # 找一个不平行于main_axis的向量
        if abs(main_axis.z) < 0.9:
            temp = Vector((0, 0, 1))
        else:
            temp = Vector((1, 0, 0))
        
        # 创建两个垂直于主轴的正交向量
        perp1 = main_axis.cross(temp)
        perp1.normalize()
        perp2 = main_axis.cross(perp1)
        perp2.normalize()
        
        return perp1, perp2
    
    def generate_blade_inspection(self, context, blade, camera, start_frame, props, hub_center=None):
        """为单个叶片生成四个面的巡检路径（基于mesh分析和ray casting）"""
        
        # 分析叶片的主轴和边界（使用轮毂中心）
        root_pos, tip_pos, main_axis, blade_width = self.get_blade_axis_and_bounds(blade, hub_center)
        
        if root_pos is None:
            self.report({'WARNING'}, f"Could not analyze blade {blade.name}")
            return start_frame, []
        
        # 创建BVH树用于ray casting
        bvh = self.create_bvh_tree(blade)
        if bvh is None:
            self.report({'WARNING'}, f"Could not create BVH tree for {blade.name}")
            return start_frame, []
        
        blade_length = (tip_pos - root_pos).length
        current_frame = start_frame
        distance = props.distance_from_blade
        frames_per_face = props.frames_per_face
        extension = props.extension_ratio
        
        # 延伸扫描范围以确保完全覆盖叶尖
        extended_length = blade_length * (1 + 2 * extension)
        extended_root = root_pos - main_axis * (blade_length * extension)
        extended_tip = tip_pos + main_axis * (blade_length * extension)
        
        # 存储所有路径点用于可视化
        blade_path_groups = []
        
        # 获取垂直于叶片主轴的两个正交方向（使用叶片自身坐标系）
        perp1, perp2 = self.get_perpendicular_vectors(main_axis, blade, hub_center)
        
        # 调试信息：输出扫描方向
        blade_origin = blade.matrix_world.translation
        self.report({'INFO'}, f"Blade {blade.name} origin: ({blade_origin.x:.2f}, {blade_origin.y:.2f}, {blade_origin.z:.2f})")
        self.report({'INFO'}, f"Blade {blade.name} perp1 direction: ({perp1.x:.2f}, {perp1.y:.2f}, {perp1.z:.2f})")
        
        # 定义四个面的方向（相对于叶片主轴的垂直方向）
        # perp1 指向外侧（远离轮毂），-perp1 指向内侧（靠近轮毂/正面）
        face_directions = [
            -perp1,     # 侧面1 - 内侧/正面（靠近轮毂）
            perp2,      # 侧面2 - 前缘
            perp1,      # 侧面3 - 外侧/背面（远离轮毂）
            -perp2,     # 侧面4 - 后缘
        ]
        
        # 统计ray casting成功率
        ray_success_count = 0
        ray_total_count = 0
        
        # 保存每个面结束时的位置和旋转
        last_pos = None
        last_quat = None
        
        # 为每个面生成扫描路径
        for face_idx, direction in enumerate(face_directions):
            face_path_points = []
            prev_target = None  # 用于平滑target过渡
            
            # 沿着主轴从根部到叶尖扫描（使用延伸后的范围）
            for i in range(frames_per_face + 1):
                t = i / frames_per_face
                
                # 沿主轴的当前位置（使用延伸后的范围）
                axis_point = extended_root.lerp(extended_tip, t)
                
                ray_total_count += 1
                
                # 改进策略：优先在当前扫描方向查找表面，确保稳定性和连续性
                surface_point = None
                
                # 方法1：优先在当前扫描方向投射（最可靠）
                search_radius = max(blade_width * 1.5, 15.0) if blade_width > 1.0 else 15.0
                ray_start = axis_point - direction * search_radius
                location, normal, index, dist = bvh.ray_cast(ray_start, direction, search_radius * 3)
                
                if location:
                    # 检查点是否在当前高度附近
                    height_diff = abs((location - axis_point).dot(main_axis))
                    if height_diff < blade_length * 0.1:
                        surface_point = location
                        ray_success_count += 1
                
                # 方法2：如果直接投射失败，使用find_nearest
                if not surface_point:
                    nearest_point, nearest_normal, nearest_index, nearest_dist = bvh.find_nearest(axis_point)
                    if nearest_point and nearest_dist < blade_width * 0.8:
                        surface_point = nearest_point
                        ray_success_count += 1
                
                # 方法3：如果还是没找到，使用8方向搜索作为备选
                if not surface_point:
                    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                        angle_rad = math.radians(angle)
                        search_dir = (perp1 * math.cos(angle_rad) + perp2 * math.sin(angle_rad)).normalized()
                        
                        ray_start = axis_point + search_dir * search_radius
                        location, normal, index, dist = bvh.ray_cast(ray_start, -search_dir, search_radius * 2)
                        
                        if location:
                            height_diff = abs((location - axis_point).dot(main_axis))
                            if height_diff < blade_length * 0.15:
                                surface_point = location
                                ray_success_count += 1
                                break
                
                # 统一的相机定位策略 + target平滑过渡
                # 相机始终位于：主轴点 + 扫描方向 * 距离
                # 相机朝向：平滑的target点
                
                # 计算当前帧的理想target
                if surface_point:
                    # 找到了表面点，计算更精确的相机位置
                    surface_to_axis = (surface_point - axis_point)
                    surface_on_axis = axis_point + main_axis * surface_to_axis.dot(main_axis)
                    
                    # 相机位置：从投影点沿扫描方向偏移
                    camera_pos = surface_on_axis + direction * distance
                    current_target = surface_point
                else:
                    # 没找到表面点，估计一个合理的target位置
                    # 在当前扫描方向上估计表面位置（假设叶片宽度）
                    estimated_surface = axis_point + direction * (blade_width * 0.3)
                    camera_pos = axis_point + direction * distance
                    current_target = estimated_surface
                    
                    # 调试信息
                    if i % 10 == 0:
                        self.report({'WARNING'}, f"Face {face_idx+1}, frame {i}: No surface found, using estimated target")
                
                # 平滑target过渡：如果有上一帧的target，进行插值
                if prev_target is not None:
                    # 使用较大的插值权重确保平滑（0.7表示70%使用新target，30%保留旧target）
                    target = prev_target.lerp(current_target, 0.7)
                else:
                    target = current_target
                
                # 保存当前target用于下一帧
                prev_target = target.copy()
                
                # 设置相机位置
                camera.location = camera_pos
                camera.keyframe_insert(data_path="location", frame=current_frame + i)
                
                # 保存路径点
                face_path_points.append(camera_pos.copy())
                
                # 计算相机朝向（看向叶片表面）- 使用四元数避免欧拉角抖动
                direction_to_target = target - camera_pos
                if direction_to_target.length > 0.001:
                    # 计算旋转，相机的-Z轴指向目标
                    rot_quat = direction_to_target.to_track_quat('-Z', 'Y')
                    camera.rotation_quaternion = rot_quat
                    camera.keyframe_insert(data_path="rotation_quaternion", frame=current_frame + i)
                    
                    # 保存最后一帧的位置和旋转，用于生成过渡
                    if i == frames_per_face:
                        last_pos = camera_pos.copy()
                        last_quat = rot_quat.copy()
            
            # 保存这个面的路径点
            blade_path_groups.append(face_path_points)
            current_frame += frames_per_face + 1  # +1是因为包含了最后一帧
            
            # 在面切换之间添加过渡帧（除了最后一个面）
            if face_idx < len(face_directions) - 1 and props.transition_frames > 0:
                # 获取下一个面的起始位置
                next_direction = face_directions[face_idx + 1]
                next_start_pos = extended_root + next_direction * distance
                next_target = extended_root
                
                # 生成平滑过渡
                for trans_i in range(props.transition_frames):
                    t = (trans_i + 1) / (props.transition_frames + 1)
                    
                    # 位置插值
                    trans_pos = last_pos.lerp(next_start_pos, t)
                    camera.location = trans_pos
                    camera.keyframe_insert(data_path="location", frame=current_frame + trans_i)
                    
                    # 旋转插值（四元数球面插值）
                    next_dir = next_target - next_start_pos
                    if next_dir.length > 0.001:
                        next_quat = next_dir.to_track_quat('-Z', 'Y')
                        trans_quat = last_quat.slerp(next_quat, t)
                        camera.rotation_quaternion = trans_quat
                        camera.keyframe_insert(data_path="rotation_quaternion", frame=current_frame + trans_i)
                
                current_frame += props.transition_frames
        
        # 报告ray casting成功率
        success_rate = (ray_success_count / ray_total_count * 100) if ray_total_count > 0 else 0
        self.report({'INFO'}, f"Blade {blade.name}: Surface detection success rate {success_rate:.1f}%")
        
        return current_frame, blade_path_groups, last_pos, last_quat


class BLADE_OT_ClearInspectionPath(bpy.types.Operator):
    """清除巡检路径和相机动画"""
    bl_idname = "blade.clear_inspection_path"
    bl_label = "Clear Inspection Path"
    bl_description = "Clear visualization paths and camera keyframes"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.blade_inspection_props
        
        # 清除InspectionPaths集合
        collection_name = "InspectionPaths"
        removed_count = 0
        
        if collection_name in bpy.data.collections:
            collection = bpy.data.collections[collection_name]
            # 计算删除的对象数量
            removed_count = len(collection.objects)
            # 删除集合中的所有对象
            for obj in list(collection.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            # 从场景中取消链接集合
            bpy.context.scene.collection.children.unlink(collection)
            # 删除集合
            bpy.data.collections.remove(collection)
        
        # 清除任何遗留的InspectionPath对象
        for obj in list(bpy.data.objects):
            if obj.name.startswith("InspectionPath_"):
                bpy.data.objects.remove(obj, do_unlink=True)
                removed_count += 1
        
        # 清理未使用的curve数据
        for curve in list(bpy.data.curves):
            if curve.name.startswith("InspectionPath_") and curve.users == 0:
                bpy.data.curves.remove(curve)
        
        # 清理未使用的材质
        for mat in list(bpy.data.materials):
            if mat.name.startswith("PathMat_") and mat.users == 0:
                bpy.data.materials.remove(mat)
        
        # 清除相机关键帧（包括四元数）
        if props.camera:
            if props.camera.animation_data:
                props.camera.animation_data_clear()
            # 确保相机使用欧拉角模式（以防之前使用了四元数）
            props.camera.rotation_mode = 'QUATERNION'
            self.report({'INFO'}, f"Cleared {removed_count} path objects and camera keyframes")
        else:
            self.report({'INFO'}, f"Cleared {removed_count} path objects")
        
        return {'FINISHED'}


class BLADE_PT_InspectionPanel(bpy.types.Panel):
    """叶片巡检面板"""
    bl_label = "Blade Inspection"
    bl_idname = "BLADE_PT_inspection_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Blade Inspection'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.blade_inspection_props
        
        # 叶片选择区域
        box = layout.box()
        box.label(text="Blade Selection:", icon='MESH_DATA')
        box.prop(props, "blade1")
        box.prop(props, "blade2")
        box.prop(props, "blade3")
        
        # 相机选择区域
        box = layout.box()
        box.label(text="Camera Settings:", icon='CAMERA_DATA')
        box.prop(props, "camera")
        box.prop(props, "distance_from_blade")
        
        # 动画参数
        box = layout.box()
        box.label(text="Animation Settings:", icon='TIME')
        box.prop(props, "frames_per_face")
        box.prop(props, "transition_frames")
        box.prop(props, "extension_ratio")
        box.prop(props, "create_path_visualization")
        
        # 计算总帧数：每个叶片4个面，每个面之间有过渡，叶片之间也有过渡
        scan_frames = props.frames_per_face * 4 * 3
        face_transitions = props.transition_frames * 3 * 3  # 每个叶片3个面切换
        blade_transitions = props.transition_frames * 2  # 2个叶片切换
        total_frames = scan_frames + face_transitions + blade_transitions
        box.label(text=f"Total Frames: ~{total_frames}")
        
        # 生成按钮
        layout.separator()
        row = layout.row(align=True)
        row.operator("blade.generate_inspection_path", icon='PLAY')
        row.operator("blade.clear_inspection_path", icon='X', text="Clear")


# 注册类列表
classes = (
    BladeInspectionProperties,
    BLADE_OT_GenerateInspectionPath,
    BLADE_OT_ClearInspectionPath,
    BLADE_PT_InspectionPanel,
)


def register():
    """注册插件 - 避免panel消失bug"""
    # 先注销（如果已存在）
    try:
        unregister()
    except:
        pass
    
    # 注册所有类
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # 添加属性组到场景
    bpy.types.Scene.blade_inspection_props = bpy.props.PointerProperty(
        type=BladeInspectionProperties
    )
    
    print("Blade Inspection Panel registered successfully")


def unregister():
    """注销插件"""
    # 删除场景属性
    if hasattr(bpy.types.Scene, "blade_inspection_props"):
        del bpy.types.Scene.blade_inspection_props
    
    # 注销所有类（逆序）
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except:
            pass
    
    print("Blade Inspection Panel unregistered")


if __name__ == "__main__":
    register()
