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

# Configuration
COLLECTION_NAME = "InspectionPaths"
PATH_OBJECT_PREFIX = "InspectionPath_"
PATH_MATERIAL_PREFIX = "PathMat_"
PATH_COLORS = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1)]
DEFAULT_SEARCH_RADIUS = 15.0
SURFACE_SMOOTH_ALPHA = 0.3


# ==================== Helper Functions ====================

def _cleanup_inspection_resources(collection_name, object_prefix, material_prefix):
    """Cleanup inspection resources (collections, objects, materials)"""
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.context.scene.collection.children.unlink(collection)
        bpy.data.collections.remove(collection)
    
    for obj in list(bpy.data.objects):
        if obj.name.startswith(object_prefix):
            bpy.data.objects.remove(obj, do_unlink=True)
    
    for curve in list(bpy.data.curves):
        if curve.name.startswith(object_prefix) and curve.users == 0:
            bpy.data.curves.remove(curve)
    
    for mat in list(bpy.data.materials):
        if mat.name.startswith(material_prefix) and mat.users == 0:
            bpy.data.materials.remove(mat)


def _get_or_create_collection(name):
    """Get or create collection"""
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def _create_curve_object(name, points):
    """Create curve object"""
    curve_data = bpy.data.curves.new(name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 2
    curve_data.bevel_depth = 0.02
    
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(len(points) - 1)
    for i, point in enumerate(points):
        polyline.points[i].co = (point.x, point.y, point.z, 1)
    
    return bpy.data.objects.new(name, curve_data)


def _create_emission_material(name, color):
    """Create emission material"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    if mat.node_tree:
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        emission = nodes.new(type='ShaderNodeEmission')
        emission.inputs['Color'].default_value = color
        emission.inputs['Strength'].default_value = 2.0
        
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)
        
        mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    return mat


def _get_world_vertices(obj):
    """Get world space vertices"""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    
    if not mesh:
        eval_obj.to_mesh_clear()
        return None
    
    verts = [obj.matrix_world @ v.co for v in mesh.vertices]
    eval_obj.to_mesh_clear()
    return verts


def _create_bvh_tree_from_object(obj):
    """Create BVH tree from object"""
    if obj.type != 'MESH':
        return None
    
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    mesh.transform(obj.matrix_world)
    
    bvh = BVHTree.FromPolygons(
        [v.co for v in mesh.vertices],
        [p.vertices for p in mesh.polygons]
    )
    eval_obj.to_mesh_clear()
    return bvh


def _create_transition_animation(camera, start_frame, frame_count, start_pos, end_pos, start_quat, target):
    """Create transition animation"""
    for i in range(frame_count):
        t = (i + 1) / frame_count
        trans_pos = start_pos.lerp(end_pos, t)
        camera.location = trans_pos
        camera.keyframe_insert(data_path="location", frame=start_frame + i)
        
        next_dir = target - end_pos
        if next_dir.length > 0.001:
            next_quat = next_dir.to_track_quat('-Z', 'Y')
            trans_quat = start_quat.slerp(next_quat, t)
            camera.rotation_quaternion = trans_quat
            camera.keyframe_insert(data_path="rotation_quaternion", frame=start_frame + i)
    
    return start_frame + frame_count


# ==================== Class Definitions ====================

def update_rotation_angle(self, context):
    """Update node group when rotation_angle changes"""
    try:
        if "Static Windblade Rotation" in bpy.data.node_groups:
            node_group = bpy.data.node_groups["Static Windblade Rotation"]
            if "Rot" in node_group.nodes:
                node_group.nodes["Rot"].outputs[0].default_value = self.rotation_angle
        else:
            print("Node group 'Static Windblade Rotation' not found")
    except Exception as e:
        print(f"Error updating rotation angle: {e}")


class BladeInspectionProperties(bpy.types.PropertyGroup):
    """Properties for blade and camera selection"""
    blade_center: bpy.props.PointerProperty(
        name="Blade Center",
        type=bpy.types.Object,
        description="Select blade center object (hub)"
    )
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
    rotation_angle: bpy.props.FloatProperty(
        name="Rotation Angle",
        description="Rotation angle of the blade hub in degrees",
        default=0.0,
        min=-3.14,
        max=3.14,
        update=update_rotation_angle
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
    """Generate blade inspection path"""
    bl_idname = "blade.generate_inspection_path"
    bl_label = "Generate Inspection Path"
    bl_description = "Generate camera animation path for blade inspection"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.blade_inspection_props
        
        if not props.camera:
            self.report({'ERROR'}, "Please select a camera")
            return {'CANCELLED'}
        
        blades = [props.blade1, props.blade2, props.blade3, props.blade_center]
        if not all(blades):
            self.report({'ERROR'}, "Please select all three blades and the blade center")
            return {'CANCELLED'}
        
        blades = [props.blade1, props.blade2, props.blade3]
        camera = props.camera
        
        if camera.animation_data:
            camera.animation_data_clear()
        
        if props.create_path_visualization:
            self.cleanup_old_paths()
        
        current_frame = 0
        bpy.context.scene.frame_start = 0
        all_path_points = []
        
        hub_center = self.find_hub_center(blades)
        if hub_center:
            self.report({'INFO'}, f"Hub center found at: ({hub_center.x:.2f}, {hub_center.y:.2f}, {hub_center.z:.2f})")
        
        last_blade_end_pos = None
        last_blade_end_quat = None
        
        for blade_idx, blade in enumerate(blades):
            if blade is None:
                continue
            
            if blade_idx > 0 and last_blade_end_pos and props.transition_frames > 0:
                next_root, next_tip, next_axis, _ = self.get_blade_axis_and_bounds(blade, hub_center)
                if next_root:
                    perp1, perp2 = self.get_perpendicular_vectors(next_axis, blade, hub_center)
                    next_start_pos = next_root + perp1 * props.distance_from_blade
                    current_frame = _create_transition_animation(
                        camera, current_frame, props.transition_frames,
                        last_blade_end_pos, next_start_pos, last_blade_end_quat, next_root
                    )
            
            self.report({'INFO'}, f"Processing blade {blade_idx + 1}: {blade.name}")
            current_frame, blade_paths, end_pos, end_quat = self.generate_blade_inspection(
                context, blade, camera, current_frame, props, hub_center
            )
            if blade_paths:
                all_path_points.extend(blade_paths)
            
            last_blade_end_pos = end_pos
            last_blade_end_quat = end_quat
        
        bpy.context.scene.frame_end = current_frame - 1
        bpy.context.scene.frame_set(0)
        
        if props.create_path_visualization and all_path_points:
            self.create_path_curves(all_path_points)
        
        self.report({'INFO'}, f"Inspection path generated successfully! Total frames: {current_frame - 1}")
        return {'FINISHED'}
    
    def cleanup_old_paths(self):
        """Cleanup old path visualization"""
        _cleanup_inspection_resources(COLLECTION_NAME, PATH_OBJECT_PREFIX, PATH_MATERIAL_PREFIX)
    
    def create_path_curves(self, path_groups):
        """Create curve objects for path visualization"""
        collection = _get_or_create_collection(COLLECTION_NAME)
        
        for idx, points in enumerate(path_groups):
            if len(points) < 2:
                continue
            
            curve_obj = _create_curve_object(f"{PATH_OBJECT_PREFIX}{idx}", points)
            collection.objects.link(curve_obj)
            curve_obj.hide_render = True
            
            mat = _create_emission_material(f"{PATH_MATERIAL_PREFIX}{idx}", PATH_COLORS[idx % len(PATH_COLORS)])
            curve_obj.data.materials.append(mat)
    
    def find_hub_center(self, blades):
        """Find hub center by analyzing blade root positions"""
        root_positions = []
        
        for blade in blades:
            if blade and blade.type == 'MESH':
                depsgraph = bpy.context.evaluated_depsgraph_get()
                eval_obj = blade.evaluated_get(depsgraph)
                mesh = eval_obj.to_mesh()
                
                if mesh and len(mesh.vertices) > 0:
                    verts_world = [blade.matrix_world @ v.co for v in mesh.vertices]
                    centroid = sum(verts_world, Vector()) / len(verts_world)
                    root_positions.append(centroid)
                
                eval_obj.to_mesh_clear()
        
        if len(root_positions) >= 2:
            hub = sum(root_positions, Vector()) / len(root_positions)
            return hub
        
        return None
    
    def get_blade_axis_and_bounds(self, blade, hub_center=None):
        """Get blade main axis and bounds via PCA analysis"""
        if blade.type != 'MESH':
            return None, None, None, None
        
        verts_world = _get_world_vertices(blade)
        if not verts_world or len(verts_world) < 2:
            return None, None, None, None
        
        verts_array = np.array([(v.x, v.y, v.z) for v in verts_world])
        centroid = np.mean(verts_array, axis=0)
        centered = verts_array - centroid
        
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        main_axis_idx = np.argmax(eigenvalues)
        main_axis = Vector(eigenvectors[:, main_axis_idx])
        main_axis.normalize()
        
        projections = [v.dot(main_axis) for v in verts_world]
        min_proj = min(projections)
        max_proj = max(projections)
        
        root_pos = Vector(centroid) + main_axis * min_proj
        tip_pos = Vector(centroid) + main_axis * max_proj
        
        if hub_center:
            min_dist = float('inf')
            closest_vert = root_pos
            for v in verts_world:
                dist = (v - hub_center).length
                if dist < min_dist:
                    min_dist = dist
                    closest_vert = v
            
            root_pos = hub_center
            blade_centroid = Vector(centroid)
            temp_axis = (blade_centroid - hub_center).normalized()
            
            max_dist = 0
            for v in verts_world:
                proj = (v - hub_center).dot(temp_axis)
                if proj > max_dist:
                    max_dist = proj
                    tip_pos = v
            
            main_axis = (tip_pos - root_pos).normalized()
        
        perp_distances = []
        for v in verts_world:
            v_proj = (v - root_pos).dot(main_axis)
            closest_point = root_pos + main_axis * v_proj
            dist = (v - closest_point).length
            perp_distances.append(dist)
        
        blade_width = max(perp_distances) * 2 if perp_distances else 1.0
        
        return root_pos, tip_pos, main_axis, blade_width
    
    def create_bvh_tree(self, obj):
        """Create BVH tree for ray casting"""
        return _create_bvh_tree_from_object(obj)
    
    def get_perpendicular_vectors(self, main_axis, blade=None, hub_center=None):
        """Get two orthogonal vectors perpendicular to main axis"""
        if blade is not None and hub_center is not None:
            blade_origin = blade.matrix_world.translation
            radial_dir = (blade_origin - hub_center).normalized()
            radial_proj = radial_dir - main_axis * radial_dir.dot(main_axis)
            
            if radial_proj.length > 0.01:
                radial_proj.normalize()
                perp1 = radial_proj
                perp2 = main_axis.cross(perp1)
                perp2.normalize()
                return perp1, perp2
        
        if abs(main_axis.z) < 0.9:
            temp = Vector((0, 0, 1))
        else:
            temp = Vector((1, 0, 0))
        
        perp1 = main_axis.cross(temp)
        perp1.normalize()
        perp2 = main_axis.cross(perp1)
        perp2.normalize()
        
        return perp1, perp2
    
    def _detect_surface(self, bvh, origin, direction, main_axis, blade_width, blade_length, perp1, perp2):
        """Robust surface detection: ray cast -> nearest point -> radial search"""
        search_radius = max(blade_width * 1.5, DEFAULT_SEARCH_RADIUS) if blade_width > 1.0 else DEFAULT_SEARCH_RADIUS
        
        location, *_ = bvh.ray_cast(origin - direction * search_radius, direction, search_radius * 3)
        if location and abs((location - origin).dot(main_axis)) < blade_length * 0.1:
            return location, True

        nearest_pos, _, _, nearest_dist = bvh.find_nearest(origin)
        if nearest_pos and nearest_dist < blade_width * 0.8:
            return nearest_pos, True

        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            search_dir = (perp1 * math.cos(rad) + perp2 * math.sin(rad)).normalized()
            loc, *_ = bvh.ray_cast(origin + search_dir * search_radius, -search_dir, search_radius * 2)
            if loc and abs((loc - origin).dot(main_axis)) < blade_length * 0.15:
                return loc, True
        
        return None, False

    def generate_blade_inspection(self, context, blade, camera, start_frame, props, hub_center=None):
        """Generate S-shaped (zigzag) inspection path for four blade faces"""
        root_pos, tip_pos, main_axis, blade_width = self.get_blade_axis_and_bounds(blade, hub_center)
        if root_pos is None:
            self.report({'WARNING'}, f"Could not analyze blade {blade.name}")
            return start_frame, [], None, None
        
        bvh = self.create_bvh_tree(blade)
        if bvh is None: 
            return start_frame, [], None, None
        
        blade_length = (tip_pos - root_pos).length
        extended_root = root_pos - main_axis * (blade_length * props.extension_ratio)
        extended_tip = tip_pos + main_axis * (blade_length * props.extension_ratio)
        perp1, perp2 = self.get_perpendicular_vectors(main_axis, blade, hub_center)
        
        face_directions = [-perp1, perp2, perp1, -perp2]
        current_frame = start_frame
        blade_path_groups = []
        prev_target = None 
        last_cam_pos = None
        last_cam_quat = None
        ray_hits = 0
        ray_total = 0
        
        for face_idx, direction in enumerate(face_directions):
            face_path_points = []
            frames_count = props.frames_per_face
            is_reverse_pass = (face_idx % 2 == 1)
            
            for i in range(frames_count + 1):
                progress = i / frames_count
                t = 1.0 - progress if is_reverse_pass else progress
                axis_point = extended_root.lerp(extended_tip, t)
                
                ray_total += 1
                surface_point, found = self._detect_surface(
                    bvh, axis_point, direction, main_axis, 
                    blade_width, blade_length, perp1, perp2
                )
                
                if found:
                    ray_hits += 1
                    current_target = surface_point
                    surface_to_axis = (surface_point - axis_point)
                    surface_on_axis_proj = axis_point + main_axis * surface_to_axis.dot(main_axis)
                    camera_pos = surface_on_axis_proj + direction * props.distance_from_blade
                else:
                    estimated_surface = axis_point + direction * (blade_width * 0.3)
                    current_target = estimated_surface
                    camera_pos = axis_point + direction * props.distance_from_blade
                    if i % 30 == 0: 
                        self.report({'DEBUG'}, f"Surface lost at frame {current_frame + i}")
                
                target = prev_target.lerp(current_target, SURFACE_SMOOTH_ALPHA) if prev_target else current_target
                prev_target = target.copy()
                
                camera.location = camera_pos
                camera.keyframe_insert(data_path="location", frame=current_frame + i)
                
                direction_to_target = target - camera_pos
                if direction_to_target.length > 0.001:
                    rot_quat = direction_to_target.to_track_quat('-Z', 'Y')
                    camera.rotation_quaternion = rot_quat
                    camera.keyframe_insert(data_path="rotation_quaternion", frame=current_frame + i)
                    
                    if i == frames_count:
                        last_cam_pos, last_cam_quat = camera_pos, rot_quat
                
                face_path_points.append(camera_pos.copy())
            
            blade_path_groups.append(face_path_points)
            current_frame += frames_count + 1
            
            if face_idx < len(face_directions) - 1 and props.transition_frames > 0:
                next_direction = face_directions[face_idx + 1]
                next_is_reverse = ((face_idx + 1) % 2 == 1)
                next_ref_point = extended_tip if next_is_reverse else extended_root
                next_start_pos = next_ref_point + next_direction * props.distance_from_blade
                current_frame = _create_transition_animation(
                    camera, current_frame, props.transition_frames,
                    last_cam_pos, next_start_pos, last_cam_quat, next_ref_point
                )
        
        success_rate = (ray_hits / ray_total * 100) if ray_total > 0 else 0
        self.report({'INFO'}, f"Blade {blade.name} Path Generated: {success_rate:.1f}% coverage")
        return current_frame, blade_path_groups, last_cam_pos, last_cam_quat


class BLADE_OT_ClearInspectionPath(bpy.types.Operator):
    """Clear inspection path and camera animation"""
    bl_idname = "blade.clear_inspection_path"
    bl_label = "Clear Inspection Path"
    bl_description = "Clear visualization paths and camera keyframes"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.blade_inspection_props
        
        removed_count = len(bpy.data.collections.get(COLLECTION_NAME, bpy.data.collections.new("temp")).objects)
        _cleanup_inspection_resources(COLLECTION_NAME, PATH_OBJECT_PREFIX, PATH_MATERIAL_PREFIX)
        
        if props.camera:
            if props.camera.animation_data:
                props.camera.animation_data_clear()
            props.camera.rotation_mode = 'QUATERNION'
            self.report({'INFO'}, f"已清除 {removed_count} 个路径对象和相机关键帧")
        else:
            self.report({'INFO'}, f"已清除 {removed_count} 个路径对象")
        
        return {'FINISHED'}


class BLADE_PT_InspectionPanel(bpy.types.Panel):
    """Blade inspection panel"""
    bl_label = "Blade Inspection"
    bl_idname = "BLADE_PT_inspection_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Blade Inspection'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.blade_inspection_props
        
        box = layout.box()
        box.label(text="Blade Selection:", icon='MESH_DATA')
        box.prop(props, "blade_center")
        box.prop(props, "blade1")
        box.prop(props, "blade2")
        box.prop(props, "blade3")
        box.prop(props, "rotation_angle", text="Rotation Angle", icon='DRIVER_ROTATIONAL_DIFFERENCE')

        box = layout.box()
        box.label(text="Camera Settings:", icon='CAMERA_DATA')
        box.prop(props, "camera")
        box.prop(props, "distance_from_blade")
        
        box = layout.box()
        box.label(text="Animation Settings:", icon='TIME')
        box.prop(props, "frames_per_face")
        box.prop(props, "transition_frames")
        box.prop(props, "extension_ratio")
        box.prop(props, "create_path_visualization")
        
        scan_frames = props.frames_per_face * 4 * 3
        face_transitions = props.transition_frames * 3 * 3
        blade_transitions = props.transition_frames * 2
        total_frames = scan_frames + face_transitions + blade_transitions
        box.label(text=f"Total Frames: ~{total_frames}")
        
        layout.separator()
        row = layout.row(align=True)
        row.operator("blade.generate_inspection_path", icon='PLAY')
        row.operator("blade.clear_inspection_path", icon='X', text="Clear")


classes = (
    BladeInspectionProperties,
    BLADE_OT_GenerateInspectionPath,
    BLADE_OT_ClearInspectionPath,
    BLADE_PT_InspectionPanel,
)


def register():
    """Register addon"""
    try:
        unregister()
    except:
        pass
    
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.blade_inspection_props = bpy.props.PointerProperty(
        type=BladeInspectionProperties
    )
    
    print("Blade Inspection Panel registered successfully")


def unregister():
    """Unregister addon"""
    if hasattr(bpy.types.Scene, "blade_inspection_props"):
        del bpy.types.Scene.blade_inspection_props
    
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except:
            pass
    
    print("Blade Inspection Panel unregistered")


if __name__ == "__main__":
    register()
