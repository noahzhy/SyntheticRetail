"""
Blender rendering script for synthetic retail data generation.
This script should be run inside Blender with bpy available.

Usage:
    blender --background --python blender_renderer.py -- --recipe scene_recipe.json --output ./output
"""
import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import math

# Check if running in Blender
try:
    import bpy
except ImportError:
    print("ERROR: This script must be run inside Blender")
    print("Usage: blender --background --python blender_renderer.py -- --recipe scene.json --output ./out")
    sys.exit(1)


class BlenderRenderer:
    """Handles Blender scene setup and rendering"""
    
    def __init__(self, recipe_path: str, output_dir: str):
        self.recipe_path = recipe_path
        self.output_dir = Path(output_dir)
        self.recipe = self._load_recipe()
        self.scene_id = self.recipe['scene_id']
        
    def _load_recipe(self) -> Dict[str, Any]:
        """Load scene recipe from JSON"""
        with open(self.recipe_path, 'r') as f:
            return json.load(f)
    
    def setup_scene(self):
        """Setup Blender scene based on recipe"""
        print(f"Setting up scene: {self.recipe.get('scene_id', 'unknown')}")
        
        # Clear existing scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # Setup render settings
        self._setup_render_settings()
        
        # Create environment
        self._setup_world()
        
        # Create shelves and slots
        self._create_shelf_structure()
        
        # Place products
        self._place_products()
        
        # Setup camera
        self._setup_camera()
        
        # Setup lighting
        self._setup_lighting()
        
        print("Scene setup complete")
    
    def _setup_render_settings(self):
        """Configure rendering settings"""
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 128
        scene.cycles.use_denoising = True
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = False
        
        # Enable GPU if available
        try:
            scene.cycles.device = 'GPU'
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = 'CUDA'  # Or 'OPTIX', 'HIP'
            prefs.get_devices()
            for device in prefs.devices:
                device.use = True
            print("GPU rendering enabled")
        except:
            print("Using CPU rendering")
        
        # Setup compositor for multi-channel output
        scene.use_nodes = True
        scene.render.use_compositing = True
    
    def _setup_world(self):
        """Setup world environment"""
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
        world.use_nodes = True
        
        # Get node tree
        nodes = world.node_tree.nodes
        nodes.clear()
        
        # Add background
        node_background = nodes.new('ShaderNodeBackground')
        node_background.inputs[0].default_value = (0.95, 0.95, 0.95, 1.0)  # Light gray
        node_background.inputs[1].default_value = 1.0  # Strength
        
        # Add output
        node_output = nodes.new('ShaderNodeOutputWorld')
        
        # Link nodes
        world.node_tree.links.new(node_background.outputs[0], node_output.inputs[0])
    
    def _create_shelf_structure(self):
        """Create realistic shelf geometry"""
        shelf_material = self._create_wood_material()
        
        for shelf_data in self.recipe['shelves']:
            shelf_id = shelf_data['shelf_id']
            height = shelf_data['height']
            num_slots = shelf_data['num_slots']
            
            # Create main shelf board
            bpy.ops.mesh.primitive_cube_add(
                size=1,
                location=(0, 0, height)
            )
            shelf = bpy.context.active_object
            shelf.name = f"Shelf_{shelf_id}"
            shelf.scale = (0.6, 0.15, 0.01)  # 1.2m x 0.3m x 2cm
            
            # Apply material
            if shelf.data.materials:
                shelf.data.materials[0] = shelf_material
            else:
                shelf.data.materials.append(shelf_material)
            
            # Add support bars
            bar_material = self._create_metal_material()
            for x_pos in [-0.6, 0.6]:
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=0.015,
                    depth=height,
                    location=(x_pos, 0, height/2)
                )
                bar = bpy.context.active_object
                bar.name = f"Support_{shelf_id}_{x_pos}"
                if bar.data.materials:
                    bar.data.materials[0] = bar_material
                else:
                    bar.data.materials.append(bar_material)
    
    def _create_wood_material(self):
        """Create wooden material for shelves"""
        mat = bpy.data.materials.new(name="Wood")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        # Add Principled BSDF
        node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        node_bsdf.inputs['Base Color'].default_value = (0.55, 0.35, 0.15, 1.0)  # Brown
        node_bsdf.inputs['Roughness'].default_value = 0.6
        node_bsdf.inputs['Specular IOR Level'].default_value = 0.3
        
        # Add output
        node_output = nodes.new('ShaderNodeOutputMaterial')
        
        # Link
        mat.node_tree.links.new(node_bsdf.outputs[0], node_output.inputs[0])
        
        return mat
    
    def _create_metal_material(self):
        """Create metal material for support bars"""
        mat = bpy.data.materials.new(name="Metal")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        # Add Principled BSDF
        node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        node_bsdf.inputs['Base Color'].default_value = (0.3, 0.3, 0.3, 1.0)  # Dark gray
        node_bsdf.inputs['Metallic'].default_value = 0.9
        node_bsdf.inputs['Roughness'].default_value = 0.2
        
        # Add output
        node_output = nodes.new('ShaderNodeOutputMaterial')
        
        # Link
        mat.node_tree.links.new(node_bsdf.outputs[0], node_output.inputs[0])
        
        return mat
    
    def _create_product_material(self, color_rgb: Tuple[float, float, float]):
        """Create material for product with given color"""
        mat = bpy.data.materials.new(name=f"Product_{color_rgb}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        # Add Principled BSDF
        node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        node_bsdf.inputs['Base Color'].default_value = (*color_rgb, 1.0)
        node_bsdf.inputs['Roughness'].default_value = 0.4
        node_bsdf.inputs['Specular IOR Level'].default_value = 0.5
        
        # Add output
        node_output = nodes.new('ShaderNodeOutputMaterial')
        
        # Link
        mat.node_tree.links.new(node_bsdf.outputs[0], node_output.inputs[0])
        
        return mat
    
    def _place_products(self):
        """Place products according to recipe"""
        # Category colors
        category_colors = {
            'beverages': (0.3, 0.5, 0.8),    # Blue
            'snacks': (0.2, 0.7, 0.3),       # Green
            'dairy': (0.9, 0.9, 0.4),        # Yellow
            'unknown': (0.5, 0.5, 0.5)       # Gray
        }
        
        for shelf_data in self.recipe['shelves']:
            shelf_id = shelf_data['shelf_id']
            shelf_height = shelf_data['height']
            num_slots = shelf_data['num_slots']
            slot_width = 1.2 / num_slots
            
            for slot_data in shelf_data['slots']:
                slot_id = slot_data['slot_id']
                
                if slot_data['is_empty']:
                    continue
                
                # Calculate slot center position
                x_base = -0.6 + (slot_id + 0.5) * slot_width
                y_base = 0
                z_base = shelf_height + 0.01  # Just above shelf
                
                num_products = len(slot_data['products'])
                
                for prod_idx, product in enumerate(slot_data['products']):
                    # Calculate position with spacing
                    if num_products == 1:
                        offset_x = 0
                    else:
                        spacing = slot_width * 0.7 / max(num_products, 1)
                        offset_x = (prod_idx - (num_products - 1) / 2) * spacing
                    
                    product_x = x_base + offset_x + product['position'][0] * 0.02
                    product_y = y_base + product['position'][1] * 0.02
                    product_z = z_base
                    
                    # Create product box
                    bpy.ops.mesh.primitive_cube_add(
                        size=1,
                        location=(product_x, product_y, product_z + 0.06)  # Center of box
                    )
                    obj = bpy.context.active_object
                    obj.name = f"Product_{shelf_id}_{slot_id}_{product['sku_id']}"
                    
                    # Scale to product size
                    scale = product.get('scale', 1.0)
                    obj.scale = (0.025 * scale, 0.02 * scale, 0.06 * scale)
                    
                    # Rotate
                    obj.rotation_euler[2] = math.radians(product.get('rotation', 0))
                    
                    # Get category from metadata or default
                    category = 'unknown'
                    if 'metadata' in self.recipe:
                        sku_dist = self.recipe['metadata'].get('category_distribution', {})
                        # Simple heuristic - in production would lookup from catalog
                    
                    # Create and apply material
                    color = category_colors.get(category, category_colors['unknown'])
                    mat = self._create_product_material(color)
                    
                    if obj.data.materials:
                        obj.data.materials[0] = mat
                    else:
                        obj.data.materials.append(mat)
                    
                    # Set custom properties for instance ID
                    obj.pass_index = self._get_instance_id(product['sku_id'])
    
    def _get_instance_id(self, sku_id: str) -> int:
        """Convert SKU ID to instance ID"""
        # Simple hash-based ID generation
        try:
            return int(sku_id.replace('SKU', '').replace('sku', ''))
        except:
            return hash(sku_id) % 10000
    
    def _setup_camera(self):
        """Setup camera based on recipe"""
        cam_pos = self.recipe['camera_position']
        cam_rot = self.recipe['camera_rotation']
        
        bpy.ops.object.camera_add(location=cam_pos)
        camera = bpy.context.active_object
        camera.rotation_euler = (
            math.radians(cam_rot[0]),
            math.radians(cam_rot[1]),
            math.radians(cam_rot[2])
        )
        
        # Set as active camera
        bpy.context.scene.camera = camera
        
        # Set FOV
        camera.data.lens_unit = 'FOV'
        camera.data.angle = math.radians(50)
        
        print(f"Camera positioned at {cam_pos}")
    
    def _setup_lighting(self):
        """Setup realistic lighting"""
        # Add sun light (main key light)
        bpy.ops.object.light_add(
            type='SUN',
            location=(0, 0, 5)
        )
        sun = bpy.context.active_object
        sun.data.energy = 2.0
        sun.rotation_euler = (math.radians(45), 0, math.radians(30))
        
        # Add area lights for fill
        for i, (x, y) in enumerate([(-2, -2), (2, -2)]):
            bpy.ops.object.light_add(
                type='AREA',
                location=(x, y, 3)
            )
            light = bpy.context.active_object
            light.data.energy = 100
            light.data.size = 2
            light.data.color = (1.0, 0.95, 0.9)  # Warm light
        
        print("Lighting setup complete")
    
    def render(self):
        """Render all required channels"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        scene_id = self.scene_id
        
        # Render RGB
        print("Rendering RGB...")
        rgb_path = self.output_dir / f"{scene_id}_rgb.png"
        bpy.context.scene.render.filepath = str(rgb_path)
        bpy.ops.render.render(write_still=True)
        print(f"RGB saved to {rgb_path}")
        
        # Setup and render instance ID pass
        print("Rendering instance mask...")
        self._render_instance_id()
        
        # Setup and render depth pass
        print("Rendering depth map...")
        self._render_depth()
        
        return str(self.output_dir)
    
    def _render_instance_id(self):
        """Render instance ID pass"""
        scene = bpy.context.scene
        scene.view_layers["ViewLayer"].use_pass_object_index = True
        
        # Setup compositor nodes to output instance ID
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
        
        render_layers = tree.nodes.new('CompositorNodeRLayers')
        output = tree.nodes.new('CompositorNodeOutputFile')
        output.base_path = str(self.output_dir)
        output.file_slots.clear()
        output.file_slots.new(f"{self.scene_id}_instance")
        output.format.file_format = 'PNG'
        output.format.color_mode = 'BW'
        output.format.color_depth = '16'
        
        tree.links.new(render_layers.outputs['IndexOB'], output.inputs[0])
        
        # Render
        bpy.ops.render.render(write_still=False, animation=False)
        print(f"Instance mask saved")
    
    def _render_depth(self):
        """Render depth pass"""
        scene = bpy.context.scene
        scene.view_layers["ViewLayer"].use_pass_z = True
        
        # Setup compositor for depth
        tree = scene.node_tree
        tree.nodes.clear()
        
        render_layers = tree.nodes.new('CompositorNodeRLayers')
        normalize = tree.nodes.new('CompositorNodeNormalize')
        output = tree.nodes.new('CompositorNodeOutputFile')
        output.base_path = str(self.output_dir)
        output.file_slots.clear()
        output.file_slots.new(f"{self.scene_id}_depth")
        output.format.file_format = 'PNG'
        output.format.color_mode = 'BW'
        
        tree.links.new(render_layers.outputs['Depth'], normalize.inputs[0])
        tree.links.new(normalize.outputs[0], output.inputs[0])
        
        # Render
        bpy.ops.render.render(write_still=False, animation=False)
        print(f"Depth map saved")


def main():
    """Main entry point when run from Blender"""
    import argparse
    
    # Parse arguments after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True, help="Path to scene recipe JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args(argv)
    
    print("=" * 60)
    print("Blender Synthetic Data Renderer")
    print("=" * 60)
    print(f"Recipe: {args.recipe}")
    print(f"Output: {args.output}")
    print()
    
    # Create renderer and execute
    try:
        renderer = BlenderRenderer(args.recipe, args.output)
        renderer.setup_scene()
        output_path = renderer.render()
        
        print()
        print("=" * 60)
        print(f"Rendering complete: {output_path}")
        print("=" * 60)
    except Exception as e:
        print(f"ERROR: Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

            shelf_height = shelf_data['height']
            num_slots = shelf_data['num_slots']
            slot_width = 1.2 / num_slots
            
            for slot_data in shelf_data['slots']:
                slot_id = slot_data['slot_id']
                
                if slot_data['is_empty']:
                    continue
                
                # Calculate slot center position
                x_base = -0.6 + (slot_id + 0.5) * slot_width
                y_base = 0
                z_base = shelf_height + 0.2  # Products sit on shelf
                
                for product in slot_data['products']:
                    # In production, this would load the actual product model
                    # For now, create a placeholder cube
                    bpy.ops.mesh.primitive_cube_add(
                        size=0.08,
                        location=(
                            x_base + product['position'][0],
                            y_base + product['position'][1],
                            z_base + product['position'][2]
                        )
                    )
                    obj = bpy.context.active_object
                    obj.name = f"Product_{shelf_id}_{slot_id}_{product['sku_id']}"
                    obj.rotation_euler[2] = math.radians(product['rotation'])
                    obj.scale = (product['scale'], product['scale'], product['scale'])
                    
                    # Set custom properties for instance ID
                    obj['sku_id'] = product['sku_id']
                    obj.pass_index = self._get_instance_id(product['sku_id'])
    
    def _get_instance_id(self, sku_id: str) -> int:
        """Convert SKU ID to instance ID"""
        # Simple hash-based ID generation
        return int(sku_id.replace('SKU', ''))
    
    def _setup_camera(self):
        """Setup camera based on recipe"""
        cam_pos = self.recipe['camera_position']
        cam_rot = self.recipe['camera_rotation']
        
        bpy.ops.object.camera_add(location=cam_pos)
        camera = bpy.context.active_object
        camera.rotation_euler = (
            math.radians(cam_rot[0]),
            math.radians(cam_rot[1]),
            math.radians(cam_rot[2])
        )
        
        # Set as active camera
        bpy.context.scene.camera = camera
        
        # Set FOV
        camera.data.lens_unit = 'FOV'
        camera.data.angle = math.radians(50)  # Can be loaded from config
    
    def _setup_lighting(self):
        """Setup scene lighting"""
        # Add environment lighting
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
        world.use_nodes = True
        
        # Add area lights
        for i in range(2):
            bpy.ops.object.light_add(
                type='AREA',
                location=((-1)**i * 2, -2, 3)
            )
            light = bpy.context.active_object
            light.data.energy = 100
            light.data.size = 2
    
    def render(self):
        """Render all required channels"""
        scene_id = self.recipe['scene_id']
        
        # Create output directory
        output_path = self.output_dir / scene_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Render RGB
        rgb_path = output_path / f"{scene_id}_rgb.png"
        bpy.context.scene.render.filepath = str(rgb_path)
        bpy.ops.render.render(write_still=True)
        
        # Render instance ID (using IndexOB pass)
        self._render_instance_id(output_path, scene_id)
        
        # Render depth
        self._render_depth(output_path, scene_id)
        
        return str(output_path)
    
    def _render_instance_id(self, output_path: Path, scene_id: str):
        """Render instance ID pass"""
        scene = bpy.context.scene
        scene.view_layers["ViewLayer"].use_pass_object_index = True
        
        # Setup compositor nodes to output instance ID
        tree = scene.node_tree
        tree.nodes.clear()
        
        render_layers = tree.nodes.new('CompositorNodeRLayers')
        output = tree.nodes.new('CompositorNodeOutputFile')
        output.base_path = str(output_path)
        output.file_slots[0].path = f"{scene_id}_instance"
        
        tree.links.new(render_layers.outputs['IndexOB'], output.inputs[0])
        
        bpy.ops.render.render(write_still=True)
    
    def _render_depth(self, output_path: Path, scene_id: str):
        """Render depth pass"""
        scene = bpy.context.scene
        scene.view_layers["ViewLayer"].use_pass_z = True
        
        # Setup compositor for depth
        tree = scene.node_tree
        tree.nodes.clear()
        
        render_layers = tree.nodes.new('CompositorNodeRLayers')
        normalize = tree.nodes.new('CompositorNodeNormalize')
        output = tree.nodes.new('CompositorNodeOutputFile')
        output.base_path = str(output_path)
        output.file_slots[0].path = f"{scene_id}_depth"
        
        tree.links.new(render_layers.outputs['Depth'], normalize.inputs[0])
        tree.links.new(normalize.outputs[0], output.inputs[0])
        
        bpy.ops.render.render(write_still=True)


def main():
    """Main entry point when run from Blender"""
    import argparse
    
    # Parse arguments after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True, help="Path to scene recipe JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args(argv)
    
    # Create renderer and execute
    renderer = BlenderRenderer(args.recipe, args.output)
    renderer.setup_scene()
    output_path = renderer.render()
    
    print(f"Rendering complete: {output_path}")


if __name__ == "__main__":
    # Check if running in Blender
    if 'bpy' in sys.modules:
        main()
    else:
        print("This script must be run inside Blender")
        print("Usage: blender --background --python blender_renderer.py -- --recipe scene.json --output ./out")
