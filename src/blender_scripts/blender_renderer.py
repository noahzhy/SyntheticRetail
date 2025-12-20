"""
Blender rendering script for synthetic retail data generation.
This script should be run inside Blender with bpy available.

Usage:
    blender --background --python blender_renderer.py -- --recipe scene_recipe.json --output ./output
"""
import bpy
import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import math


class BlenderRenderer:
    """Handles Blender scene setup and rendering"""
    
    def __init__(self, recipe_path: str, output_dir: str):
        self.recipe_path = recipe_path
        self.output_dir = Path(output_dir)
        self.recipe = self._load_recipe()
        
    def _load_recipe(self) -> Dict[str, Any]:
        """Load scene recipe from JSON"""
        with open(self.recipe_path, 'r') as f:
            return json.load(f)
    
    def setup_scene(self):
        """Setup Blender scene based on recipe"""
        # Clear existing scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # Setup render settings
        self._setup_render_settings()
        
        # Create shelves and slots
        self._create_shelf_structure()
        
        # Place products
        self._place_products()
        
        # Setup camera
        self._setup_camera()
        
        # Setup lighting
        self._setup_lighting()
    
    def _setup_render_settings(self):
        """Configure rendering settings"""
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 128  # Can be loaded from config
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        scene.render.film_transparent = False
        
        # Setup compositor for multi-channel output
        scene.use_nodes = True
        scene.render.use_compositing = True
        
    def _create_shelf_structure(self):
        """Create shelf geometry"""
        # This is a simplified version - in production, this would load
        # a pre-made shelf model with Geometry Nodes slot system
        
        for shelf_data in self.recipe['shelves']:
            shelf_id = shelf_data['shelf_id']
            height = shelf_data['height']
            num_slots = shelf_data['num_slots']
            
            # Create shelf plane
            bpy.ops.mesh.primitive_plane_add(
                size=1.2,
                location=(0, 0, height)
            )
            shelf = bpy.context.active_object
            shelf.name = f"Shelf_{shelf_id}"
            
            # Scale to shelf dimensions (1.2m x 0.3m)
            shelf.scale = (0.6, 0.15, 1)
            
            # Create slot markers (for debugging/visualization)
            slot_width = 1.2 / num_slots
            for i in range(num_slots):
                x_pos = -0.6 + (i + 0.5) * slot_width
                bpy.ops.mesh.primitive_cube_add(
                    size=0.01,
                    location=(x_pos, 0, height + 0.01)
                )
                marker = bpy.context.active_object
                marker.name = f"Slot_{shelf_id}_{i}"
                marker.hide_render = True  # Hidden in final render
    
    def _place_products(self):
        """Place products according to recipe"""
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
