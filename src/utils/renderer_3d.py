"""
3D renderer for generating synthetic shelf images with perspective
Uses matplotlib for 3D visualization and rendering to create more realistic scenes
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, List
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.rules.scene_generator import SceneRecipe


class Renderer3D:
    """3D renderer for more realistic shelf visualization"""
    
    def __init__(self, resolution: Tuple[int, int] = (1920, 1080)):
        self.width, self.height = resolution
        
        # Colors for different product categories
        self.category_colors = {
            'beverages': (0.3, 0.5, 0.8),    # Blue
            'snacks': (0.2, 0.7, 0.3),       # Green
            'dairy': (0.9, 0.9, 0.4),        # Yellow
            'unknown': (0.5, 0.5, 0.5)       # Gray
        }
    
    def render_scene(
        self, 
        recipe: SceneRecipe, 
        output_dir: Path,
        sku_catalog: dict
    ) -> Tuple[str, str, str]:
        """
        Render a 3D scene and return paths to RGB, instance mask, and depth images
        
        Returns:
            Tuple of (rgb_path, instance_path, depth_path)
        """
        scene_id = recipe.scene_id
        
        # Create 3D visualization
        rgb_image = self._render_3d_scene(recipe, sku_catalog)
        
        # Create instance mask and depth from 3D data
        instance_mask = self._render_instance_mask(recipe)
        depth_map = self._render_depth(recipe)
        
        # Save images
        output_dir.mkdir(parents=True, exist_ok=True)
        
        rgb_path = output_dir / f"{scene_id}_rgb.png"
        instance_path = output_dir / f"{scene_id}_instance.png"
        depth_path = output_dir / f"{scene_id}_depth.png"
        
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(instance_path), instance_mask)
        cv2.imwrite(str(depth_path), depth_map)
        
        return str(rgb_path), str(instance_path), str(depth_path)
    
    def _render_3d_scene(self, recipe: SceneRecipe, sku_catalog: dict) -> np.ndarray:
        """Render 3D scene using matplotlib"""
        # Create figure with specific size for resolution
        dpi = 100
        fig = plt.figure(figsize=(self.width/dpi, self.height/dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set view angle from camera parameters
        cam_pos = recipe.camera_position
        cam_rot = recipe.camera_rotation
        
        # Calculate view parameters
        elev = cam_rot[0] - 90  # Pitch angle
        azim = cam_rot[2]       # Yaw angle
        
        ax.view_init(elev=elev, azim=azim)
        
        # Set background color (light gray for store wall)
        ax.set_facecolor('#f0f0f0')
        fig.patch.set_facecolor('#f0f0f0')
        
        # Render shelves and products
        for shelf in recipe.shelves:
            self._add_shelf_to_plot(ax, shelf, recipe, sku_catalog)
        
        # Set axis limits for proper framing
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(0, len(recipe.shelves) * 0.5 + 0.5)
        
        # Remove axes for cleaner look
        ax.set_axis_off()
        
        # Adjust layout
        plt.tight_layout(pad=0)
        
        # Render to numpy array
        fig.canvas.draw()
        
        # Convert to numpy array using buffer_rgba
        buf = np.array(fig.canvas.renderer.buffer_rgba())
        
        # Convert RGBA to RGB
        buf = buf[:, :, :3]
        
        plt.close(fig)
        
        # Resize if needed to match exact resolution
        if buf.shape[:2] != (self.height, self.width):
            buf = cv2.resize(buf, (self.width, self.height))
        
        return buf
    
    def _add_shelf_to_plot(self, ax, shelf, recipe: SceneRecipe, sku_catalog: dict):
        """Add shelf and products to 3D plot"""
        shelf_height = shelf.height
        shelf_width = 1.2
        shelf_depth = 0.3
        
        # Draw shelf surface (brown wooden shelf)
        x_shelf = np.array([-shelf_width/2, shelf_width/2, shelf_width/2, -shelf_width/2])
        y_shelf = np.array([-shelf_depth/2, -shelf_depth/2, shelf_depth/2, shelf_depth/2])
        z_shelf = np.full(4, shelf_height)
        
        # Create shelf as 3D polygon
        vertices_shelf = [list(zip(x_shelf, y_shelf, z_shelf))]
        shelf_poly = Poly3DCollection(vertices_shelf, alpha=0.8, facecolor='#8B4513', edgecolor='#654321')
        ax.add_collection3d(shelf_poly)
        
        # Add shelf support (vertical bars)
        support_height = 0.1
        for x_pos in [-shelf_width/2, shelf_width/2]:
            vertices_support = self._create_box(
                x_pos - 0.02, -shelf_depth/2, shelf_height - support_height,
                0.04, shelf_depth, support_height
            )
            support_poly = Poly3DCollection(vertices_support, alpha=0.9, facecolor='#654321', edgecolor='#4a3319')
            ax.add_collection3d(support_poly)
        
        # Calculate slot positions
        num_slots = shelf.num_slots
        slot_width = shelf_width / num_slots
        
        # Add products
        for slot in shelf.slots:
            if slot.is_empty:
                continue
            
            slot_center_x = -shelf_width/2 + (slot.slot_id + 0.5) * slot_width
            
            # Spread products within slot
            num_products = len(slot.products)
            for prod_idx, product in enumerate(slot.products):
                # Get product info
                sku_id = product.sku_id
                category = 'unknown'
                
                if sku_catalog and 'skus' in sku_catalog:
                    for sku in sku_catalog['skus']:
                        if sku['id'] == sku_id:
                            category = sku.get('category', 'unknown')
                            break
                
                # Calculate product position with spacing
                if num_products == 1:
                    offset_x = 0
                else:
                    spacing = slot_width * 0.7 / max(num_products, 1)
                    offset_x = (prod_idx - (num_products - 1) / 2) * spacing
                
                product_x = slot_center_x + offset_x + product.position[0] * 0.02
                product_y = product.position[1] * 0.02
                product_z = shelf_height
                
                # Product dimensions (scale based on category)
                width = 0.05 * product.scale
                depth = 0.04 * product.scale
                height = 0.12 * product.scale
                
                # Get color for category
                color = self.category_colors.get(category, self.category_colors['unknown'])
                
                # Add slight color variation
                color = tuple(min(1.0, max(0.0, c + random.uniform(-0.1, 0.1))) for c in color)
                
                # Create 3D box for product
                vertices = self._create_box(
                    product_x - width/2, product_y - depth/2, product_z,
                    width, depth, height
                )
                
                # Add product with rotation
                product_poly = Poly3DCollection(
                    vertices, 
                    alpha=0.9, 
                    facecolor=color, 
                    edgecolor=(0, 0, 0, 0.5),
                    linewidths=0.5
                )
                ax.add_collection3d(product_poly)
                
                # Add label on front face (optional)
                label_z = product_z + height / 2
                ax.text(
                    product_x, product_y - depth/2 - 0.01, label_z,
                    sku_id.replace('SKU', ''),
                    fontsize=6,
                    ha='center',
                    va='center',
                    color='black'
                )
    
    def _create_box(self, x, y, z, width, depth, height):
        """Create vertices for a 3D box"""
        # Define 8 vertices of the box
        vertices = [
            [x, y, z],
            [x + width, y, z],
            [x + width, y + depth, z],
            [x, y + depth, z],
            [x, y, z + height],
            [x + width, y, z + height],
            [x + width, y + depth, z + height],
            [x, y + depth, z + height]
        ]
        
        # Define the 6 faces of the box
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[2], vertices[3]]   # Bottom
        ]
        
        return faces
    
    def _render_instance_mask(self, recipe: SceneRecipe) -> np.ndarray:
        """Render instance segmentation mask (2D projection)"""
        mask = np.zeros((self.height, self.width), dtype=np.uint16)
        
        # Project 3D positions to 2D screen coordinates
        shelf_width = 1.2
        
        instance_id = 1
        
        for shelf in recipe.shelves:
            num_slots = shelf.num_slots
            slot_width = shelf_width / num_slots
            shelf_y_screen = int(self.height * (1 - (shelf.height + 0.3) / 2.5))
            
            for slot in shelf.slots:
                if slot.is_empty:
                    continue
                
                slot_center_x = -shelf_width/2 + (slot.slot_id + 0.5) * slot_width
                num_products = len(slot.products)
                
                for prod_idx, product in enumerate(slot.products):
                    # Calculate product position
                    if num_products == 1:
                        offset_x = 0
                    else:
                        spacing = slot_width * 0.7 / max(num_products, 1)
                        offset_x = (prod_idx - (num_products - 1) / 2) * spacing
                    
                    product_x = slot_center_x + offset_x + product.position[0] * 0.02
                    
                    # Convert 3D to 2D screen coordinates
                    screen_x = int((product_x / shelf_width + 0.5) * self.width)
                    screen_y = shelf_y_screen
                    
                    # Product dimensions on screen
                    width_screen = int(50 * product.scale)
                    height_screen = int(120 * product.scale)
                    
                    # Draw rectangle for instance
                    cv2.rectangle(
                        mask,
                        (screen_x - width_screen//2, screen_y - height_screen),
                        (screen_x + width_screen//2, screen_y),
                        instance_id,
                        -1
                    )
                    
                    instance_id += 1
        
        return mask
    
    def _render_depth(self, recipe: SceneRecipe) -> np.ndarray:
        """Render depth map based on 3D positions"""
        depth = np.ones((self.height, self.width), dtype=np.uint8) * 255
        
        shelf_width = 1.2
        
        for shelf in recipe.shelves:
            num_slots = shelf.num_slots
            slot_width = shelf_width / num_slots
            shelf_y_screen = int(self.height * (1 - (shelf.height + 0.3) / 2.5))
            
            # Draw shelf depth
            shelf_x_start = int(0.1 * self.width)
            shelf_x_end = int(0.9 * self.width)
            cv2.rectangle(
                depth,
                (shelf_x_start, shelf_y_screen - 10),
                (shelf_x_end, shelf_y_screen),
                180,
                -1
            )
            
            for slot in shelf.slots:
                if slot.is_empty:
                    continue
                
                slot_center_x = -shelf_width/2 + (slot.slot_id + 0.5) * slot_width
                num_products = len(slot.products)
                
                for prod_idx, product in enumerate(slot.products):
                    # Calculate product position
                    if num_products == 1:
                        offset_x = 0
                    else:
                        spacing = slot_width * 0.7 / max(num_products, 1)
                        offset_x = (prod_idx - (num_products - 1) / 2) * spacing
                    
                    product_x = slot_center_x + offset_x + product.position[0] * 0.02
                    product_depth = product.position[1]
                    
                    # Convert 3D to 2D screen coordinates
                    screen_x = int((product_x / shelf_width + 0.5) * self.width)
                    screen_y = shelf_y_screen
                    
                    # Product dimensions on screen
                    width_screen = int(50 * product.scale)
                    height_screen = int(120 * product.scale)
                    
                    # Depth value based on y position (closer = darker)
                    depth_value = int(120 - product_depth * 1000)
                    depth_value = max(50, min(200, depth_value))
                    
                    # Draw rectangle for depth
                    cv2.rectangle(
                        depth,
                        (screen_x - width_screen//2, screen_y - height_screen),
                        (screen_x + width_screen//2, screen_y),
                        depth_value,
                        -1
                    )
        
        return depth
