"""
Simple 2D renderer for generating synthetic shelf images
This is a placeholder renderer until Blender integration is complete
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, List
import random

from src.rules.scene_generator import SceneRecipe


class SimpleRenderer:
    """Simple 2D renderer for demonstration purposes"""
    
    def __init__(self, resolution: Tuple[int, int] = (1920, 1080)):
        self.width, self.height = resolution
        
        # Colors for different product categories
        self.category_colors = {
            'beverages': (180, 120, 60),    # Blue tones
            'snacks': (50, 180, 50),        # Green tones
            'dairy': (220, 220, 100),       # Yellow tones
            'unknown': (128, 128, 128)      # Gray
        }
        
        # Product shape templates
        self.product_shapes = {
            'bottle': 'bottle',
            'box': 'box',
            'can': 'can'
        }
    
    def render_scene(
        self, 
        recipe: SceneRecipe, 
        output_dir: Path,
        sku_catalog: dict
    ) -> Tuple[str, str, str]:
        """
        Render a scene and return paths to RGB, instance mask, and depth images
        
        Returns:
            Tuple of (rgb_path, instance_path, depth_path)
        """
        scene_id = recipe.scene_id
        
        # Create RGB image
        rgb_image = self._render_rgb(recipe, sku_catalog)
        
        # Create instance mask
        instance_mask = self._render_instance_mask(recipe)
        
        # Create depth map
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
    
    def _render_rgb(self, recipe: SceneRecipe, sku_catalog: dict) -> np.ndarray:
        """Render RGB image"""
        # Create background (store wall)
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
        
        # Add some texture to background
        noise = np.random.randint(-20, 20, (self.height, self.width, 3), dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Calculate shelf positions
        total_height = len(recipe.shelves) * 200  # pixels per shelf
        start_y = (self.height - total_height) // 2
        
        # Render shelves and products
        for shelf_idx, shelf in enumerate(recipe.shelves):
            shelf_y = start_y + shelf_idx * 200
            
            # Draw shelf
            cv2.rectangle(
                image,
                (50, shelf_y + 150),
                (self.width - 50, shelf_y + 160),
                (139, 69, 19),  # Brown
                -1
            )
            
            # Calculate slot positions
            slot_width = (self.width - 100) // shelf.num_slots
            
            for slot in shelf.slots:
                if slot.is_empty:
                    continue
                
                slot_x = 50 + slot.slot_id * slot_width
                
                # Render products in this slot
                num_products = len(slot.products)
                for prod_idx, product in enumerate(slot.products):
                    # Get product info
                    sku_id = product.sku_id
                    category = 'unknown'
                    
                    # Try to get category from catalog
                    if sku_catalog and 'skus' in sku_catalog:
                        for sku in sku_catalog['skus']:
                            if sku['id'] == sku_id:
                                category = sku.get('category', 'unknown')
                                break
                    
                    # Calculate product position with better spacing
                    # Spread products across the slot width
                    if num_products == 1:
                        offset_x = 0
                    else:
                        # Spread products evenly across slot
                        spacing = slot_width * 0.7 / max(num_products, 1)
                        offset_x = (prod_idx - (num_products - 1) / 2) * spacing
                    
                    product_x = slot_x + slot_width // 2 + int(offset_x) + int(product.position[0] * 10)  # Reduced position variation
                    product_y = shelf_y + 100
                    
                    # Get color for category
                    base_color = self.category_colors.get(category, self.category_colors['unknown'])
                    
                    # Add variation
                    color = tuple(int(c + random.randint(-20, 20)) for c in base_color)
                    color = tuple(max(0, min(255, c)) for c in color)
                    
                    # Draw product (simple box with label)
                    width = int(50 * product.scale)
                    height = int(120 * product.scale)
                    
                    # Product body
                    cv2.rectangle(
                        image,
                        (product_x - width//2, product_y - height),
                        (product_x + width//2, product_y),
                        color,
                        -1
                    )
                    
                    # Product outline
                    cv2.rectangle(
                        image,
                        (product_x - width//2, product_y - height),
                        (product_x + width//2, product_y),
                        (0, 0, 0),
                        2
                    )
                    
                    # Add simple label
                    label_y = product_y - height // 2
                    cv2.rectangle(
                        image,
                        (product_x - width//2 + 5, label_y - 15),
                        (product_x + width//2 - 5, label_y + 15),
                        (255, 255, 255),
                        -1
                    )
                    
                    # Draw SKU text (simplified)
                    sku_num = sku_id.replace('SKU', '')
                    cv2.putText(
                        image,
                        sku_num,
                        (product_x - 10, label_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 0),
                        1
                    )
        
        return image
    
    def _render_instance_mask(self, recipe: SceneRecipe) -> np.ndarray:
        """Render instance segmentation mask"""
        mask = np.zeros((self.height, self.width), dtype=np.uint16)
        
        # Calculate shelf positions
        total_height = len(recipe.shelves) * 200
        start_y = (self.height - total_height) // 2
        
        instance_id = 1
        
        for shelf_idx, shelf in enumerate(recipe.shelves):
            shelf_y = start_y + shelf_idx * 200
            slot_width = (self.width - 100) // shelf.num_slots
            
            for slot in shelf.slots:
                if slot.is_empty:
                    continue
                
                slot_x = 50 + slot.slot_id * slot_width
                num_products = len(slot.products)
                
                for prod_idx, product in enumerate(slot.products):
                    # Use same spacing logic as RGB rendering
                    if num_products == 1:
                        offset_x = 0
                    else:
                        spacing = slot_width * 0.7 / max(num_products, 1)
                        offset_x = (prod_idx - (num_products - 1) / 2) * spacing
                    
                    product_x = slot_x + slot_width // 2 + int(offset_x) + int(product.position[0] * 10)
                    product_y = shelf_y + 100
                    
                    width = int(50 * product.scale)
                    height = int(120 * product.scale)
                    
                    # Draw product with instance ID
                    cv2.rectangle(
                        mask,
                        (product_x - width//2, product_y - height),
                        (product_x + width//2, product_y),
                        instance_id,
                        -1
                    )
                    
                    instance_id += 1
        
        return mask
    
    def _render_depth(self, recipe: SceneRecipe) -> np.ndarray:
        """Render depth map"""
        depth = np.ones((self.height, self.width), dtype=np.uint8) * 255  # Far = white
        
        # Calculate shelf positions
        total_height = len(recipe.shelves) * 200
        start_y = (self.height - total_height) // 2
        
        for shelf_idx, shelf in enumerate(recipe.shelves):
            shelf_y = start_y + shelf_idx * 200
            
            # Shelf depth
            cv2.rectangle(
                depth,
                (50, shelf_y + 150),
                (self.width - 50, shelf_y + 160),
                200,  # Mid-depth
                -1
            )
            
            slot_width = (self.width - 100) // shelf.num_slots
            
            for slot in shelf.slots:
                if slot.is_empty:
                    continue
                
                slot_x = 50 + slot.slot_id * slot_width
                num_products = len(slot.products)
                
                for prod_idx, product in enumerate(slot.products):
                    # Use same spacing logic as RGB rendering
                    if num_products == 1:
                        offset_x = 0
                    else:
                        spacing = slot_width * 0.7 / max(num_products, 1)
                        offset_x = (prod_idx - (num_products - 1) / 2) * spacing
                    
                    product_x = slot_x + slot_width // 2 + int(offset_x) + int(product.position[0] * 10)
                    product_y = shelf_y + 100
                    
                    width = int(50 * product.scale)
                    height = int(120 * product.scale)
                    
                    # Products are closer (darker in depth map)
                    depth_value = 100 + random.randint(-20, 20)
                    cv2.rectangle(
                        depth,
                        (product_x - width//2, product_y - height),
                        (product_x + width//2, product_y),
                        depth_value,
                        -1
                    )
        
        return depth
