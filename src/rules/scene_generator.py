"""
Scene recipe generation with reproducible seeds and rule-based constraints
"""
import random
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
import json

from src.config import Config, SKUCatalog, SKU


@dataclass
class ProductPlacement:
    """Single product placement in a slot"""
    sku_id: str
    position: Tuple[float, float, float]  # x, y, z offset within slot
    rotation: float  # rotation around z-axis in degrees
    scale: float = 1.0


@dataclass
class SlotRecipe:
    """Recipe for a single slot"""
    slot_id: int
    products: List[ProductPlacement] = field(default_factory=list)
    is_empty: bool = False


@dataclass
class ShelfRecipe:
    """Recipe for a single shelf"""
    shelf_id: int
    num_slots: int
    slots: List[SlotRecipe] = field(default_factory=list)
    height: float = 0.0  # vertical position


@dataclass
class SceneRecipe:
    """Complete scene recipe with all placement information"""
    scene_id: str
    seed: int
    shelves: List[ShelfRecipe] = field(default_factory=list)
    camera_position: Tuple[float, float, float] = (0, 0, 0)
    camera_rotation: Tuple[float, float, float] = (0, 0, 0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneRecipe':
        """Create from dictionary"""
        shelves = [
            ShelfRecipe(
                shelf_id=shelf['shelf_id'],
                num_slots=shelf['num_slots'],
                height=shelf['height'],
                slots=[
                    SlotRecipe(
                        slot_id=slot['slot_id'],
                        is_empty=slot['is_empty'],
                        products=[
                            ProductPlacement(**prod) 
                            for prod in slot['products']
                        ]
                    )
                    for slot in shelf['slots']
                ]
            )
            for shelf in data['shelves']
        ]
        
        return cls(
            scene_id=data['scene_id'],
            seed=data['seed'],
            shelves=shelves,
            camera_position=tuple(data['camera_position']),
            camera_rotation=tuple(data['camera_rotation']),
            metadata=data.get('metadata', {})
        )


class RuleEngine:
    """Rule-based scene generation engine"""
    
    def __init__(self, config: Config, catalog: SKUCatalog):
        self.config = config
        self.catalog = catalog
        
    def generate_scene_seed(self, scene_idx: int) -> int:
        """Generate reproducible seed for a scene"""
        seed_str = f"{self.config.scene.seed_base}_{scene_idx}"
        hash_obj = hashlib.md5(seed_str.encode())
        return int(hash_obj.hexdigest(), 16) % (2**31)
    
    def generate_scene_recipe(self, scene_idx: int) -> SceneRecipe:
        """Generate a complete scene recipe"""
        seed = self.generate_scene_seed(scene_idx)
        rng = random.Random(seed)
        
        scene_id = f"scene_{scene_idx:06d}"
        
        # Determine number of shelves
        num_shelves = rng.randint(*self.config.scene.shelves_per_scene)
        
        # Generate shelves
        shelves = []
        for shelf_idx in range(num_shelves):
            shelf_recipe = self._generate_shelf_recipe(shelf_idx, rng)
            shelves.append(shelf_recipe)
        
        # Generate camera parameters
        camera_pos, camera_rot = self._generate_camera_params(rng, num_shelves)
        
        # Create metadata
        metadata = self._generate_metadata(scene_idx, seed, shelves)
        
        return SceneRecipe(
            scene_id=scene_id,
            seed=seed,
            shelves=shelves,
            camera_position=camera_pos,
            camera_rotation=camera_rot,
            metadata=metadata
        )
    
    def _generate_shelf_recipe(self, shelf_idx: int, rng: random.Random) -> ShelfRecipe:
        """Generate recipe for a single shelf"""
        num_slots = rng.randint(*self.config.scene.slots_per_shelf)
        height = shelf_idx * self.config.shelf.vertical_spacing
        
        # Determine category for this shelf (category grouping)
        shelf_category = None
        if self.config.placement.category_grouping:
            shelf_category = rng.choice(self.catalog.get_all_categories())
        
        slots = []
        for slot_idx in range(num_slots):
            slot_recipe = self._generate_slot_recipe(
                slot_idx, rng, shelf_category
            )
            slots.append(slot_recipe)
        
        return ShelfRecipe(
            shelf_id=shelf_idx,
            num_slots=num_slots,
            height=height,
            slots=slots
        )
    
    def _generate_slot_recipe(
        self, 
        slot_idx: int, 
        rng: random.Random,
        preferred_category: Optional[str] = None
    ) -> SlotRecipe:
        """Generate recipe for a single slot"""
        
        # Decide if slot should be empty
        if self.config.placement.allow_empty_slots and rng.random() < 0.15:
            return SlotRecipe(slot_id=slot_idx, is_empty=True)
        
        # Determine number of products
        num_products = rng.randint(
            self.config.placement.min_products_per_slot,
            self.config.placement.max_products_per_slot
        )
        
        if num_products == 0:
            return SlotRecipe(slot_id=slot_idx, is_empty=True)
        
        # Select products
        products = []
        available_skus = self.catalog.skus
        
        # Filter by category if grouping is enabled
        if preferred_category:
            available_skus = self.catalog.get_skus_by_category(preferred_category)
            if not available_skus:
                available_skus = self.catalog.skus
        
        for prod_idx in range(num_products):
            # Weighted random selection
            weights = [sku.weight for sku in available_skus]
            sku = rng.choices(available_skus, weights=weights, k=1)[0]
            
            # Generate placement parameters
            x_offset = (prod_idx - num_products / 2) * 0.02  # slight offset
            y_offset = rng.uniform(-0.01, 0.01)
            z_offset = 0.0
            
            rotation = rng.uniform(-5, 5)  # slight random rotation
            scale = rng.uniform(0.95, 1.05)  # slight scale variation
            
            placement = ProductPlacement(
                sku_id=sku.id,
                position=(x_offset, y_offset, z_offset),
                rotation=rotation,
                scale=scale
            )
            products.append(placement)
        
        return SlotRecipe(
            slot_id=slot_idx,
            products=products,
            is_empty=False
        )
    
    def _generate_camera_params(
        self, 
        rng: random.Random, 
        num_shelves: int
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Generate camera position and rotation"""
        
        # Camera height based on shelf configuration
        shelf_center_height = (num_shelves - 1) * self.config.shelf.vertical_spacing / 2
        cam_height = shelf_center_height + rng.uniform(*self.config.camera.height)
        
        # Camera distance
        cam_distance = rng.uniform(*self.config.camera.distance)
        
        # Camera angles
        angle_h = rng.uniform(*self.config.camera.angle_horizontal)
        angle_v = rng.uniform(*self.config.camera.angle_vertical)
        
        # Calculate position (camera facing shelf from front)
        import math
        cam_x = cam_distance * math.sin(math.radians(angle_h))
        cam_y = -cam_distance * math.cos(math.radians(angle_h))
        cam_z = cam_height
        
        # Rotation (Euler angles)
        rot_x = 90 + angle_v  # pitch
        rot_y = 0  # roll
        rot_z = angle_h  # yaw
        
        return (cam_x, cam_y, cam_z), (rot_x, rot_y, rot_z)
    
    def _generate_metadata(
        self, 
        scene_idx: int, 
        seed: int, 
        shelves: List[ShelfRecipe]
    ) -> Dict[str, Any]:
        """Generate scene metadata"""
        
        # Count products by SKU and category
        sku_counts = {}
        category_counts = {}
        
        total_products = 0
        total_slots = 0
        empty_slots = 0
        
        for shelf in shelves:
            total_slots += shelf.num_slots
            for slot in shelf.slots:
                if slot.is_empty:
                    empty_slots += 1
                else:
                    for product in slot.products:
                        total_products += 1
                        sku_counts[product.sku_id] = sku_counts.get(product.sku_id, 0) + 1
                        
                        # Get category
                        try:
                            sku = self.catalog.get_sku_by_id(product.sku_id)
                            category_counts[sku.category] = category_counts.get(sku.category, 0) + 1
                        except ValueError:
                            pass
        
        return {
            'scene_index': scene_idx,
            'seed': seed,
            'num_shelves': len(shelves),
            'total_slots': total_slots,
            'empty_slots': empty_slots,
            'total_products': total_products,
            'sku_distribution': sku_counts,
            'category_distribution': category_counts
        }
