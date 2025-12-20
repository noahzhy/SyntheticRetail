"""
Core configuration management for Synthetic Retail Data Factory
"""
import yaml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Dataset output configuration"""
    output_dir: str
    images_dir: str
    annotations_dir: str
    manifests_dir: str


@dataclass
class SceneConfig:
    """Scene generation configuration"""
    seed_base: int
    num_scenes: int
    shelves_per_scene: List[int]
    slots_per_shelf: List[int]


@dataclass
class ShelfConfig:
    """Physical shelf parameters"""
    width: float
    height: float
    depth: float
    vertical_spacing: float


@dataclass
class SlotConfig:
    """Slot parameters"""
    width: float
    height: float
    depth: float


@dataclass
class PlacementConfig:
    """Product placement rules"""
    max_products_per_slot: int
    min_products_per_slot: int
    occlusion_threshold: float
    allow_empty_slots: bool
    category_grouping: bool


@dataclass
class RenderingConfig:
    """Rendering configuration"""
    resolution: List[int]
    samples: int
    use_denoising: bool
    channels: List[str]


@dataclass
class CameraConfig:
    """Camera configuration"""
    height: List[float]
    distance: List[float]
    angle_horizontal: List[float]
    angle_vertical: List[float]
    fov: float


@dataclass
class AnnotationConfig:
    """Annotation export configuration"""
    format: str
    include_occluded: bool
    min_bbox_area: int
    polygon_simplification_tolerance: float


@dataclass
class QCConfig:
    """Quality control configuration"""
    min_products_per_image: int
    max_occlusion_rate: float
    check_bbox_validity: bool
    check_empty_images: bool
    resample_on_failure: bool
    max_resample_attempts: int


@dataclass
class DistributionConfig:
    """Distribution monitoring configuration"""
    track_sku_balance: bool
    track_category_balance: bool
    track_occlusion_distribution: bool
    report_frequency: int


@dataclass
class Config:
    """Main configuration class"""
    dataset: DatasetConfig
    scene: SceneConfig
    shelf: ShelfConfig
    slot: SlotConfig
    placement: PlacementConfig
    rendering: RenderingConfig
    camera: CameraConfig
    annotation: AnnotationConfig
    qc: QCConfig
    distribution: DistributionConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            dataset=DatasetConfig(**data['dataset']),
            scene=SceneConfig(**data['scene']),
            shelf=ShelfConfig(**data['shelf']),
            slot=SlotConfig(**data['slot']),
            placement=PlacementConfig(**data['placement']),
            rendering=RenderingConfig(**data['rendering']),
            camera=CameraConfig(**data['camera']),
            annotation=AnnotationConfig(**data['annotation']),
            qc=QCConfig(**data['qc']),
            distribution=DistributionConfig(**data['distribution'])
        )
    
    def get_output_paths(self) -> Dict[str, Path]:
        """Get resolved output paths"""
        base = Path(self.dataset.output_dir)
        return {
            'base': base,
            'images': base / self.dataset.images_dir,
            'annotations': base / self.dataset.annotations_dir,
            'manifests': base / self.dataset.manifests_dir
        }


@dataclass
class SKU:
    """Product SKU definition"""
    id: str
    name: str
    category: str
    subcategory: str
    blend_file: str
    object_name: str
    dimensions: List[float]  # [width, depth, height]
    weight: float = 1.0


@dataclass
class SKUCatalog:
    """SKU catalog management"""
    skus: List[SKU]
    category_affinity: Dict[str, List[str]]
    constraints: Dict[str, List[str]]
    
    @classmethod
    def from_yaml(cls, catalog_path: str) -> 'SKUCatalog':
        """Load SKU catalog from YAML file"""
        with open(catalog_path, 'r') as f:
            data = yaml.safe_load(f)
        
        skus = [SKU(**sku_data) for sku_data in data['skus']]
        category_affinity = data.get('category_affinity', {})
        constraints = data.get('constraints', {})
        
        return cls(
            skus=skus,
            category_affinity=category_affinity,
            constraints=constraints
        )
    
    def get_sku_by_id(self, sku_id: str) -> SKU:
        """Get SKU by ID"""
        for sku in self.skus:
            if sku.id == sku_id:
                return sku
        raise ValueError(f"SKU {sku_id} not found")
    
    def get_skus_by_category(self, category: str) -> List[SKU]:
        """Get all SKUs in a category"""
        return [sku for sku in self.skus if sku.category == category]
    
    def get_all_categories(self) -> List[str]:
        """Get list of all unique categories"""
        return list(set(sku.category for sku in self.skus))
