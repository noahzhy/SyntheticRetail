"""
Manifest generation for dataset traceability and audit
"""
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ImageManifest:
    """Manifest for a single generated image"""
    scene_id: str
    image_path: str
    annotation_path: str
    timestamp: str
    seed: int
    recipe_hash: str
    num_products: int
    num_valid_annotations: int
    qc_passed: bool
    qc_issues: List[str]
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class DatasetManifest:
    """Manifest for entire dataset"""
    dataset_id: str
    creation_date: str
    config_hash: str
    total_scenes: int
    total_products: int
    failed_scenes: int
    distribution_stats: Dict[str, Any]
    images: List[ImageManifest]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        return data
    
    def save(self, output_path: str):
        """Save manifest to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, manifest_path: str) -> 'DatasetManifest':
        """Load manifest from JSON file"""
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        
        # Convert image manifests
        images = [ImageManifest(**img) for img in data.pop('images')]
        
        return cls(**data, images=images)


class ManifestGenerator:
    """Generate and manage dataset manifests"""
    
    def __init__(self):
        self.image_manifests: List[ImageManifest] = []
    
    def add_image(
        self,
        scene_id: str,
        image_path: str,
        annotation_path: str,
        seed: int,
        recipe_hash: str,
        num_products: int,
        num_valid_annotations: int,
        qc_passed: bool,
        qc_issues: List[str],
        metrics: Dict[str, Any]
    ):
        """Add image to manifest"""
        manifest = ImageManifest(
            scene_id=scene_id,
            image_path=str(image_path),
            annotation_path=str(annotation_path),
            timestamp=datetime.now().isoformat(),
            seed=seed,
            recipe_hash=recipe_hash,
            num_products=num_products,
            num_valid_annotations=num_valid_annotations,
            qc_passed=qc_passed,
            qc_issues=qc_issues,
            metrics=metrics
        )
        self.image_manifests.append(manifest)
    
    def generate_dataset_manifest(
        self,
        dataset_id: str,
        config_hash: str,
        distribution_stats: Dict[str, Any]
    ) -> DatasetManifest:
        """Generate complete dataset manifest"""
        total_products = sum(img.num_products for img in self.image_manifests)
        failed_scenes = sum(1 for img in self.image_manifests if not img.qc_passed)
        
        return DatasetManifest(
            dataset_id=dataset_id,
            creation_date=datetime.now().isoformat(),
            config_hash=config_hash,
            total_scenes=len(self.image_manifests),
            total_products=total_products,
            failed_scenes=failed_scenes,
            distribution_stats=distribution_stats,
            images=self.image_manifests
        )
    
    def save_manifest(
        self,
        output_path: str,
        dataset_id: str,
        config_hash: str,
        distribution_stats: Dict[str, Any]
    ):
        """Save complete manifest to file"""
        manifest = self.generate_dataset_manifest(
            dataset_id, config_hash, distribution_stats
        )
        manifest.save(output_path)


def calculate_hash(data: Any) -> str:
    """Calculate hash for data traceability"""
    import hashlib
    from dataclasses import is_dataclass, asdict
    
    # Convert dataclass to dict if needed
    if is_dataclass(data):
        data = asdict(data)
    elif isinstance(data, dict):
        # Recursively convert nested dataclasses
        data = {k: asdict(v) if is_dataclass(v) else v for k, v in data.items()}
    
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]
