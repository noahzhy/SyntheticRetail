"""
Main pipeline orchestration for synthetic data generation
"""
import sys
from pathlib import Path
from typing import Optional
import subprocess
import json
import hashlib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, SKUCatalog
from src.rules.scene_generator import RuleEngine, SceneRecipe
from src.utils.quality_control import QualityController, DistributionMonitor
from src.utils.manifest import ManifestGenerator, calculate_hash
from src.utils.simple_renderer import SimpleRenderer
import yaml


class SyntheticDataPipeline:
    """Main pipeline for synthetic data generation"""
    
    def __init__(
        self,
        config_path: str = "configs/default_config.yaml",
        catalog_path: str = "configs/sku_catalog.yaml",
        blender_path: str = "blender"
    ):
        """Initialize pipeline"""
        self.config = Config.from_yaml(config_path)
        self.catalog = SKUCatalog.from_yaml(catalog_path)
        self.catalog_path = catalog_path
        self.blender_path = blender_path
        
        # Initialize renderer
        self.renderer = SimpleRenderer(
            resolution=tuple(self.config.rendering.resolution)
        )
        
        # Initialize components
        self.rule_engine = RuleEngine(self.config, self.catalog)
        self.qc = QualityController(
            min_products=self.config.qc.min_products_per_image,
            max_occlusion_rate=self.config.qc.max_occlusion_rate,
            check_bbox_validity=self.config.qc.check_bbox_validity,
            check_empty_images=self.config.qc.check_empty_images
        )
        self.monitor = DistributionMonitor()
        self.manifest_gen = ManifestGenerator()
        
        # Setup output directories
        self._setup_output_dirs()
        
    def _setup_output_dirs(self):
        """Create output directory structure"""
        paths = self.config.get_output_paths()
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # Also create temp directory for recipes
        self.recipe_dir = Path("dataset/temp/recipes")
        self.recipe_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_dataset(self, num_scenes: Optional[int] = None):
        """Generate complete dataset"""
        if num_scenes is None:
            num_scenes = self.config.scene.num_scenes
        
        print(f"Starting synthetic data generation: {num_scenes} scenes")
        print("=" * 60)
        
        successful_scenes = 0
        scene_idx = 0
        attempts = 0
        max_total_attempts = num_scenes * 3  # Limit total attempts
        
        while successful_scenes < num_scenes and attempts < max_total_attempts:
            attempts += 1
            
            try:
                success = self._generate_single_scene(scene_idx)
                
                if success:
                    successful_scenes += 1
                    scene_idx += 1
                    
                    # Generate distribution report periodically
                    if successful_scenes % self.config.distribution.report_frequency == 0:
                        print("\n" + self.monitor.generate_report())
                
            except Exception as e:
                print(f"Error generating scene {scene_idx}: {e}")
                scene_idx += 1
                continue
        
        # Generate final report and manifest
        print("\n" + self.monitor.generate_report())
        self._save_dataset_manifest()
        
        print(f"\nDataset generation complete!")
        print(f"Successful scenes: {successful_scenes}/{num_scenes}")
        print(f"Total attempts: {attempts}")
    
    def _generate_single_scene(self, scene_idx: int) -> bool:
        """Generate a single scene with QC"""
        print(f"\nGenerating scene {scene_idx}...", end=" ")
        
        # Generate scene recipe
        recipe = self.rule_engine.generate_scene_recipe(scene_idx)
        
        # Save recipe to temp file
        recipe_path = self.recipe_dir / f"{recipe.scene_id}.json"
        with open(recipe_path, 'w') as f:
            f.write(recipe.to_json())
        
        # Render scene using simple renderer
        print("rendering...", end=" ")
        render_success, rgb_path, instance_path, depth_path = self._render_scene(recipe, recipe_path)
        
        if not render_success:
            print("FAILED (render)")
            return False
        
        # Generate annotations from instance mask
        print("annotating...", end=" ")
        annotations = self._generate_annotations_from_mask(recipe, instance_path, depth_path)
        
        # Export annotations to Pascal VOC format
        self._export_annotations(recipe, annotations, rgb_path)
        
        # Run QC
        print("QC...", end=" ")
        qc_report = self.qc.validate_scene(
            recipe.scene_id,
            annotations,
            tuple(self.config.rendering.resolution)
        )
        
        # Update monitoring
        self.monitor.update(annotations, qc_report.passed)
        
        # Add to manifest
        recipe_hash = calculate_hash(recipe.to_dict())
        paths = self.config.get_output_paths()
        
        self.manifest_gen.add_image(
            scene_id=recipe.scene_id,
            image_path=str(paths['images'] / f"{recipe.scene_id}_rgb.png"),
            annotation_path=str(paths['annotations'] / f"{recipe.scene_id}.xml"),
            seed=recipe.seed,
            recipe_hash=recipe_hash,
            num_products=qc_report.num_products,
            num_valid_annotations=qc_report.num_valid_annotations,
            qc_passed=qc_report.passed,
            qc_issues=qc_report.issues,
            metrics=qc_report.metrics
        )
        
        if qc_report.passed:
            print("PASSED")
            return True
        else:
            print(f"FAILED ({', '.join(qc_report.issues[:2])})")
            if self.config.qc.resample_on_failure:
                return False  # Will retry with different scene
            return True  # Count as processed even if failed
    
    def _render_scene(self, recipe: SceneRecipe, recipe_path: Path) -> tuple:
        """Render scene using simple 2D renderer"""
        try:
            # Load catalog for rendering
            with open(self.catalog_path, 'r') as f:
                catalog_dict = yaml.safe_load(f)
            
            paths = self.config.get_output_paths()
            rgb_path, instance_path, depth_path = self.renderer.render_scene(
                recipe,
                paths['images'],
                catalog_dict
            )
            return True, rgb_path, instance_path, depth_path
        except Exception as e:
            print(f"Render error: {e}")
            return False, None, None, None
    
    def _generate_annotations_from_mask(
        self, 
        recipe: SceneRecipe, 
        instance_path: str,
        depth_path: str
    ) -> list:
        """Generate annotations from instance mask and depth map"""
        import cv2
        from src.annotations.annotation_generator import AnnotationGenerator
        
        # Load images
        instance_mask = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)
        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        
        # Create SKU mapping (instance_id -> sku_id)
        sku_mapping = {}
        category_mapping = {}
        instance_id = 1
        
        for shelf in recipe.shelves:
            for slot in shelf.slots:
                for product in slot.products:
                    sku_mapping[instance_id] = product.sku_id
                    
                    # Get category
                    try:
                        sku = self.catalog.get_sku_by_id(product.sku_id)
                        category_mapping[product.sku_id] = sku.category
                    except ValueError:
                        category_mapping[product.sku_id] = "unknown"
                    
                    instance_id += 1
        
        # Generate annotations
        ann_gen = AnnotationGenerator(
            min_bbox_area=self.config.annotation.min_bbox_area,
            polygon_tolerance=self.config.annotation.polygon_simplification_tolerance,
            occlusion_threshold=self.config.placement.occlusion_threshold
        )
        
        annotations = ann_gen.generate_annotations(
            instance_mask,
            depth_map,
            sku_mapping,
            category_mapping
        )
        
        return annotations
    
    def _export_annotations(
        self, 
        recipe: SceneRecipe, 
        annotations: list,
        rgb_path: str
    ):
        """Export annotations to Pascal VOC XML format"""
        from src.annotations.annotation_generator import AnnotationGenerator
        import cv2
        
        # Load image to get dimensions
        image = cv2.imread(rgb_path)
        if image is None:
            return
        
        height, width, channels = image.shape
        
        # Export to Pascal VOC
        ann_gen = AnnotationGenerator()
        paths = self.config.get_output_paths()
        annotation_path = paths['annotations'] / f"{recipe.scene_id}.xml"
        
        ann_gen.export_pascal_voc(
            annotations,
            rgb_path,
            (width, height, channels),
            str(annotation_path)
        )
    
    def _save_dataset_manifest(self):
        """Save dataset manifest"""
        stats = self.monitor.get_stats()
        config_hash = calculate_hash(self.config.__dict__)
        
        paths = self.config.get_output_paths()
        manifest_path = paths['manifests'] / "dataset_manifest.json"
        
        self.manifest_gen.save_manifest(
            str(manifest_path),
            dataset_id="synthetic_retail_v1",
            config_hash=config_hash,
            distribution_stats=stats.to_dict()
        )
        
        print(f"\nManifest saved to: {manifest_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Synthetic Retail Data Factory - Generate synthetic retail shelf images"
    )
    parser.add_argument(
        "--config",
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--catalog",
        default="configs/sku_catalog.yaml",
        help="Path to SKU catalog file"
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        help="Number of scenes to generate (overrides config)"
    )
    parser.add_argument(
        "--blender-path",
        default="blender",
        help="Path to Blender executable"
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = SyntheticDataPipeline(
        config_path=args.config,
        catalog_path=args.catalog,
        blender_path=args.blender_path
    )
    
    pipeline.generate_dataset(num_scenes=args.num_scenes)


if __name__ == "__main__":
    main()
