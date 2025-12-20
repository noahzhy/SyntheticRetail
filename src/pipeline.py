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
        self.blender_path = blender_path
        
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
        
        # Render scene (this would call Blender in production)
        # For now, we'll simulate the rendering
        print("rendering...", end=" ")
        render_success = self._render_scene_mock(recipe, recipe_path)
        
        if not render_success:
            print("FAILED (render)")
            return False
        
        # Generate annotations (mock)
        print("annotating...", end=" ")
        annotations = self._generate_annotations_mock(recipe)
        
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
    
    def _render_scene_mock(self, recipe: SceneRecipe, recipe_path: Path) -> bool:
        """Mock rendering - in production this calls Blender"""
        # In production, this would be:
        # subprocess.run([
        #     self.blender_path,
        #     "--background",
        #     "--python", "src/blender_scripts/blender_renderer.py",
        #     "--",
        #     "--recipe", str(recipe_path),
        #     "--output", str(self.config.get_output_paths()['base'])
        # ])
        
        # For now, just simulate success
        return True
    
    def _generate_annotations_mock(self, recipe: SceneRecipe) -> list:
        """Mock annotation generation"""
        # In production, this would use the actual AnnotationGenerator
        # with real instance masks and depth maps
        
        from src.annotations.annotation_generator import Annotation, BoundingBox
        import random
        
        annotations = []
        rng = random.Random(recipe.seed)
        
        res_w, res_h = self.config.rendering.resolution
        
        for shelf in recipe.shelves:
            for slot in shelf.slots:
                for product in slot.products:
                    # Create mock annotation with bounds checking
                    w = rng.randint(50, 150)
                    h = rng.randint(80, 200)
                    x = rng.randint(100, max(101, res_w - w - 100))
                    y = rng.randint(100, max(101, res_h - h - 100))
                    
                    # Ensure bbox is within image bounds
                    xmax = min(x + w, res_w - 1)
                    ymax = min(y + h, res_h - 1)
                    
                    bbox = BoundingBox(x, y, xmax, ymax)
                    
                    try:
                        sku = self.catalog.get_sku_by_id(product.sku_id)
                        category = sku.category
                    except ValueError:
                        category = "unknown"
                    
                    annotation = Annotation(
                        sku_id=product.sku_id,
                        category=category,
                        bbox=bbox,
                        polygon=[(x, y), (xmax, y), (xmax, ymax), (x, ymax)],
                        occlusion_ratio=rng.uniform(0.0, 0.5),
                        truncated=False,
                        difficult=False
                    )
                    annotations.append(annotation)
        
        return annotations
    
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
