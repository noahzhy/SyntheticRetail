# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/noahzhy/SyntheticRetail.git
cd SyntheticRetail

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### Generate a Small Test Dataset

```bash
# Generate 10 scenes with default configuration
python main.py --num-scenes 10
```

### Check the Output

After generation, you'll find:

```
dataset/
├── images/           # RGB images (when using real Blender)
├── annotations/      # Pascal VOC XML annotations
└── manifests/        # Dataset manifest with traceability
    └── dataset_manifest.json
```

### View the Manifest

```bash
cat dataset/manifests/dataset_manifest.json
```

This shows:
- Total scenes and products generated
- SKU and category distribution
- QC metrics and failed scenes
- Complete traceability data

## Configuration

### Customize Scene Generation

Edit `configs/default_config.yaml`:

```yaml
scene:
  num_scenes: 1000          # Total scenes to generate
  shelves_per_scene: [3, 5] # Min/max shelves
  slots_per_shelf: [8, 12]  # Min/max slots per shelf

rendering:
  resolution: [1920, 1080]  # Image resolution
  samples: 128              # Render quality

qc:
  min_products_per_image: 5 # Minimum products for valid scene
  max_occlusion_rate: 0.8   # Maximum allowed occlusion
```

### Add Products

Edit `configs/sku_catalog.yaml`:

```yaml
skus:
  - id: "SKU007"
    name: "New Product"
    category: "snacks"
    subcategory: "candy"
    blend_file: "assets/snacks/new_product.blend"
    object_name: "NewProduct"
    dimensions: [0.10, 0.05, 0.15]  # width, depth, height (meters)
    weight: 1.0  # Sampling weight
```

## Integration with Blender

### Current Status

The current implementation uses **mock rendering** for demonstration. To integrate with real Blender:

1. Install Blender 3.0+
2. Update the `_render_scene_mock` method in `src/pipeline.py` to call Blender:

```python
def _render_scene(self, recipe, recipe_path):
    """Real Blender rendering"""
    result = subprocess.run([
        self.blender_path,
        "--background",
        "--python", "src/blender_scripts/blender_renderer.py",
        "--",
        "--recipe", str(recipe_path),
        "--output", str(self.config.get_output_paths()['base'])
    ])
    return result.returncode == 0
```

3. Create product assets as .blend files with proper materials
4. Set up the shelf geometry with Geometry Nodes slot system

### Blender Script

The `src/blender_scripts/blender_renderer.py` script:
- Loads scene recipes (JSON)
- Creates shelf structure with slots
- Places products according to recipe
- Renders multiple channels (RGB, instance ID, depth)
- Exports to organized directory structure

## Quality Control

The system automatically:

1. **Validates scenes** - Checks product count, occlusion, bbox validity
2. **Resamples failures** - Regenerates scenes that fail QC
3. **Tracks distribution** - Monitors SKU/category balance
4. **Generates reports** - Creates audit trails and statistics

### Distribution Report

Generated every 100 scenes (configurable):

```
============================================================
DISTRIBUTION REPORT
============================================================
Total Scenes: 100
Failed Scenes: 5
Total Products: 5,234

SKU Distribution:
  SKU001: 892 (17.0%)
  SKU002: 864 (16.5%)
  ...

Category Distribution:
  beverages: 1,567 (29.9%)
  snacks: 1,889 (36.1%)
  dairy: 1,778 (34.0%)

Occlusion Distribution:
  0.0-0.2: 2,245 (42.9%)
  0.2-0.4: 1,987 (38.0%)
  0.4-0.6: 892 (17.0%)
  0.6-0.8: 110 (2.1%)
  0.8-1.0: 0 (0.0%)
============================================================
```

## Advanced Features

### Reproducible Generation

Every scene has a deterministic seed:

```python
# Generate scene 42 - always produces the same result
recipe = rule_engine.generate_scene_recipe(42)
```

### Scene Recipes

Recipes are saved as JSON for debugging and reproduction:

```bash
# View a scene recipe
cat dataset/temp/recipes/scene_000042.json
```

### Custom Rules

Extend the rule engine:

```python
from src.rules.scene_generator import RuleEngine

class MyRuleEngine(RuleEngine):
    def _generate_slot_recipe(self, slot_idx, rng, category):
        # Add custom placement logic
        # E.g., always place larger items at the back
        pass
```

## Troubleshooting

### Import Errors

```bash
# Ensure you're in the project root
cd SyntheticRetail
python main.py
```

### Missing Dependencies

```bash
pip install -r requirements.txt
```

### Blender Not Found

```bash
# Specify Blender path explicitly
python main.py --blender-path /path/to/blender
```

## Performance Tips

1. **Parallel Generation** - Run multiple instances with different scene ranges
2. **GPU Rendering** - Configure Blender to use GPU (Cycles with CUDA/OptiX)
3. **Lower Samples** - Reduce `rendering.samples` for faster iteration
4. **Batch Processing** - Generate in batches and merge manifests

## Next Steps

1. Create product assets in Blender
2. Set up Geometry Nodes slot system
3. Run test renders with Blender
4. Generate full dataset
5. Train your model on the synthetic data

## Support

For issues or questions:
- Open an issue on GitHub
- Check the main README.md
- Review the source code documentation
