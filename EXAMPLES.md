# Example: Generating a Custom Dataset

This example shows how to customize the synthetic data factory for your specific needs.

## Scenario: Beverage-Only Dataset

Let's create a dataset focused only on beverages with specific requirements:

### 1. Create Custom Configuration

Create `configs/beverages_config.yaml`:

```yaml
# Inherit from default, override specific values
dataset:
  output_dir: "./dataset_beverages"
  images_dir: "images"
  annotations_dir: "annotations"
  manifests_dir: "manifests"

scene:
  seed_base: 2024
  num_scenes: 500
  shelves_per_scene: [2, 4]  # Fewer shelves
  slots_per_shelf: [10, 15]  # More slots

placement:
  max_products_per_slot: 4   # Allow more bottles per slot
  min_products_per_slot: 1
  occlusion_threshold: 0.3
  allow_empty_slots: false   # No empty slots
  category_grouping: false   # Mix beverage types

camera:
  height: [1.4, 1.6]         # Eye-level view
  distance: [2.0, 2.5]       # Closer shots
  angle_horizontal: [-20, 20]
  angle_vertical: [-10, 10]
  fov: 45                    # Narrower field of view

qc:
  min_products_per_image: 15 # More products required
  max_occlusion_rate: 0.6    # Allow moderate occlusion
```

### 2. Create Beverage-Only Catalog

Create `configs/beverages_catalog.yaml`:

```yaml
skus:
  # Soft Drinks
  - id: "BEV001"
    name: "Cola 500ml"
    category: "soft_drinks"
    subcategory: "cola"
    blend_file: "assets/beverages/cola_500ml.blend"
    object_name: "Cola_500"
    dimensions: [0.065, 0.065, 0.22]
    weight: 1.5  # More frequent

  - id: "BEV002"
    name: "Cola 1.5L"
    category: "soft_drinks"
    subcategory: "cola"
    blend_file: "assets/beverages/cola_1500ml.blend"
    object_name: "Cola_1500"
    dimensions: [0.095, 0.095, 0.32]
    weight: 1.0

  # Water
  - id: "BEV003"
    name: "Water 500ml"
    category: "water"
    subcategory: "still"
    blend_file: "assets/beverages/water_500ml.blend"
    object_name: "Water_500"
    dimensions: [0.060, 0.060, 0.20]
    weight: 2.0  # Very frequent

  # Juice
  - id: "BEV004"
    name: "Orange Juice 1L"
    category: "juice"
    subcategory: "orange"
    blend_file: "assets/beverages/juice_orange_1l.blend"
    object_name: "Juice_Orange"
    dimensions: [0.080, 0.080, 0.24]
    weight: 1.0

  # Energy Drinks
  - id: "BEV005"
    name: "Energy Drink 250ml"
    category: "energy"
    subcategory: "caffeinated"
    blend_file: "assets/beverages/energy_250ml.blend"
    object_name: "Energy_250"
    dimensions: [0.055, 0.055, 0.18]
    weight: 0.8

category_affinity:
  soft_drinks: ["soft_drinks", "energy"]
  water: ["water"]
  juice: ["juice"]

constraints:
  heavy_items: ["BEV002", "BEV004"]  # Large bottles on lower shelves
  premium_items: ["BEV005"]          # Energy drinks
```

### 3. Generate the Dataset

```bash
python main.py \
  --config configs/beverages_config.yaml \
  --catalog configs/beverages_catalog.yaml \
  --num-scenes 500
```

### 4. Verify Output

```bash
# Check distribution
cat dataset_beverages/manifests/dataset_manifest.json | jq '.distribution_stats'

# Expected output:
{
  "total_scenes": 500,
  "total_products": 12500,
  "sku_distribution": {
    "BEV001": 2800,
    "BEV002": 1850,
    "BEV003": 3700,
    "BEV004": 1850,
    "BEV005": 1300
  },
  "category_distribution": {
    "soft_drinks": 4650,
    "water": 3700,
    "juice": 1850,
    "energy": 1300
  }
}
```

## Scenario: High-Occlusion Training Data

Create challenging scenes with heavy occlusion for robust model training:

### Custom Configuration

```yaml
placement:
  max_products_per_slot: 5   # More products
  min_products_per_slot: 3   # Dense packing

camera:
  angle_horizontal: [-45, 45]  # Extreme angles
  angle_vertical: [-20, 20]

qc:
  min_products_per_image: 20   # Many products
  max_occlusion_rate: 0.9      # Allow high occlusion
```

## Scenario: Clean Product Shots

Create minimal-occlusion data for initial training:

```yaml
placement:
  max_products_per_slot: 1     # One product per slot
  min_products_per_slot: 1

camera:
  angle_horizontal: [-5, 5]    # Frontal views only
  angle_vertical: [-5, 5]

qc:
  max_occlusion_rate: 0.2      # Very low occlusion only
```

## Programmatic Generation

You can also control generation programmatically:

```python
from src.pipeline import SyntheticDataPipeline
from src.config import Config

# Load configuration
config = Config.from_yaml('configs/my_config.yaml')

# Modify at runtime
config.scene.num_scenes = 1000
config.rendering.samples = 256  # Higher quality

# Create pipeline
pipeline = SyntheticDataPipeline(
    config_path='configs/my_config.yaml',
    catalog_path='configs/my_catalog.yaml'
)

# Generate with custom logic
for batch in range(10):
    print(f"Generating batch {batch}")
    pipeline.generate_dataset(num_scenes=100)
    
    # Check distribution and adjust
    stats = pipeline.monitor.get_stats()
    if stats.failed_scenes > 10:
        # Adjust QC thresholds
        pipeline.qc.max_occlusion_rate = 0.9
```

## Batch Processing

Generate large datasets in parallel:

```bash
# Terminal 1 - Scenes 0-999
python main.py --num-scenes 1000 &

# Terminal 2 - Scenes 1000-1999
# (modify seed_base in config)
python main.py --config configs/config_batch2.yaml --num-scenes 1000 &

# Terminal 3 - Scenes 2000-2999
python main.py --config configs/config_batch3.yaml --num-scenes 1000 &
```

## Merging Datasets

Combine multiple batches:

```python
import json
from pathlib import Path

manifests = []
for batch_dir in Path('dataset_batches').glob('batch_*'):
    manifest_path = batch_dir / 'manifests' / 'dataset_manifest.json'
    with open(manifest_path) as f:
        manifests.append(json.load(f))

# Merge manifests
merged = {
    'dataset_id': 'merged_dataset',
    'total_scenes': sum(m['total_scenes'] for m in manifests),
    'total_products': sum(m['total_products'] for m in manifests),
    # ... merge other fields
}
```

## Integration with Training Pipeline

```python
from src.pipeline import SyntheticDataPipeline
import torch
from torch.utils.data import Dataset

class SyntheticRetailDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.images = self.manifest['images']
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_data = self.images[idx]
        # Load image and annotations
        image = load_image(image_data['image_path'])
        annotations = load_annotations(image_data['annotation_path'])
        return image, annotations

# Use in training
dataset = SyntheticRetailDataset('dataset/manifests/dataset_manifest.json')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
```

## Tips

1. **Start Small** - Test with 10-20 scenes first
2. **Iterate on Config** - Adjust parameters based on output quality
3. **Monitor Distribution** - Ensure balanced SKU representation
4. **Review QC Reports** - Check for systematic failures
5. **Use Seeds** - Reproduce specific scenes for debugging
