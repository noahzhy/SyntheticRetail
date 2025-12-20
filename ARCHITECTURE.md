# SyntheticRetail Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      SyntheticRetail Pipeline                   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Configuration Layer (src/config.py)                            │
│  ├── Config Management (YAML → Python dataclasses)              │
│  ├── SKU Catalog Loading                                        │
│  └── Parameter Validation                                       │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Rule Engine (src/rules/scene_generator.py)                     │
│  ├── Deterministic Seed Generation                              │
│  ├── Shelf Layout Planning                                      │
│  ├── Slot-based Product Placement                               │
│  ├── Category Grouping                                          │
│  ├── Camera Parameter Generation                                │
│  └── Scene Recipe Export (JSON)                                 │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Blender Rendering (src/blender_scripts/blender_renderer.py)   │
│  ├── Scene Setup from Recipe                                    │
│  ├── Shelf & Slot Geometry Creation                             │
│  ├── Product Instance Placement                                 │
│  ├── Multi-Channel Rendering:                                   │
│  │   ├── RGB Image                                              │
│  │   ├── Instance ID Mask                                       │
│  │   └── Depth Map                                              │
│  └── Export to Organized Structure                              │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Annotation Generation (src/annotations/annotation_generator.py)│
│  ├── Instance Mask Processing                                   │
│  ├── Bounding Box Calculation                                   │
│  ├── Polygon Contour Extraction                                 │
│  ├── Occlusion Ratio Computation                                │
│  └── Pascal VOC XML Export                                      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Quality Control (src/utils/quality_control.py)                 │
│  ├── Product Count Validation                                   │
│  ├── Occlusion Rate Checking                                    │
│  ├── Bounding Box Validation                                    │
│  ├── Empty Image Detection                                      │
│  └── Resample Decision Making                                   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Distribution Monitoring (src/utils/quality_control.py)         │
│  ├── SKU Distribution Tracking                                  │
│  ├── Category Balance Monitoring                                │
│  ├── Occlusion Statistics                                       │
│  └── Report Generation                                          │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Manifest System (src/utils/manifest.py)                        │
│  ├── Image-Level Manifests                                      │
│  ├── Dataset-Level Manifest                                     │
│  ├── Seed & Hash Recording                                      │
│  ├── QC Metrics Storage                                         │
│  └── Complete Audit Trail                                       │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Output Dataset                                                  │
│  ├── dataset/images/          (RGB renders)                     │
│  ├── dataset/annotations/     (Pascal VOC XML)                  │
│  └── dataset/manifests/       (Traceability data)               │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Configuration Phase
```
configs/default_config.yaml ─┐
                             ├─→ Config Object
configs/sku_catalog.yaml ────┘
```

### 2. Scene Generation Phase
```
Scene Index + Seed Base ─→ Deterministic Seed ─→ Scene Recipe
                                                   ├── Shelves
                                                   ├── Slots
                                                   ├── Products
                                                   └── Camera
```

### 3. Rendering Phase
```
Scene Recipe (JSON) ─→ Blender Script ─→ Multi-Channel Output
                                          ├── RGB Image
                                          ├── Instance Mask
                                          └── Depth Map
```

### 4. Annotation Phase
```
Instance Mask + Depth Map ─→ Annotation Generator ─→ Pascal VOC XML
                                                      ├── Bounding Boxes
                                                      ├── Polygons
                                                      └── Occlusion Ratios
```

### 5. Quality Control Phase
```
Annotations + Image ─→ QC Validator ─→ Pass/Fail + Metrics
                                       └─→ Resample if Failed
```

### 6. Monitoring Phase
```
QC Results ─→ Distribution Monitor ─→ Statistics + Reports
```

## Key Design Principles

### 1. Reproducibility
- **Deterministic Seeds**: Every scene generated from seed = hash(base_seed + scene_idx)
- **Recipe Storage**: Complete scene specification saved as JSON
- **Version Control**: Config hashes tracked in manifests

### 2. Modularity
- **Separation of Concerns**: Each module handles one responsibility
- **Plugin Architecture**: Easy to extend with custom rules/renderers
- **Configurable Pipeline**: All parameters externalized to YAML

### 3. Quality Assurance
- **Multi-Stage Validation**: Pre-render, post-render, and distribution checks
- **Automatic Resampling**: Failed scenes regenerated automatically
- **Comprehensive Metrics**: Track quality across entire dataset

### 4. Traceability
- **Complete Audit Trail**: Every image traced back to seed and config
- **Manifest System**: JSON manifests for programmatic access
- **Versioning**: Config and recipe hashes for reproducibility

### 5. Scalability
- **Batch Processing**: Generate scenes independently
- **Parallel Generation**: Multiple instances with different ranges
- **Incremental Progress**: Save manifests progressively

## Module Responsibilities

### config.py
- Load and validate YAML configuration
- Type-safe dataclass representations
- SKU catalog management
- Output path resolution

### rules/scene_generator.py
- Generate reproducible scene recipes
- Apply placement rules and constraints
- Category grouping and affinity
- Camera parameter generation
- Recipe serialization

### blender_scripts/blender_renderer.py
- Interface with Blender API (bpy)
- Create scene geometry
- Place product instances
- Setup rendering pipeline
- Export multi-channel outputs

### annotations/annotation_generator.py
- Process instance segmentation masks
- Calculate bounding boxes
- Extract polygon contours
- Compute occlusion ratios
- Export Pascal VOC XML

### utils/quality_control.py
- Validate scene quality
- Check product counts
- Monitor occlusion rates
- Track distribution statistics
- Generate QC reports

### utils/manifest.py
- Create image-level manifests
- Generate dataset manifest
- Calculate hashes for traceability
- Store QC metrics
- Manage audit trail

### pipeline.py
- Orchestrate entire workflow
- Coordinate all modules
- Handle error recovery
- Progress reporting
- Manifest aggregation

## Extension Points

### Custom Rules
Extend `RuleEngine` to implement custom placement logic:
```python
class CustomRuleEngine(RuleEngine):
    def _generate_slot_recipe(self, slot_idx, rng, category):
        # Custom logic
        pass
```

### Custom Annotations
Extend `AnnotationGenerator` for different formats:
```python
class COCOAnnotationGenerator(AnnotationGenerator):
    def export_coco(self, annotations, output_path):
        # COCO format export
        pass
```

### Custom QC
Extend `QualityController` with custom checks:
```python
class CustomQC(QualityController):
    def validate_scene(self, scene_id, annotations, image_size):
        report = super().validate_scene(scene_id, annotations, image_size)
        # Add custom checks
        return report
```

## Performance Considerations

### Bottlenecks
1. **Blender Rendering**: CPU/GPU intensive (minutes per scene)
2. **Annotation Processing**: Mask processing (seconds per scene)
3. **I/O Operations**: File writes (milliseconds per scene)

### Optimization Strategies
1. **Parallel Generation**: Multiple Blender instances
2. **GPU Rendering**: Cycles with CUDA/OptiX
3. **Batch Processing**: Generate in chunks
4. **Caching**: Reuse geometry when possible
5. **Progressive Saving**: Save manifests incrementally

## Future Enhancements

1. **Geometry Nodes System**: Production-ready slot templates
2. **Asset Library**: Real product models with materials
3. **Domain Randomization**: Lighting, textures, backgrounds
4. **Advanced Occlusion**: Physics-based product interaction
5. **Multi-Format Export**: COCO, YOLO, TFRecord
6. **Web Interface**: Dataset management dashboard
7. **Distributed Generation**: Cloud-based rendering
8. **Active Learning**: Feedback loop for hard examples
