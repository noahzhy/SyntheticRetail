# 3D Rendering Upgrade

## Overview
Upgraded the synthetic data factory from 2D rectangle visualization to full 3D scene rendering with perspective projection.

## User Request
@noahzhy requested: "使用3d场景渲染可视化货架场景" (Use 3D scene rendering to visualize shelf scenes)

## Implementation

### New 3D Renderer (`src/utils/renderer_3d.py`)

**Key Features:**
- Full 3D visualization using matplotlib's `Axes3D`
- Products rendered as 3D boxes with volume and depth
- Shelves rendered as 3D surfaces with brown wooden texture
- Vertical support bars for realistic shelf structure
- Perspective projection with configurable camera angles
- SKU labels displayed on product fronts
- Category-based color coding with variations

**Technical Details:**
- Uses `Poly3DCollection` for 3D polygon rendering
- Camera positioning via `view_init(elev, azim)` from scene recipes
- Generates instance masks and depth maps from 3D coordinates
- Renders to numpy array at 1920x1080 resolution
- Non-interactive backend for server-side rendering

### 3D Geometry

**Shelf Structure:**
- Main shelf surface (1.2m × 0.3m)
- Vertical support bars on both sides
- Brown color (#8B4513) for wooden appearance
- 3D depth for realistic appearance

**Product Boxes:**
- 6 faces rendered as 3D polygons
- Dimensions: 0.05m × 0.04m × 0.12m (scaled by product.scale)
- Color based on category with random variations
- Black edge lines for definition
- Proper Z-positioning on shelf surface

**Camera System:**
- Elevation angle from recipe.camera_rotation[0]
- Azimuth angle from recipe.camera_rotation[2]
- Configurable view distance and FOV
- Perspective projection for depth perception

## Results

### Performance Comparison

**2D Renderer (previous):**
- File size: 3.6-4.0 MB per RGB image
- Rendering: ~5 seconds per scene
- Appearance: Flat colored rectangles
- Total dataset (5 scenes): ~20 MB

**3D Renderer (new):**
- File size: 44-122 KB per RGB image
- Rendering: ~12 seconds per scene
- Appearance: 3D boxes with perspective
- Total dataset (5 scenes): 644 KB

**Improvement:**
- 97% reduction in file size
- More realistic 3D visualization
- True depth information
- Better spatial understanding

### Test Results

Generated 5 complete scenes:
```
Successful scenes: 5/5
Total attempts: 5
Total Products: 255

SKU Distribution:
  SKU001-SKU006: Balanced 14-21% each

Category Distribution:
  beverages: 30.2%
  dairy: 40.0%
  snacks: 29.8%

Occlusion:
  94.5% products with <20% occlusion
  5.5% products with 20-40% occlusion
```

## Visual Comparison

### 2D Renderer Output
- Products as flat rectangles
- No depth perception
- Orthographic view
- Large file sizes (PNG compression limited by noise)

### 3D Renderer Output
- Products as 3D boxes with faces
- Clear depth and perspective
- Camera angle adjustable
- Small file sizes (clean geometric shapes compress well)
- SKU labels visible
- Shelf structure visible

## Usage

No changes required to existing commands:

```bash
# Same usage as before
python main.py --num-scenes 10

# Output in dataset/images/ with 3D rendering
ls dataset/images/
# scene_000000_rgb.png    (44-122KB, 3D visualization)
# scene_000000_instance.png
# scene_000000_depth.png
```

## Technical Architecture

```python
# 3D Rendering Pipeline
1. Create matplotlib figure with 3D axis
2. Set camera view (elevation, azimuth)
3. For each shelf:
   a. Render shelf surface as 3D polygon
   b. Add support bars
   c. For each product:
      - Create 3D box (6 faces)
      - Apply category color
      - Add SKU label
4. Render to numpy array
5. Generate instance mask (2D projection)
6. Generate depth map (Z-coordinates)
7. Save all channels
```

## Benefits

1. **More Realistic**: Actual 3D visualization with perspective
2. **Better Depth**: True 3D coordinates for depth calculation
3. **Smaller Files**: 97% reduction due to clean geometric rendering
4. **Production Ready**: Can swap with Blender for photorealism
5. **Same Pipeline**: No changes to annotation or QC systems
6. **Flexible**: Easy to adjust camera angles and positions

## Future Enhancements

The 3D renderer provides a solid foundation for:
- Adding textures to products
- Implementing lighting and shadows
- Adding background elements
- Domain randomization
- Easy upgrade path to Blender 3D rendering

## Files Changed

- `src/utils/renderer_3d.py` - New 3D renderer (400 lines)
- `src/pipeline.py` - Import Renderer3D instead of SimpleRenderer
- `requirements.txt` - Added matplotlib>=3.5.0

## Commit

**commit:** 5673465
**message:** "Upgrade to 3D scene rendering with matplotlib"
