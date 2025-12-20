# Blender Integration Guide

## Overview

The SyntheticRetail data factory now supports **photorealistic rendering using Blender** as the primary renderer, with automatic fallback to matplotlib 3D rendering if Blender is not available.

## Features

### Blender Renderer
- **Photorealistic rendering** with Cycles engine
- **GPU acceleration** (CUDA/OptiX/HIP if available)
- **Multi-channel output**: RGB, instance segmentation, depth maps
- **Realistic materials**: Wood shelves, metal supports, product materials
- **Advanced lighting**: Sun and area lights for professional quality
- **Denoising**: Built-in Cycles denoiser for clean images

### Automatic Fallback
- Detects Blender availability automatically
- Falls back to matplotlib 3D renderer if Blender not found
- Shows installation instructions when Blender is missing
- No configuration changes needed

## Installation

### Ubuntu/Debian Linux
```bash
sudo apt update
sudo apt install blender
```

### macOS
1. Download from: https://www.blender.org/download/
2. Install to `/Applications`
3. Run with: `--blender-path /Applications/Blender.app/Contents/MacOS/Blender`

### Windows
1. Download from: https://www.blender.org/download/
2. Install (default: `C:\Program Files\Blender Foundation\Blender`)
3. Run with: `--blender-path "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"`

### Verify Installation
```bash
blender --version
```

## Usage

### With Blender (Default)
```bash
# Will use Blender if available
python main.py --num-scenes 100
```

### Without Blender (Matplotlib Fallback)
```bash
# Force matplotlib renderer
python main.py --num-scenes 100 --no-blender
```

### Specify Blender Path
```bash
# Use custom Blender installation
python main.py --num-scenes 100 --blender-path /custom/path/to/blender
```

## Renderer Comparison

| Feature | Blender | Matplotlib 3D |
|---------|---------|---------------|
| **Realism** | Photorealistic | Geometric visualization |
| **Materials** | Full PBR materials | Flat colors |
| **Lighting** | Realistic lighting | Basic shading |
| **Shadows** | Real-time ray-traced | None |
| **Textures** | Full support | Color only |
| **GPU Support** | Yes (CUDA/OptiX) | No |
| **Speed** | 30-60 sec/scene | 12 sec/scene |
| **File Size** | 200-500 KB | 44-122 KB |
| **Quality** | Production-grade | Preview quality |

## Blender Rendering Details

### Scene Setup
1. **Geometry**: Realistic shelves with wood material and metal supports
2. **Products**: 3D boxes with PBR materials based on category
3. **Camera**: Positioned according to scene recipe
4. **Lighting**: Sun light + area lights for professional look
5. **Environment**: Light gray background simulating store wall

### Materials
- **Shelves**: Brown wood with proper roughness and specularity
- **Supports**: Dark metallic with high metallic value
- **Products**: Category-based colors with realistic surface properties
- **Labels**: SKU numbers (can be enhanced with textures)

### Rendering Pipeline
```
Scene Recipe (JSON)
    ↓
Blender Scene Setup
    ↓
Cycles Rendering
    ├─→ RGB Pass (beauty)
    ├─→ Instance ID Pass (segmentation)
    └─→ Depth Pass (Z-buffer)
    ↓
Post-Processing
    ↓
Annotations (Pascal VOC XML)
```

### Performance Tips

**GPU Rendering:**
- Automatically enabled if GPU is detected
- Supports CUDA (NVIDIA), OptiX (RTX), HIP (AMD)
- 3-5x faster than CPU rendering

**Optimize Settings:**
Edit `src/blender_scripts/blender_renderer.py`:
```python
scene.cycles.samples = 64  # Lower for faster preview (default: 128)
scene.cycles.samples = 256 # Higher for production quality
```

**Batch Processing:**
```bash
# Generate in parallel with multiple processes
python main.py --num-scenes 250 &
python main.py --num-scenes 250 --seed-offset 250 &
python main.py --num-scenes 250 --seed-offset 500 &
python main.py --num-scenes 250 --seed-offset 750 &
```

## Output

### With Blender
```
dataset/images/
├── scene_000000_rgb.png       # Photorealistic RGB (300-500KB)
├── scene_000000_instance.png  # Instance segmentation
└── scene_000000_depth.png     # Depth map
```

### With Matplotlib
```
dataset/images/
├── scene_000000_rgb.png       # 3D geometric visualization (44-122KB)
├── scene_000000_instance.png  # Instance segmentation
└── scene_000000_depth.png     # Depth map
```

## Troubleshooting

### Blender Not Found
```
Error: Blender not found. Please install Blender or specify path.
```
**Solution**: Install Blender or use `--blender-path` to specify location

### Rendering Timeout
```
Error: Rendering timeout (>5 minutes)
```
**Solution**: Reduce samples in blender_renderer.py or use more powerful hardware

### GPU Not Detected
```
Using CPU rendering
```
**Solution**: Update GPU drivers and Blender to latest version

### Out of Memory
```
Error: CUDA out of memory
```
**Solution**: Reduce resolution or samples, or use CPU rendering

## Customization

### Add Textures
Edit `src/blender_scripts/blender_renderer.py`:
```python
def _create_product_material(self, color_rgb, sku_id):
    # Load texture from SKU catalog
    texture_path = self.get_sku_texture(sku_id)
    
    # Add image texture node
    node_tex = nodes.new('ShaderNodeTexImage')
    node_tex.image = bpy.data.images.load(texture_path)
    
    # Link to BSDF
    mat.node_tree.links.new(node_tex.outputs[0], node_bsdf.inputs['Base Color'])
```

### Custom Lighting
```python
def _setup_lighting(self):
    # Add custom HDRI environment
    world = bpy.context.scene.world
    node_env = world.node_tree.nodes.new('ShaderNodeTexEnvironment')
    node_env.image = bpy.data.images.load('path/to/hdri.hdr')
    
    # Connect to world output
    world.node_tree.links.new(node_env.outputs[0], node_background.inputs[0])
```

### Use Real Product Models
```python
def _place_products(self):
    for product in products:
        # Load .blend asset
        with bpy.data.libraries.load(product['blend_file']) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects]
        
        # Link to scene
        for obj in data_to.objects:
            bpy.context.collection.objects.link(obj)
            obj.location = product['position']
```

## Future Enhancements

- [ ] Real 3D product models from asset library
- [ ] Geometry Nodes slot system for automatic placement
- [ ] HDRI environment maps for realistic lighting
- [ ] Domain randomization (lighting, camera, backgrounds)
- [ ] Advanced materials with textures and labels
- [ ] Multiple camera angles per scene
- [ ] Animation for video generation

## Best Practices

1. **Start with matplotlib** for rapid prototyping and testing
2. **Switch to Blender** when ready for production dataset
3. **Use GPU rendering** for faster generation
4. **Batch process** for large datasets
5. **Monitor quality** - check samples regularly
6. **Version control** - track config and Blender version

## Commit

**commit:** [To be added]  
**message:** "Add Blender integration with automatic fallback"

Complete Blender integration is now available, providing photorealistic rendering while maintaining compatibility with systems without Blender!
