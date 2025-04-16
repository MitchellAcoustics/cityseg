# Configuration Guide

This comprehensive guide explains how to configure CitySeg for different processing needs. CitySeg's configuration system is designed to be flexible yet straightforward, allowing both simple default settings and fine-grained control.

## Configuration Methods

CitySeg offers multiple ways to define your configuration:

### 1. YAML File (Recommended)

The most flexible approach is to use a YAML configuration file:

```python
import cityseg as cs

# Load from YAML file
config = cs.Config.from_yaml("config.yaml")

# Process with loaded config
processor = cs.create_processor(config)
processor.process()
```

### 2. Direct Object Creation

You can also create configuration objects directly in Python:

```python
import cityseg as cs

# Create configuration programmatically
config = cs.Config(
    input="dataset/images/street1.jpg",
    output_dir="results",
    model=cs.ModelConfig(
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        max_size=1280,
        device="auto"  # Automatically select best available device
    ),
    visualization=cs.VisualizationConfig(
        alpha=0.7,
        colormap="default"
    )
)
```

### 3. Environment Variables

CitySeg also supports environment variable overrides for key settings:

```bash
# Set environment variables
export CITYSEG_MODEL="nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
export CITYSEG_DEVICE="cpu"  # Force CPU processing
```

Environment variables take precedence over file-based configuration.

## Complete Configuration Reference

Below is a comprehensive configuration file with all available options and detailed comments:

```yaml
#======================================#
#        Input Configuration           #
#======================================#
# Path to input (image, video, or directory)
input: "videos/traffic_scene.mp4"

# Output location and naming
output_dir: "results/traffic_analysis"             # Output directory
output_prefix: "traffic_segmentation"              # Prefix for output files

#======================================#
#        Model Configuration           #
#======================================#
model:
  # Model identifier (HuggingFace model ID or local path)
  name: "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
  
  # Model type - Options:
  # - "oneformer" (OneFormer models)
  # - "mask2former" (Mask2Former models)
  # - "segformer" (SegFormer models)
  # - null (auto-detect based on model name)
  model_type: null
  
  # Maximum dimension for image processing
  # - Integer value: resize to this maximum dimension while preserving aspect ratio
  # - null: use original resolution (warning: high memory usage)
  max_size: 1280
  
  # Processing device
  # - "cuda": NVIDIA GPU with CUDA
  # - "mps": Apple Silicon GPU
  # - "cpu": CPU processing (slower)
  # - "auto": Automatically select best available device
  device: "auto"
  
  # Number of worker processes for data loading
  # - 0: Disable multiprocessing
  # - n: Use n worker processes
  num_workers: 4

#======================================#
#      Processing Configuration        #
#======================================#
# For video inputs: process every nth frame
frame_step: 10

# Batch processing settings
batch_size: 8                                      # Frames per batch
output_fps: null                                   # Output video FPS (null = same as input)

# Processing control
force_reprocess: false                             # Reprocess even if output exists
disable_tqdm: false                               # Disable progress bars

#======================================#
#       Output Configuration          #
#======================================#
# Output types to generate
save_raw_segmentation: true                        # Save raw segmentation maps
save_colored_segmentation: true                    # Save colored segmentation video
save_overlay: true                                 # Save overlay on original video
analyze_results: true                              # Generate analysis data

#======================================#
#    Visualization Configuration       #
#======================================#
visualization:
  # Overlay transparency (0.0 = transparent, 1.0 = opaque)
  alpha: 0.6
  
  # Color scheme for segmentation visualization
  # - "default": Use model's default colors
  # - "cityscapes": Cityscapes dataset colors
  # - "ade20k": ADE20K dataset colors
  # - "mapillary": Mapillary Vistas dataset colors
  colormap: "default"
  
  # Style settings
  border_size: 1                                   # Contour width for boundaries
  text_size: 0.5                                   # Size of text annotations
```

## Configuration Sections Explained

### Input and Output

| Option | Type | Description |
|--------|------|-------------|
| `input` | `str` | Path to input image, video file, or directory containing videos |
| `output_dir` | `str` | Directory to save all output files |
| `output_prefix` | `str` | Prefix for output filenames. If omitted, a name is generated from the input filename and model |

### Model Configuration

| Option | Type | Description |
|--------|------|-------------|
| `model.name` | `str` | HuggingFace model identifier or path to local model |
| `model.model_type` | `str` | Type of model architecture (oneformer, mask2former, segformer, or null for auto-detection) |
| `model.max_size` | `int` | Maximum dimension for resizing input (preserves aspect ratio) |
| `model.device` | `str` | Processing device (cuda, mps, cpu, or auto for automatic selection) |
| `model.num_workers` | `int` | Number of worker processes for data loading |

### Processing Options

| Option | Type | Description |
|--------|------|-------------|
| `frame_step` | `int` | For video: process every nth frame (higher values = faster processing, less temporal resolution) |
| `batch_size` | `int` | Number of frames to process in each batch (higher values = faster, but more memory) |
| `output_fps` | `int` | FPS for output videos (null = same as input) |
| `force_reprocess` | `bool` | When true, reprocess even if output files already exist |
| `disable_tqdm` | `bool` | Disable progress bars during processing |

### Output Options

| Option | Type | Description |
|--------|------|-------------|
| `save_raw_segmentation` | `bool` | Save raw segmentation maps (class IDs) |
| `save_colored_segmentation` | `bool` | Save visualization with colors representing semantic classes |
| `save_overlay` | `bool` | Save original input with colored segmentation overlay |
| `analyze_results` | `bool` | Generate statistical analysis of segmentation results |

### Visualization Options

| Option | Type | Description |
|--------|------|-------------|
| `visualization.alpha` | `float` | Opacity of segmentation overlay (0.0-1.0) |
| `visualization.colormap` | `str` | Color scheme for visualizations |
| `visualization.border_size` | `int` | Width of contour lines for segment boundaries |
| `visualization.text_size` | `float` | Size of text annotations in visualizations |

## Example Configurations

### High Quality Image Processing

```yaml
input: "dataset/aerial_view.jpg"
output_dir: "results"
model:
  name: "facebook/mask2former-swin-large-ade-semantic"
  max_size: 2048  # Higher resolution for detailed processing
  device: "cuda"
visualization:
  alpha: 0.7
  colormap: "ade20k"
```

### Fast Video Processing

```yaml
input: "videos/street_cam/"
output_dir: "results/street_analysis"
model:
  name: "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"  # Smaller, faster model
  max_size: 640  # Lower resolution for speed
frame_step: 30  # Process every 30th frame (1 frame per second at 30fps video)
batch_size: 16  # Process more frames at once
save_raw_segmentation: false  # Only save visualizations
save_colored_segmentation: true
save_overlay: true
```

### Research Analysis with Complete Data

```yaml
input: "dataset/research_videos/" 
output_dir: "analysis_results"
model:
  name: "shi-labs/oneformer_cityscapes_swin_large"
  max_size: 1280
frame_step: 5  # More temporal detail
save_raw_segmentation: true
save_colored_segmentation: true
save_overlay: true
analyze_results: true  # Generate statistical analysis
visualization:
  alpha: 0.6
  colormap: "cityscapes"
```

For programmatic configuration details, see the [Config API Reference](../api/core/config.md).