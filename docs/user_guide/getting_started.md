# Getting Started with CitySeg

This guide will help you set up CitySeg and run your first semantic segmentation tasks using either the simple legacy interface or the more flexible component-based architecture.

## Installation

### Using pip

The recommended way to install CitySeg is using pip:

```bash
pip install cityseg
```

### Development Installation

For the latest features or development work:

```bash
# Clone the repository
git clone https://github.com/MitchellAcoustics/cityseg.git
cd cityseg

# Install in development mode
pip install -e .
```

### System Requirements

- Python 3.10 or higher
- PyTorch 2.0+ (automatically installed as a dependency)
- For GPU acceleration: CUDA-compatible GPU (for PyTorch) or Apple Silicon (for MPS)

## Usage Options

CitySeg provides two interfaces to accommodate different needs:

### 1. Legacy Interface (Simple)

Ideal for quick processing tasks with minimal code:

```python
import cityseg as cs

# Create configuration
config = cs.Config(
    input="path/to/image.jpg",                           # Input image/video/directory
    output_dir="path/to/output",                         # Output location
    model=cs.ModelConfig(                                # Model configuration
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        max_size=1200                                   # Max dimension for processing
    )
)

# Alternative: load from YAML
# config = cs.Config.from_yaml("config.yaml")

# Process input with one line
cs.create_processor(config).process()
```

### 2. Component-based Architecture (Flexible)

For more control over each processing step:

```python
import cityseg as cs
from cityseg.components import ImageProcessor, SegmentationProcessor
from cityseg.analysis import VisualizationHandler
from cityseg.storage import StorageAdapter

# Load configuration
config = cs.Config.from_yaml("config.yaml")

# Process with explicit steps
image = ImageProcessor.load_image(config.input)                      # Load
resized = ImageProcessor.resize_image(image, config.model.max_size)  # Resize

# Create segmentation pipeline and process
seg_pipeline = SegmentationProcessor.create_pipeline(config.model)
result = SegmentationProcessor.process_image(resized, seg_pipeline) 

# Generate visualizations
visualization = VisualizationHandler.visualize_segmentation(
    image, result["seg_map"], result.get("palette"),
    alpha=config.visualization.alpha
)

# Save results
storage = StorageAdapter()
output_path = storage.save_segmentation_data(
    result["seg_map"], result.get("metadata"), config.output_path
)
```

### 3. Hamilton Workflow (Advanced)

For automated dependency tracking and caching:

```python
import cityseg as cs
from cityseg.workflow import process

# Configuration with complex settings
config = cs.Config.from_yaml("config.yaml")

# Process with automated workflow
results = process(config)

# Results contain all pipeline outputs
print(f"Processed {config.input_type.name}")
print(f"Available results: {list(results.keys())}")
```

## Configuration

CitySeg uses a flexible YAML configuration system. Here's a comprehensive example:

```yaml
# Input and output settings
input: "videos/street_scene.mp4"
output_dir: "results"
output_prefix: "cityseg_analysis"

# Model configuration
model:
  name: "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
  model_type: null         # Auto-detect model type
  max_size: 1280           # Maximum dimension for processing
  device: "auto"          # "cuda", "cpu", "mps", or "auto"
  num_workers: 4           # For parallel processing

# Processing options
frame_step: 10             # Process every 10th frame for videos
batch_size: 8              # Number of frames to process in each batch

# Output options
save_raw_segmentation: true
save_colored_segmentation: true
save_overlay: true
analyze_results: true

# Visualization settings
visualization:
  alpha: 0.7               # Overlay opacity (0.0-1.0)
  colormap: "default"      # Colormap for segmentation

# Advanced options
force_reprocess: false     # Skip already processed files
disable_tqdm: false        # Progress bars
```

See the [Configuration Guide](user_guide/configuration.md) for detailed options.

## Common Use Cases

### Processing a Single Image

```python
import cityseg as cs

# Create a basic configuration
config = cs.Config(
    input="path/to/image.jpg",
    output_dir="results",
    model=cs.ModelConfig(
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024", 
        device="auto"
    )
)

# Process the image
cs.create_processor(config).process()
```

### Processing a Video File

```python
import cityseg as cs

# Create configuration for video
config = cs.Config(
    input="path/to/video.mp4",
    output_dir="results",
    model=cs.ModelConfig(
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    ),
    frame_step=5,  # Process every 5th frame
    save_overlay=True,  # Create overlay visualization
)

# Process the video
cs.create_processor(config).process()
```

### Processing Multiple Videos

```python
import cityseg as cs

# Create configuration for a directory of videos
config = cs.Config(
    input="path/to/video_directory",
    output_dir="results",
    model=cs.ModelConfig(
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    ),
    frame_step=10,
)

# Process all videos in the directory
cs.create_processor(config).process()
```

## Next Steps

- Explore the [Component Demo](demos/component_demo.ipynb) for hands-on examples
- Learn about [Configuration Options](user_guide/configuration.md) for advanced settings
- Dive into the [API Reference](api/core/config.md) for detailed documentation