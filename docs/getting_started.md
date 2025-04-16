# Getting Started with CitySeg

This guide will help you set up and run your first semantic segmentation task using CitySeg.

## Installation

Install CitySeg using pip:

```bash
pip install cityseg
```

## Basic Usage

CitySeg offers two ways to use the library: the legacy interface for backward compatibility and the new component-based API for more flexibility.

### Legacy Interface

The simplest way to use CitySeg with the legacy interface:

```python
import cityseg as cs

# Load configuration from a YAML file
config = cs.Config.from_yaml("path/to/your/config.yaml")

# Create processor (legacy interface)
processor = cs.create_processor(config)

# Process input
processor.process()
```

### Component-based API

For more control, you can use the component-based API:

```python
import cityseg as cs

# Load configuration
config = cs.Config.from_yaml("path/to/your/config.yaml")

# For complete pipeline with caching using Hamilton
result = cs.process(config)

# Or use components directly for more control
from cityseg.components import ImageProcessor, SegmentationProcessor
from cityseg.analysis import VisualizationHandler

# Load and process an image
image = ImageProcessor.load_image(config.input)
resized_image = ImageProcessor.resize_image(image, config.model.max_size)

# Create the segmentation pipeline
segmentation_pipeline = SegmentationProcessor.create_pipeline(config.model)

# Process the image
result = SegmentationProcessor.process_image(resized_image, segmentation_pipeline)

# Visualize the result
visualization = VisualizationHandler.visualize_segmentation(
    image, result["seg_map"], result.get("palette")
)
```

## Configuration

CitySeg uses a YAML configuration file to set up the segmentation pipeline. Here's a basic example:

```yaml
input: path/to/your/input/file_or_directory
output_prefix: path/to/your/output/directory/output
model:
  name: shi-labs/oneformer_cityscapes_swin_large
  max_size: 1920  # Set to null to maintain original resolution
  device: cuda  # or cpu or mps
frame_step: 5  # For video processing, process every 5th frame
save_raw_segmentation: true
save_colored_segmentation: true
save_overlay: true
visualization:
  alpha: 0.5
  colormap: default
```

For more detailed information on configuration options, see the [Configuration](user_guide/configuration.md) section in the User Guide.

## Next Steps

- Learn about [Image Processing](user_guide/image_processing.md)
- Explore [Video Processing](user_guide/video_processing.md)
- Check out the [Examples](examples/single_image_processing.ipynb) for more advanced usage