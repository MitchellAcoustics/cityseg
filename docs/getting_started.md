# Getting Started with CitySeg

This guide will help you set up and run your first semantic segmentation task using CitySeg.

## Installation

Install CitySeg using pip:

```bash
pip install cityseg
```

## Basic Usage

Here's a simple example to get you started:

```python
import cityseg as cs

# Load configuration from a YAML file
config = cs.Config.from_yaml("path/to/your/config.yaml")

# Create processor
processor = cs.create_processor(config)

# Process input
processor.process()
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