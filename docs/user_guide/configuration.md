# Configuring CitySeg

This guide explains how to configure CitySeg for your specific use case.

## Configuration File

CitySeg uses a YAML configuration file to set up the segmentation pipeline. Here's a comprehensive example:

```yaml
# Input configuration
input: path/to/your/input/file_or_directory
output_dir: path/to/your/output/directory/output
output_prefix: "segmentation_result"

# Model configuration
model:
  name: "facebook/mask2former-swin-large-mapillary-vistas-semantic"
  model_type: "mask2former"  # Optional: can be 'oneformer', 'mask2former', or null for auto-detection
  max_size: null  # Optional: maximum size for input images/frames
  device: "mps"  # Options: "cuda", "cpu", "mps", or null for auto-detection

# Processing configuration
frame_step: 10  # Process every 5th frame
batch_size: 15  # Number of frames to process in each batch
output_fps: null  # Optional: FPS for output video (if different from input)

# Output options
save_raw_segmentation: true
save_colored_segmentation: true
save_overlay: true

# Visualization configuration
visualization:
  alpha: 0.5  # Opacity of the segmentation overlay
  colormap: "default"  # Colormap for segmentation visualization

# Advanced options
force_reprocess: false  # Set to true to reprocess even if output files exist
```

## Configuration Options

### Input and Output

- `input`: Path to the input image, video, or directory.
- `output_dir`: Directory to save output files.
- `output_prefix`: Path prefix for output files. Ignored when processing a directory of multiple videos. If not specified, a custom prefix is generated using `f"{input.name}_{model.name}_step{frame_step}"`.

### Model Configuration

- `model.name`: Name of the pre-trained model to use.
- `model.max_size`: Maximum size for input resizing.
- `model.device`: Device to use for processing (cuda, cpu, or mps).

### Processing Options

- `frame_step`: For video processing, process every nth frame.
- `save_raw_segmentation`: Whether to save raw segmentation maps.
- `save_colored_segmentation`: Whether to save colored segmentation video.
- `save_overlay`: Whether to save a video with a transparent overlay of the colored segmentation on top of the input video.

### Visualization

- `visualization.alpha`: Alpha value for blending segmentation with original image.
- `visualization.colormap`: Colormap to use for visualization.

For more detailed information on each configuration option, refer to the [Config API Reference](../api/config.md).