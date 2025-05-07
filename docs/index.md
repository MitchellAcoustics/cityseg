# CitySeg: Urban Semantic Segmentation Pipeline

> **Version 0.4.0** introduces a complete reorganization to a modular component-based architecture, with enhanced testing infrastructure and improved storage mechanisms. The legacy interface remains available for backward compatibility. See the [Changelog](changelog.md) for details.

CitySeg is a flexible and efficient pipeline for performing semantic segmentation on images and videos of urban environments, designed to handle both small-scale analyses and large dataset processing.

## Key Features

### Model & Dataset Support
- **Multiple Models**: SegFormer, OneFormer, Mask2Former models
- **Various Datasets**: Compatible with Cityscapes, ADE20k, Mapillary Vistas
- **Flexible Processing**: Automatic resolution adjustment for high-res inputs

### Processing Capabilities
- **Multiple Input Types**: Process single images, videos, or entire directories
- **Batch Processing**: Efficient handling of large video collections
- **Resumable Processing**: Skip already processed files for interrupted workflows

### Analysis & Storage
- **Comprehensive Analysis**: Pixel-level category distribution and statistics
- **Advanced Storage**: Zarr format for segmentation maps and Parquet for analytics
- **Visualization Options**: Segmentation overlays, heatmaps, and category highlighting

## Architecture Overview

CitySeg uses a component-based architecture that separates concerns into logical modules while providing simple high-level interfaces:

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Input       │    │  Processing    │    │   Output      │
│  Components   │───▶│  Components   │───▶│  Components   │
└───────────────┘    └───────────────┘    └───────────────┘
       ▲                     ▲                    ▲
       │                     │                    │
       └─────────────┬───────┴──────────┬────────┘
                     │                  │
             ┌───────────────┐  ┌──────────────┐
             │ Configuration │  │ Workflow     │
             │ Management    │  │ Management   │
             └───────────────┘  └──────────────┘
```

## Quick Start

### Legacy Interface (Simplest)
```python
import cityseg as cs

# Load configuration
config = cs.Config.from_yaml("config.yaml")

# Create and run processor with one line
cs.create_processor(config).process()
```

### Component-based Interface (Flexible)
```python
import cityseg as cs
from cityseg.components import ImageProcessor, SegmentationProcessor
from cityseg.analysis import VisualizationHandler

# Load configuration and create components
config = cs.Config.from_yaml("config.yaml")

# Process an image with fine-grained control
image = ImageProcessor.load_image(config.input)
resized = ImageProcessor.resize_image(image, config.model.max_size)
segmentation_pipeline = SegmentationProcessor.create_pipeline(config.model)
result = SegmentationProcessor.process_image(resized, segmentation_pipeline)

# Create visualization
visualization = VisualizationHandler.visualize_segmentation(
    image, result["seg_map"], result.get("palette"), alpha=0.6
)
```

Explore the [Getting Started](getting_started.md) guide for detailed setup instructions and more examples.

## Project Structure

The project's modular architecture separates functionality into logical components:

| Module | Description | Key Components |
|--------|-------------|----------------|
| `core/` | Core functionality | Configuration, exceptions |
| `components/` | Processing components | Image, video, segmentation, pipeline |
| `analysis/` | Result analysis | Analytics, visualization |
| `storage/` | Data persistence | Zarr and Parquet adapters |
| `utils/` | Common utilities | Logging, color palettes |
| `workflow/` | Process management | Hamilton framework integration |
| `legacy/` | Compatibility layer | Legacy processor interfaces |

For detailed API documentation, see the [API Reference](api/core/config.md) section.