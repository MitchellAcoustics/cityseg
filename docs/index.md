# CitySeg: Urban Semantic Segmentation Pipeline

Welcome to the documentation for CitySeg, a flexible and efficient pipeline for performing semantic segmentation on images and videos of urban environments.

## Features

- Support for multiple segmentation models (OneFormer)
- Compatible with various datasets (Cityscapes, ADE20k, Mapillary Vistas)
- Flexible image resizing for processing high-resolution inputs
- Comprehensive analysis of segmentation results
- Support for both image and video inputs
- Multi-video processing capability for entire directories
- Caching of processed segmentation maps in HDF5 format for quick re-analysis
- Output includes segmentation maps, colored segmentations, overlay visualizations, and detailed CSV reports

## Quick Start

```python
import cityseg as cs

# Load configuration
config = cs.Config.from_yaml("config.yaml")

# Create processor
processor = cs.create_processor(config)

# Process input
processor.process()
```

For more detailed information on how to use CitySeg, check out our [Getting Started](getting_started.md) guide.

## Project Structure

CitySeg is organized into several Python modules:

- `config.py`: Configuration classes for the pipeline
- `pipeline.py`: Core segmentation pipeline implementation
- `processors.py`: Classes for processing images, videos, and directories
- `utils.py`: Utility functions for analysis, file operations, and logging
- `exceptions.py`: Custom exception classes for error handling

For detailed API documentation, visit our [API Reference](api/config.md) section.