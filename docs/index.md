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

The project is organized into logical modules with a component-based architecture:

- `main.py`: Entry point of the application, responsible for initializing and running the segmentation pipeline.
- `core/`: Core functionality and configuration:
  - `config.py`: Configuration classes and validation
  - `exceptions.py`: Custom exception classes for error handling
- `components/`: Modular components implementing core functionality:
  - `image.py`: Image processing operations
  - `video.py`: Video handling and frame extraction
  - `segmentation.py`: Segmentation model integration
  - `dataset.py`: Dataset handling and management
  - `pipeline.py`: Pipeline creation and coordination
- `analysis/`: Analysis and visualization:
  - `analyzer.py`: Segmentation analysis and metrics
  - `visualization.py`: Visualization of segmentation results
- `storage/`: Data storage and retrieval:
  - `storage.py`: Zarr and Parquet storage adapters
- `utils/`: Utility functions:
  - `common.py`: Common utility functions
  - `palettes.py`: Color palettes for different datasets
- `workflow/`: Hamilton-based workflow engine:
  - `hamilton.py`: Workflow definitions and execution
- `legacy/`: Backwards compatibility adapters:
  - `processors.py`: Legacy processor interface


For detailed API documentation, visit our [API Reference](api/config.md) section.