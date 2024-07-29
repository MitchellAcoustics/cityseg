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

The project is organized into several Python modules, each serving a specific purpose within the CitySeg pipeline:

- `main.py`: Entry point of the application, responsible for initializing and running the segmentation pipeline.
- `config.py`: Defines configuration classes and handles loading and validating configuration settings.
- `pipeline.py`: Implements the core segmentation pipeline, including model loading and inference.
- `processors.py`: Contains classes for processing images, videos, and directories, managing the segmentation workflow.
- `segmentation_analyzer.py`: Provides functionality for analyzing segmentation results, including computing statistics and generating reports.
- `video_file_iterator.py`: Implements an iterator for efficiently processing multiple video files in a directory.
- `visualization_handler.py`: Handles the visualization of segmentation results using color palettes.
- `file_handler.py`: Manages file operations related to saving and loading segmentation data and metadata.
- `utils.py`: Provides utility functions for various tasks, including data handling and logging.
- `palettes.py`: Defines color palettes for different datasets used in segmentation.
- `exceptions.py`: Custom exception classes for error handling throughout the pipeline.


For detailed API documentation, visit our [API Reference](api/config.md) section.