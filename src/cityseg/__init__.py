"""
Semantic Segmentation Pipeline

This package provides a flexible and efficient semantic segmentation pipeline
for processing images and videos. It supports multiple segmentation models
and datasets.

Main components:
- Config: Configuration class for the pipeline
- SegmentationPipeline: Core pipeline for semantic segmentation
- SegmentationProcessor: Processor for individual images and videos
- DirectoryProcessor: Processor for handling multiple videos in a directory
- create_processor: Factory function for creating image/video processors
- Exceptions: Custom exception classes for error handling

The package also includes utility functions for segmentation map analysis,
visualization, and logging.

For detailed usage instructions, please refer to the package documentation.
"""

__version__ = "0.3.1rc0"

# Core xarray-based functionality
from .components import (
    SegmentationDataset,
    create_segmentation_dataset,
    load_segmentation_dataset,
    save_segmentation_dataset,
)

# Segmentation functions
from .segmentation import (
    load_segmentation_model,
    segment_image,
    segment_video,
    segment_from_path,
)

# Essential exceptions
from .exceptions import ConfigurationError, InputError, ModelError, ProcessingError

__all__ = [
    # Core Dataset API
    "SegmentationDataset",
    "create_segmentation_dataset",
    "load_segmentation_dataset",
    "save_segmentation_dataset",
    # Segmentation Functions
    "load_segmentation_model",
    "segment_image",
    "segment_video",
    "segment_from_path",
    # Exceptions
    "ConfigurationError",
    "InputError",
    "ModelError",
    "ProcessingError",
]
