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
    MediaDataset,
    SegmentationDataset,
    load_image,
    load_video,
    load_media_dataset,
    save_media_dataset,
)

# Segmentation functions
from .segmentation import (
    apply_segmentation,
    load_segmentation_model,
    segment_image,
    segment_video,
    segment_from_path,
    segment_image_file,
    segment_video_file,
)

# Essential exceptions
from .exceptions import ConfigurationError, InputError, ModelError, ProcessingError

__all__ = [
    # Core Dataset API
    "MediaDataset",
    "SegmentationDataset",
    # Media Loading API
    "load_image",
    "load_video",
    "load_media_dataset",
    "save_media_dataset",
    # Segmentation Functions
    "apply_segmentation",
    "load_segmentation_model",
    "segment_image",
    "segment_video",
    "segment_from_path",
    "segment_image_file",
    "segment_video_file",
    # Exceptions
    "ConfigurationError",
    "InputError",
    "ModelError",
    "ProcessingError",
]
