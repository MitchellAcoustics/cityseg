"""
Semantic Segmentation Pipeline

This package provides a flexible and efficient semantic segmentation pipeline
for processing images and videos. It supports multiple segmentation models
and datasets, with capabilities for tiling large images, mixed-precision
processing, and comprehensive result analysis.

Main components:
- Config: Configuration class for the pipeline
- create_segmentation_model: Factory function for creating segmentation models
- create_processor: Factory function for creating image/video processors
- analyze_hdf5_segmaps_with_stats: Function for analyzing saved segmentation maps
"""

__version__ = "0.2.0"

from .config import Config
from .pipeline import SegmentationPipeline
from .processors import SegmentationProcessor, DirectoryProcessor, create_processor
from .exceptions import ConfigurationError, InputError, ModelError, ProcessingError

__all__ = [
    "Config",
    "SegmentationPipeline",
    "SegmentationProcessor",
    "DirectoryProcessor",
    "create_processor",
    "ConfigurationError",
    "InputError",
    "ModelError",
    "ProcessingError",
]
