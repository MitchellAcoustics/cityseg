"""
Semantic Segmentation Pipeline

This package provides a flexible and efficient semantic segmentation pipeline
for processing images and videos. It supports multiple segmentation models
and datasets, with capabilities for tiling large images, mixed-precision
processing, and comprehensive result analysis.

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

__version__ = "0.2.12"

from . import palettes
from .config import Config
from .exceptions import ConfigurationError, InputError, ModelError, ProcessingError
from .file_handler import FileHandler
from .pipeline import SegmentationPipeline, create_segmentation_pipeline
from .processing_plan import ProcessingPlan
from .processors import DirectoryProcessor, SegmentationProcessor, create_processor
from .segmentation_analyzer import SegmentationAnalyzer
from .utils import setup_logging
from .video_file_iterator import VideoFileIterator
from .visualization_handler import VisualizationHandler

__all__ = [
    "Config",
    "SegmentationPipeline",
    "create_segmentation_pipeline",
    "SegmentationProcessor",
    "SegmentationAnalyzer",
    "DirectoryProcessor",
    "create_processor",
    "ConfigurationError",
    "InputError",
    "ModelError",
    "ProcessingError",
    "setup_logging",
    "palettes",
    "FileHandler",
    "VisualizationHandler",
    "ProcessingPlan",
    "VideoFileIterator",
]
