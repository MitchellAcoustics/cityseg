"""
CitySeg: Urban Semantic Segmentation Pipeline

CitySeg is a flexible and efficient semantic segmentation pipeline for
processing images and videos of urban environments. It supports multiple
segmentation models and datasets, with capabilities for processing high-resolution
inputs and comprehensive result analysis.
"""

__version__ = "0.3.1rc0"

# Core functionality
from .core import (
    Config,
    InputType,
    ModelConfig,
    ConfigurationError,
    InputError,
    ModelError,
    ProcessingError,
)

# Primary APIs
from .workflow import process
from .legacy.processors import create_processor

# Components for advanced users
from .components import (
    DatasetBuilder,
    ImageProcessor,
    SegmentationProcessor,
    VideoProcessor,
)

# Analysis tools
from .analysis import SegmentationAnalyzer, VisualizationHandler

# Storage utilities
from .storage import StorageFactory

__all__ = [
    # Core functionality
    "Config",
    "InputType",
    "ModelConfig",
    "ConfigurationError",
    "InputError",
    "ModelError",
    "ProcessingError",
    # Primary APIs
    "process",
    "create_processor",
    # Components
    "DatasetBuilder",
    "ImageProcessor",
    "SegmentationProcessor",
    "VideoProcessor",
    # Analysis
    "SegmentationAnalyzer",
    "VisualizationHandler",
    # Storage
    "StorageFactory",
]
