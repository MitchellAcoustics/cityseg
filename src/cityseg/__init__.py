"""
Semantic Segmentation Pipeline

This package provides a flexible and efficient semantic segmentation pipeline
for processing images and videos. It supports multiple segmentation models
and datasets.

Main components:
- Config: Configuration class for the pipeline
- SegmentationPipeline: Core pipeline for semantic segmentation
- VideoResource: Resource manager for video operations
- Storage adapters: Efficient storage of segmentation data using Zarr and Parquet
- Workflow: Hamilton-based workflow for processing with proper caching
- Exceptions: Custom exception classes for error handling

The package also includes utility functions for segmentation map analysis,
visualization, and logging.

For detailed usage instructions, please refer to the package documentation.
"""

__version__ = "0.3.1rc0"

from . import palettes
from .config import Config
from .exceptions import ConfigurationError, InputError, ModelError, ProcessingError
from .file_handler import FileHandler
from .pipeline import SegmentationPipeline, create_segmentation_pipeline
from .processing_plan import ProcessingPlan
from .processors import DirectoryProcessorLegacy as DirectoryProcessor, SegmentationProcessorWrapper as SegmentationProcessor, create_processor
from .segmentation_analyzer import SegmentationAnalyzer
from .storage_adapter import (
    ZarrSegmentationStorage,
    ParquetAnalysisStorage,
    StorageFactory,
)
from .utils import setup_logging
from .video_file_iterator import VideoFileIterator
from .video_resource import VideoResource
from .visualization_handler import VisualizationHandler
from .workflow import CitysegWorkflow, create_workflow

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
    "VideoResource",
    "ZarrSegmentationStorage",
    "ParquetAnalysisStorage",
    "StorageFactory",
    "CitysegWorkflow",
    "create_workflow",
]
