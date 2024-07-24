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

__version__ = "0.1.0"

from .config import Config
from .models import create_segmentation_model
from .processors import create_processor
from .utils import analyze_hdf5_segmaps_with_stats

__all__ = [
    "Config",
    "create_segmentation_model",
    "create_processor",
    "analyze_hdf5_segmaps_with_stats",
]
