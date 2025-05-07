"""
Components for the CitySeg package.

This module provides specialized components for the CitySeg package,
each with a clear, single responsibility.
"""

from .dataset import DatasetBuilder
from .image import ImageProcessor
from .pipeline import SegmentationPipeline, create_segmentation_pipeline
from .segmentation import SegmentationProcessor
from .video import VideoFileIterator, VideoProcessor, VideoResource

__all__ = [
    "DatasetBuilder",
    "ImageProcessor",
    "SegmentationPipeline",
    "create_segmentation_pipeline",
    "SegmentationProcessor",
    "VideoFileIterator",
    "VideoProcessor",
    "VideoResource",
]
