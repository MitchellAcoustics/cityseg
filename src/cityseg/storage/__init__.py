"""
Storage components for the CitySeg package.

This module provides adapters and handlers for segmentation data storage and retrieval.
"""

from .storage import (
    FileHandler,
    ParquetAnalysisStorage,
    SegmentationStorage,
    StorageFactory,
    ZarrSegmentationStorage,
)

__all__ = [
    "FileHandler",
    "ParquetAnalysisStorage",
    "SegmentationStorage",
    "StorageFactory",
    "ZarrSegmentationStorage",
]
