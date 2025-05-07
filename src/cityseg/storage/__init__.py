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

# Try to import Lance storage
try:
    from .lance_storage import LanceSegmentationStorage

    __all__ = [
        "FileHandler",
        "LanceSegmentationStorage",
        "ParquetAnalysisStorage",
        "SegmentationStorage",
        "StorageFactory",
        "ZarrSegmentationStorage",
    ]
except ImportError:
    # Lance not available
    __all__ = [
        "FileHandler",
        "ParquetAnalysisStorage",
        "SegmentationStorage",
        "StorageFactory",
        "ZarrSegmentationStorage",
    ]
