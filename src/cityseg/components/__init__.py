"""CitySeg Components Module.

Core xarray-based segmentation dataset functionality.
"""

from .dataset import (
    SegmentationDataset,
    create_segmentation_dataset,
    load_segmentation_dataset,
    save_segmentation_dataset,
)
from .accessor import SegmentationAccessor  # This registers the .seg accessor

__all__ = [
    "SegmentationDataset",
    "create_segmentation_dataset",
    "load_segmentation_dataset",
    "save_segmentation_dataset",
    "SegmentationAccessor",
]
