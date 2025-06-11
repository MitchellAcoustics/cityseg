"""CitySeg Components Module.

Core xarray-based media and segmentation dataset functionality.
"""

from .dataset import (
    SegmentationDataset,
)
from .media import (
    MediaDataset,
    load_image,
    load_video,
    load_media_dataset,
    save_media_dataset,
)
from .accessor import SegmentationAccessor  # This registers the .seg accessor

__all__ = [
    "SegmentationDataset",
    "MediaDataset",
    "load_image",
    "load_video",
    "load_media_dataset",
    "save_media_dataset",
    "SegmentationAccessor",
]
