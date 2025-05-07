"""
Utility functions for the CitySeg package.

This module provides common utility functions used throughout the package.
"""

from .common import (
    get_palette,
    get_segmentation_batch,
    setup_logging,
    tqdm_context,
    ADE20K_PALETTE_SUBSET,
    MAPILLARY_VISTAS_PALETTE_SUBSET,
)

from .palettes import (
    ADE20K_PALETTE,
    MAPILLARY_VISTAS_PALETTE,
)

# Import CITYSCAPES_PALETTE from palettes to avoid redefinition
from .palettes import CITYSCAPES_PALETTE

__all__ = [
    # Common utilities
    "get_palette",
    "get_segmentation_batch",
    "setup_logging",
    "tqdm_context",
    # Palettes
    "ADE20K_PALETTE",
    "ADE20K_PALETTE_SUBSET",
    "CITYSCAPES_PALETTE",
    "MAPILLARY_VISTAS_PALETTE",
    "MAPILLARY_VISTAS_PALETTE_SUBSET",
]
