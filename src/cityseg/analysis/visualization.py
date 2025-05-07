"""
This module provides a class for visualizing segmentation results using color palettes.

It includes methods to visualize segmentation maps with color palettes and options for
displaying colored or blended results.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence
from loguru import logger

from ..utils import get_palette


class Palette(np.ndarray):
    """
    A specialized numpy array for color palettes used in segmentation visualization.

    This class extends ndarray with shape (n_colors, 3) and dtype=np.uint8 to represent
    an RGB color palette. It provides convenient methods for creating and manipulating
    color palettes.
    """

    def __new__(cls, input_array):
        """Create a new Palette instance from input array."""
        # Convert input to proper format if needed
        if isinstance(input_array, list):
            # Convert list of RGB tuples/lists to ndarray
            arr = np.array(input_array, dtype=np.uint8)
        else:
            # Ensure proper dtype
            arr = np.asarray(input_array, dtype=np.uint8)

        # Validate shape
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Palette must have shape (n_colors, 3), got {arr.shape}")

        # Create the array, view as our subclass
        obj = np.asarray(arr).view(cls)
        return obj

    def __array_finalize__(self, obj):
        """Finalize array creation for slicing/view operations."""
        if obj is None:
            return

    @classmethod
    def from_list(cls, colors: Sequence[tuple[int, int, int]]) -> Palette:
        """Create a palette from a list of RGB tuples."""
        return cls(np.array(colors, dtype=np.uint8))

    @classmethod
    def from_segmentation_metadata(cls, metadata: dict) -> Palette | None:
        """
        Create a palette from segmentation metadata if available.

        Returns None if no palette is found in the metadata.
        """
        palette_data = metadata.get("palette")
        if palette_data is None:
            return None
        return cls(palette_data)

    @classmethod
    def get_default(cls, num_colors: int = 256) -> Palette:
        """Get the default palette with the specified number of colors."""
        palette_list = get_palette("default")

        if num_colors <= len(palette_list):
            return cls(palette_list[:num_colors])
        else:
            # Generate more colors using HSV space
            from colorsys import hsv_to_rgb

            colors = []
            # Keep existing colors
            colors.extend(palette_list)

            # Generate additional colors as needed
            for i in range(len(palette_list), num_colors):
                h = i / num_colors
                s = 0.8
                v = 0.9
                r, g, b = hsv_to_rgb(h, s, v)
                colors.append((int(r * 255), int(g * 255), int(b * 255)))

            return cls(colors)


class VisualizationHandler:
    """
    A class for visualizing segmentation results using color palettes.

    This class provides methods to visualize segmentation maps with color palettes
    and options for displaying colored or blended results.

    Methods:
        visualize_segmentation: Visualizes segmentation results with color palettes.
    """

    @staticmethod
    def visualize_segmentation(
        images: np.ndarray | list[np.ndarray],
        seg_maps: np.ndarray | list[np.ndarray],
        palette: Palette | list[tuple[int, int, int]] | None = None,
        colored_only: bool = False,
        alpha: float = 0.5,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Visualizes segmentation results using color palettes.

        Args:
            images: Input images or a list of images.
            seg_maps: Segmentation maps or a list of maps.
            palette: Color palette for visualization. If None, a default palette is generated.
            colored_only: Flag to indicate if only colored results are desired (True) or blended with the original images (False).
            alpha: Alpha value for blending segmentation with original image (0.0-1.0).

        Returns:
            np.ndarray | list[np.ndarray]: Visualized segmentation results, either as a single array or a list of arrays.
        """
        # Normalize inputs to lists for consistent handling
        single_image = False
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
            single_image = True

        if isinstance(seg_maps, np.ndarray) and seg_maps.ndim in (
            2,
            3,
        ):  # Handle both 2D single and 3D batch segmaps
            seg_maps = [seg_maps]

        if len(images) != len(seg_maps):
            raise ValueError(
                f"Number of images ({len(images)}) must match number of segmentation maps ({len(seg_maps)})"
            )

        logger.debug(f"Visualizing segmentation for {len(images)} images")

        # Ensure we have a proper Palette object
        if palette is None:
            palette_obj = Palette.get_default()
        elif isinstance(palette, Palette):
            palette_obj = palette
        else:
            # Convert list of RGB tuples to Palette
            palette_obj = Palette(palette)

        results = []
        for image, seg_map in zip(images, seg_maps):
            # Ensure segmentation map is properly formatted for indexing
            if not isinstance(seg_map, np.ndarray):
                seg_map = np.array(seg_map)

            # Convert segmentation map indices to colors
            color_seg = np.take(palette_obj, seg_map.astype(np.int32), axis=0)

            if colored_only:
                results.append(color_seg)
            else:
                # Ensure image is uint8 for consistent blending
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)

                # Blend with configurable alpha
                img = image * (1 - alpha) + color_seg * alpha
                results.append(img.astype(np.uint8))

        # Return single array for single input, list otherwise
        return results[0] if single_image else results
