"""
This module provides a class for visualizing segmentation results using color palettes.

It includes methods to visualize segmentation maps with color palettes and options for
displaying colored or blended results.
"""

from __future__ import annotations
import numpy as np
from loguru import logger

from ..utils import get_palette


class VisualizationHandler:
    """
    A class for visualizing segmentation results using color palettes.

    This class provides methods to visualize segmentation maps with color palettes
    and options for displaying colored or blended results.

    Methods:
        visualize_segmentation: Visualizes segmentation results with color palettes.
        _generate_palette: Generates a color palette for visualization.
    """

    @staticmethod
    def visualize_segmentation(
        images: np.ndarray | list[np.ndarray],
        seg_maps: np.ndarray | list[np.ndarray],
        palette: np.ndarray | None = None,
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
        logger.debug(
            f"Visualizing segmentation for {len(images) if isinstance(images, list) else 1} images"
        )
        if palette is None:
            palette = VisualizationHandler._generate_palette(256)
        if isinstance(palette, list):
            palette = np.array(palette, dtype=np.uint8)

        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
            seg_maps = [seg_maps]

        results = []
        for image, seg_map in zip(images, seg_maps):
            color_seg = palette[seg_map]

            if colored_only:
                results.append(color_seg)
            else:
                # Ensure image is uint8 for consistent blending
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)

                # Blend with configurable alpha
                img = image * (1 - alpha) + color_seg * alpha
                results.append(img.astype(np.uint8))

        return results[0] if len(results) == 1 else results

    @staticmethod
    def _generate_palette(num_colors: int) -> np.ndarray:
        """
        Generates a color palette for visualization.

        Args:
            num_colors: Number of colors to generate in the palette.

        Returns:
            np.ndarray: Color palette array for visualization, with shape (num_colors, 3).
        """
        # Get a suitable palette (this is now in utils.common)
        palette = get_palette("default")

        if num_colors <= len(palette):
            logger.debug(f"Using existing palette with {num_colors} colors")
            return np.array(palette[:num_colors], dtype=np.uint8)
        else:
            logger.debug(f"Generating custom palette for {num_colors} colors")
            # Generate evenly distributed colors in HSV space for better visual separation
            # Then convert to RGB
            from colorsys import hsv_to_rgb

            palette = []
            for i in range(num_colors):
                h = i / num_colors
                s = 0.8
                v = 0.9
                r, g, b = hsv_to_rgb(h, s, v)
                palette.append([int(r * 255), int(g * 255), int(b * 255)])

            return np.array(palette, dtype=np.uint8)
