"""
This module provides a class for visualizing segmentation results using color palettes.

It includes methods to visualize segmentation maps with color palettes and options for
displaying colored or blended results.

Classes:
    VisualizationHandler: A class for visualizing segmentation results using color palettes.
"""

from typing import List, Optional, Union

import numpy as np
from loguru import logger


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
        images: Union[np.ndarray, List[np.ndarray]],
        seg_maps: Union[np.ndarray, List[np.ndarray]],
        palette: Optional[np.ndarray] = None,
        colored_only: bool = False,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Visualizes segmentation results using color palettes.

        This method takes input images and their corresponding segmentation maps,
        applies the specified color palette, and returns the visualized results.

        Args:
            images (Union[np.ndarray, List[np.ndarray]]): Input images or a list of images.
            seg_maps (Union[np.ndarray, List[np.ndarray]]): Segmentation maps or a list of maps.
            palette (Optional[np.ndarray]): Color palette for visualization. If None, a default palette is generated.
            colored_only (bool): Flag to indicate if only colored results are desired (True) or blended with the original images (False).

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Visualized segmentation results, either as a single array or a list of arrays.
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
                img = image * 0.5 + color_seg * 0.5
                results.append(img.astype(np.uint8))

        return results[0] if len(results) == 1 else results

    @staticmethod
    def _generate_palette(num_colors: int) -> np.ndarray:
        """
        Generates a color palette for visualization.

        This method creates a color palette with a specified number of colors,
        which can be used to visualize segmentation results.

        Args:
            num_colors (int): Number of colors to generate in the palette.

        Returns:
            np.ndarray: Color palette array for visualization, with shape (num_colors, 3).
        """
        logger.debug(f"Generating palette for {num_colors} colors")
        return np.array(
            [
                [(i * 100) % 255, (i * 150) % 255, (i * 200) % 255]
                for i in range(num_colors)
            ],
            dtype=np.uint8,
        )
