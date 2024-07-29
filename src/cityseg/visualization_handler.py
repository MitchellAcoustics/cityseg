from typing import List, Optional, Union

import numpy as np
from loguru import logger


class VisualizationHandler:
    @staticmethod
    def visualize_segmentation(
        images: Union[np.ndarray, List[np.ndarray]],
        seg_maps: Union[np.ndarray, List[np.ndarray]],
        palette: Optional[np.ndarray] = None,
        colored_only: bool = False,
    ) -> Union[np.ndarray, List[np.ndarray]]:
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
        logger.debug(f"Generating palette for {num_colors} colors")
        return np.array(
            [
                [(i * 100) % 255, (i * 150) % 255, (i * 200) % 255]
                for i in range(num_colors)
            ],
            dtype=np.uint8,
        )
