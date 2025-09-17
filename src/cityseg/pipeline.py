"""
This module provides a custom segmentation pipeline for image and video processing.

It extends the functionality of the Hugging Face Transformers library's
ImageSegmentationPipeline to support various segmentation models and
create detailed segmentation maps with associated metadata.
"""

from typing import Any

import numpy as np
import xarray as xr
from loguru import logger
from PIL import Image
from transformers import ImageSegmentationPipeline
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel

from .config import Config, ModelConfig
from .segmentation import load_segmentation_model


class SegmentationPipeline(ImageSegmentationPipeline):
    """
    A custom segmentation pipeline that extends ImageSegmentationPipeline.

    This class provides additional functionality for creating and processing
    segmentation maps, including support for different color palettes and
    batch processing of images.

    Attributes:
        palette (np.ndarray): The color palette used for visualization.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.palette = self._get_palette()

    def _get_palette(self) -> np.ndarray | None:
        """
        Get the color palette for the current model.

        Returns:
            np.ndarray | None: The color palette as a numpy array, or None if not available.
        """
        if hasattr(self.model.config, "palette"):
            return np.array(self.model.config.palette, dtype=np.uint8)
        elif "ade" in self.model.config._name_or_path:
            from .palettes import ADE20K_PALETTE

            return np.array(ADE20K_PALETTE, dtype=np.uint8)
        elif "mapillary-vistas" in self.model.config._name_or_path:
            from .palettes import MAPILLARY_VISTAS_PALETTE

            return np.array(MAPILLARY_VISTAS_PALETTE, dtype=np.uint8)
        elif "cityscapes" in self.model.config._name_or_path:
            from .palettes import CITYSCAPES_PALETTE

            return np.array(CITYSCAPES_PALETTE, dtype=np.uint8)
        else:
            return None

    def create_single_segmentation_map(
        self, annotations: list[dict[str, Any]], target_size: tuple
    ) -> dict[str, Any]:
        """
        Create a single segmentation map from annotations.

        Args:
            annotations (List[Dict[str, Any]]): List of annotation dictionaries.
            target_size (tuple): The target size of the segmentation map.

        Returns:
            Dict[str, Any]: A dictionary containing the segmentation map and associated metadata.
        """
        seg_map = np.zeros(target_size, dtype=np.int32)
        for annotation in annotations:
            mask = np.array(annotation["mask"])
            label_id = self.model.config.label2id[annotation["label"]]
            seg_map[mask != 0] = label_id

        return {
            "seg_map": seg_map,
            "label2id": self.model.config.label2id,
            "id2label": self.model.config.id2label,
            "palette": self.palette,
        }

    @staticmethod
    def _is_single_image_result(
        result: list[dict[str, Any]] | list[list[dict[str, Any]]],
    ) -> bool:
        """
        Determine if the result is for a single image or multiple images.

        Args:
            result (Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]): The result to check.

        Returns:
            bool: True if the result is for a single image, False otherwise.

        Raises:
            ValueError: If the result structure is unexpected.
        """
        if not result:
            return True
        if isinstance(result[0], dict) and "mask" in result[0]:
            return True
        if (
            isinstance(result[0], list)
            and result[0]
            and isinstance(result[0][0], dict)
            and "mask" in result[0][0]
        ):
            return False
        raise ValueError("Unexpected result structure")

    def __call__(
        self,
        images: list[Image.Image]
        | Image.Image
        | np.ndarray
        | xr.DataArray
        | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Process the input image(s) and create segmentation map(s).

        Args:
            images (Union[Image, List[Image]]): The input image(s) to process.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing segmentation maps and metadata.
        """
        if isinstance(images, (np.ndarray, xr.DataArray)):
            if isinstance(images, np.ndarray) and images.ndim == 3:
                # If images is a 3D numpy array, convert it to a PIL Image
                images = Image.fromarray(images)
            elif isinstance(images, np.ndarray) and images.ndim == 4:
                # If images is a 4D numpy array, convert each image in the batch to PIL Images
                images = [Image.fromarray(img) for img in images]
            elif isinstance(images, xr.DataArray) and images.ndim == 3:
                # If images is a 3D xarray DataArray, convert it to a PIL Image
                images = Image.fromarray(images.data)
            elif isinstance(images, xr.DataArray) and images.ndim == 4:
                # If images is a 4D xarray DataArray, convert each image in the batch to PIL Images
                images = [Image.fromarray(img.data) for img in images]

        # logger.debug("Pass image(s) up to HF pipeline...")
        result = super().__call__(images, subtask="semantic", **kwargs)
        # logger.debug("Received result from HF pipeline")
        if self._is_single_image_result(result):
            return [
                self.create_single_segmentation_map(
                    result, result[0]["mask"].size[::-1]
                )
            ]
        else:
            return [
                self.create_single_segmentation_map(
                    img_result, img_result[0]["mask"].size[::-1]
                )
                for img_result in result
            ]


@logger.catch
def create_segmentation_pipeline(
    config: Config | ModelConfig | None = None, device: str | None = "auto", **kwargs
) -> SegmentationPipeline:
    """
    Create and return a SegmentationPipeline instance based on the specified model.

    This function initializes the appropriate model and image processor based on the
    model name, and creates a SegmentationPipeline instance with these components.

    Args:
        config: The model configuration.
        **kwargs: Additional keyword arguments to pass to the SegmentationPipeline constructor.

    Returns:
        SegmentationPipeline: An instance of the SegmentationPipeline class.
    """
    # TODO: Eventually allow for instantiated models and processors to be passed in
    model, image_processor = load_segmentation_model(config, device, **kwargs)

    return SegmentationPipeline(
        model=model,
        image_processor=image_processor,
        device=device,
        subtask="semantic",
        **kwargs,
    )
