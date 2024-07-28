"""
This module provides a custom segmentation pipeline for image and video processing.

It extends the functionality of the Hugging Face Transformers library's
ImageSegmentationPipeline to support various segmentation models and
create detailed segmentation maps with associated metadata.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL.Image import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    ImageSegmentationPipeline,
    Mask2FormerForUniversalSegmentation,
    OneFormerProcessor,
)


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

    def _get_palette(self) -> Optional[np.ndarray]:
        """
        Get the color palette for the current model.

        Returns:
            Optional[np.ndarray]: The color palette as a numpy array, or None if not available.
        """
        if hasattr(self.model.config, "palette"):
            return np.array(self.model.config.palette)
        elif "ade" in self.model.config._name_or_path:
            from .palettes import ADE20K_PALETTE

            return np.array(ADE20K_PALETTE)
        elif "mapillary-vistas" in self.model.config._name_or_path:
            from .palettes import MAPILLARY_VISTAS_PALETTE

            return np.array(MAPILLARY_VISTAS_PALETTE)
        else:
            return None

    def create_single_segmentation_map(
        self, annotations: List[Dict[str, Any]], target_size: tuple
    ) -> Dict[str, Any]:
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

    def _is_single_image_result(
        self, result: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
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
        self, images: Union[Image, List[Image]], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Process the input image(s) and create segmentation map(s).

        Args:
            images (Union[Image, List[Image]]): The input image(s) to process.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing segmentation maps and metadata.
        """
        result = super().__call__(images, **kwargs)

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


def create_segmentation_pipeline(
    model_name: str, device: Optional[str] = None, pipe_batch=1, **kwargs: Any
) -> SegmentationPipeline:
    """
    Create and return a SegmentationPipeline instance based on the specified model.

    This function initializes the appropriate model and image processor based on the
    model name, and creates a SegmentationPipeline instance with these components.

    Args:
        pipe_batch:
        model_name (str): The name or path of the pre-trained model to use.
        device (Optional[str]): The device to use for processing (e.g., "cpu", "cuda"). If None, it will be automatically determined.
        **kwargs: Additional keyword arguments to pass to the SegmentationPipeline constructor.

    Returns:
        SegmentationPipeline: An instance of the SegmentationPipeline class.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the appropriate model and image processor based on the model name
    if "oneformer" in model_name.lower():
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        image_processor = OneFormerProcessor.from_pretrained(model_name)
    elif "mask2former" in model_name.lower():
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        image_processor = AutoImageProcessor.from_pretrained(model_name)
    else:
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        image_processor = AutoImageProcessor.from_pretrained(model_name)

    return SegmentationPipeline(
        model=model,
        image_processor=image_processor,
        device=device,
        batch_size=pipe_batch,
        subtask="semantic",
        **kwargs,
    )
