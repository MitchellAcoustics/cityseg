"""
This module provides a custom segmentation pipeline for image and video processing.

It extends the functionality of the Hugging Face Transformers library's
ImageSegmentationPipeline to support various segmentation models and
create detailed segmentation maps with associated metadata.
"""

import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from loguru import logger
from PIL.Image import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    AutoProcessor,
    BeitForSemanticSegmentation,
    ImageSegmentationPipeline,
    Mask2FormerForUniversalSegmentation,
    MaskFormerForInstanceSegmentation,
    OneFormerForUniversalSegmentation,
    OneFormerProcessor,
    SegformerForSemanticSegmentation,
)

from .config import ModelConfig


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

    @staticmethod
    def _is_single_image_result(
        result: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
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
    config: ModelConfig, **kwargs: Any
) -> SegmentationPipeline:
    """
    Create and return a SegmentationPipeline instance based on the specified model.

    This function initializes the appropriate model and image processor based on the
    model name, and creates a SegmentationPipeline instance with these components.

    Args:
        config:
        **kwargs: Additional keyword arguments to pass to the SegmentationPipeline constructor.

    Returns:
        SegmentationPipeline: An instance of the SegmentationPipeline class.
    """
    model_name = config.name
    model_type = config.model_type
    device = config.device
    dataset = config.dataset

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    # Initialize the appropriate model and image processor based on the model name
    if "oneformer" == model_type:
        warnings.warn(
            "OneFormer models are experimental and may not be fully supported"
        )
        try:
            model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
            image_processor = AutoProcessor.from_pretrained(model_name)
        except ValueError as e:
            logger.error(f"Error loading model: {e}")

    elif "mask2former" == model_type:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        image_processor = AutoImageProcessor.from_pretrained(model_name)

    elif "maskformer" == model_type:
        model = MaskFormerForInstanceSegmentation.from_pretrained(model_name)
        image_processor = AutoImageProcessor.from_pretrained(model_name)

    elif "beit" == model_type:
        if device != "cpu":
            logger.warning(
                "Beit models are not supported on GPU and will be loaded on CPU"
            )
        device = "cpu"
        model = BeitForSemanticSegmentation.from_pretrained(model_name)
        image_processor = AutoImageProcessor.from_pretrained(model_name)

    elif "segformer" == model_type:
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        image_processor = AutoImageProcessor.from_pretrained(model_name)

        if dataset == "sidewalk-semantic":
            logging.debug("Loading Sidewalk Semantic dataset label mappings...")
            with open("SemanticSidewalk_id2label.json") as f:
                id2label = json.load(f)
            model.config.id2label = id2label
    else:
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        image_processor = AutoImageProcessor.from_pretrained(model_name)

    return SegmentationPipeline(
        model=model,
        image_processor=image_processor,
        device=device,
        subtask="semantic",
        **kwargs,
    )
