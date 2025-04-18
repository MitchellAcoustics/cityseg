"""
This module provides a custom segmentation pipeline for image and video processing.

It extends the functionality of the Hugging Face Transformers library's
ImageSegmentationPipeline to support various segmentation models and
create detailed segmentation maps with associated metadata.
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any, TypedDict

import numpy as np
import torch
from loguru import logger
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    AutoProcessor,
    BeitForSemanticSegmentation,
    ImageSegmentationPipeline,
    Mask2FormerForUniversalSegmentation,
    MaskFormerForInstanceSegmentation,
    OneFormerForUniversalSegmentation,
    SegformerForSemanticSegmentation,
)

from cityseg.analysis.visualization import Palette
from ..core import ModelConfig

# Define types to match transformers library
Prediction = dict[str, Any]
Predictions = list[Prediction]


class SegmentationResult(TypedDict):
    """
    A typed dictionary representing the result of a segmentation operation.

    This provides a clear structure for the segmentation results, making
    it easier to work with and understand the data.

    Attributes:
        seg_map: The segmentation map as a numpy array.
        label2id: A dictionary mapping label names to their IDs.
        id2label: A dictionary mapping IDs to their label names.
        palette: The color palette used for visualization.
    """

    seg_map: np.ndarray
    label2id: dict[str, int]
    id2label: dict[str, str]
    palette: Palette | np.ndarray | None


class SegmentationPipeline(ImageSegmentationPipeline):
    """
    A custom segmentation pipeline that extends ImageSegmentationPipeline.

    This class provides additional functionality for creating and processing
    segmentation maps, including support for different color palettes and
    batch processing of images.

    Attributes:
        palette: The color palette used for visualization.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.palette = self._get_palette()

    def _get_palette(self) -> np.ndarray | None:
        """
        Get the color palette for the current model.

        Returns:
            The color palette as a numpy array, or None if not available.
        """
        if hasattr(self.model.config, "palette"):
            return np.array(self.model.config.palette, dtype=np.uint8)

        # Use the common palette lookup utility
        from ..utils import get_palette

        palette = get_palette(self.model.config._name_or_path)
        if palette is not None:
            return np.array(palette, dtype=np.uint8)
        return None

    def create_single_segmentation_map(
        self, result: list[dict[str, object]], target_size: tuple[int, int]
    ) -> SegmentationResult:
        """
        Create a single segmentation map from model outputs.

        Args:
            result: List of prediction dictionaries from the model
            target_size: The target size (height, width) of the segmentation map

        Returns:
            A dictionary containing the segmentation map and associated metadata
        """
        seg_map = np.zeros(target_size, dtype=np.int32)

        # Handle different model output formats
        for pred in result:
            if isinstance(pred, dict):
                if "mask" in pred and "label" in pred:
                    mask = np.array(pred["mask"])
                    label = pred["label"]
                    if isinstance(label, str) and hasattr(
                        self.model.config, "label2id"
                    ):
                        label_id = self.model.config.label2id.get(label, 0)
                    elif isinstance(label, (int, np.integer)):
                        label_id = int(label)
                    else:
                        continue
                    seg_map[mask != 0] = label_id

        return {
            "seg_map": seg_map,
            "label2id": getattr(self.model.config, "label2id", {}),
            "id2label": getattr(self.model.config, "id2label", {}),
            "palette": self.palette,
        }

    @staticmethod
    def _is_single_image_result(
        result: list[dict[str, object]] | list[list[dict[str, object]]],
    ) -> bool:
        """
        Determine if the result is for a single image or multiple images.

        Args:
            result: The result to check.

        Returns:
            True if the result is for a single image, False otherwise.

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

    def __call__(self, images=None, **kwargs) -> Predictions | list[Prediction]:
        """
        Process the input image(s) through the original pipeline.

        This maintains compatibility with the parent class by returning the same type.

        Args:
            images: The input image(s) to process.
            **kwargs: Additional keyword arguments.

        Returns:
            The original pipeline results in the format expected by the parent class.
        """
        # Call the parent implementation with semantic segmentation
        return super().__call__(images, subtask="semantic", **kwargs)

    def process_results(
        self, result: Predictions | list[Prediction]
    ) -> list[SegmentationResult]:
        """
        Transform the standard pipeline results into our custom SegmentationResult format.

        This separates the processing logic from the __call__ method.

        Args:
            result: The result from the original pipeline.

        Returns:
            A list of dictionaries containing segmentation maps and metadata.
        """

        def get_mask_size(prediction: dict[str, object]) -> tuple[int, int]:
            """Helper to safely extract mask size from prediction"""
            if not isinstance(prediction, dict):
                return (0, 0)
            mask = prediction.get("mask")
            if mask is None:
                return (0, 0)
            if hasattr(mask, "size"):
                size = getattr(mask, "size")
                if isinstance(size, (tuple, list)) and len(size) >= 2:
                    return (int(size[1]), int(size[0]))  # Convert to (height, width)
            if isinstance(mask, np.ndarray):
                if mask.ndim >= 2:
                    return (int(mask.shape[0]), int(mask.shape[1]))
            return (0, 0)

        def ensure_list_dict(result: object) -> list[dict[str, object]]:
            """Helper to ensure result is a list of dicts"""
            if isinstance(result, dict):
                return [result]
            if isinstance(result, list):
                return [r for r in result if isinstance(r, dict)]
            return []

        if self._is_single_image_result(result):
            predictions = ensure_list_dict(result)
            if predictions:
                size = get_mask_size(predictions[0])
                return [self.create_single_segmentation_map(predictions, size)]
            return [self.create_single_segmentation_map([], (0, 0))]
        else:
            outputs = []
            for img_result in result:
                predictions = ensure_list_dict(img_result)
                size = get_mask_size(predictions[0]) if predictions else (0, 0)
                outputs.append(self.create_single_segmentation_map(predictions, size))
            return outputs


@logger.catch
def create_segmentation_pipeline(
    config: ModelConfig, **kwargs: object
) -> SegmentationPipeline:
    """
    Create and return a SegmentationPipeline instance based on the specified model.

    This function initializes the appropriate model and image processor based on the
    model name, and creates a SegmentationPipeline instance with these components.

    Args:
        config: Model configuration containing model name, type, device, etc.
        **kwargs: Additional keyword arguments to pass to the SegmentationPipeline constructor.

    Returns:
        An instance of the SegmentationPipeline class.
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

    model = None
    image_processor = None

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
            from pathlib import Path

            # Look for the JSON file in the cityseg module directory
            module_dir = Path(__file__).parent.parent.parent
            json_path = module_dir / "SemanticSidewalk_id2label.json"

            with open(json_path) as f:
                id2label = json.load(f)
            model.config.id2label = id2label
    else:
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        image_processor = AutoImageProcessor.from_pretrained(model_name)

    if model is None or image_processor is None:
        raise RuntimeError("Failed to load model or image processor.")

    return SegmentationPipeline(
        model=model,
        image_processor=image_processor,
        device=device,
        subtask="semantic",
        num_workers=config.num_workers,
        **kwargs,
    )
