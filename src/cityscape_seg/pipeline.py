from typing import Optional
import numpy as np
from transformers import (
    ImageSegmentationPipeline,
    AutoModelForSemanticSegmentation,
    AutoImageProcessor,
)
from transformers import OneFormerProcessor, Mask2FormerForUniversalSegmentation
import torch


class SegmentationPipeline(ImageSegmentationPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.palette = self._get_palette()

    def _get_palette(self) -> Optional[np.ndarray]:
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

    def create_single_segmentation_map(self, annotations, target_size):
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

    def _is_single_image_result(self, result):
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

    def __call__(self, images, **kwargs):
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


def create_segmentation_pipeline(model_name: str, device: str = None, **kwargs):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
        subtask="semantic",
        **kwargs,
    )
