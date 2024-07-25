import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from loguru import logger
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from transformers import (
    BeitForSemanticSegmentation,
    BeitImageProcessor,
    OneFormerForUniversalSegmentation,
    OneFormerProcessor,
)

from .config import ModelConfig
from .exceptions import ModelError
from .palettes import ADE20K_PALETTE, CITYSCAPES_PALETTE
from .utils import get_device


class TqdmCompatibleSink:
    def __init__(self, level=logging.INFO):
        self.level = level

    def write(self, message):
        tqdm.write(message, end="")


# Configure loguru
logger.remove()  # Remove default handler
logger.add(TqdmCompatibleSink(), format="{time} | {level} | {message}", level="INFO")
logger.add("file_{time}.log", rotation="1 day")


class SegmentationModelBase(ABC):
    """
    Abstract base class for segmentation models.

    This class defines the interface and common functionality for all
    segmentation models used in the pipeline.

    Attributes:
        model_config (Dict[str, Any]): Configuration dictionary for the model.
        device (torch.device): Device to run the model on (CPU, CUDA, or MPS).
        mixed_precision (bool): Flag for using mixed precision in computations.
        max_size (Optional[int]): Maximum size for input image resizing.
        tile_size (Optional[int]): Size of tiles for processing large images.
        dataset (str): Dataset the model was trained on.
        model: The actual PyTorch model instance.
        processor: The model's processor for input preparation and post-processing.
        category_names (Dict[int, str]): Mapping of category IDs to names.
        num_categories (int): Total number of categories the model can predict.
        palette (List[Tuple[int, int, int]]): Color palette for visualization.
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.device = get_device(model_config.device)
        self.mixed_precision = model_config.mixed_precision
        self.max_size = model_config.max_size
        self.tile_size = model_config.tile_size
        self.dataset = model_config.dataset.lower()

        self.logger = logger.bind(model_type=self.__class__.__name__)

        self.model = self.load_model()
        self.processor = self.load_processor()
        self.category_names = self.get_category_names()
        self.num_categories = len(self.category_names)
        self.palette = self.get_palette()

        self.logger.info(
            "Model initialized",
            device=str(self.device),
            mixed_precision=self.mixed_precision,
            max_size=self.max_size,
            tile_size=self.tile_size,
            dataset=self.dataset,
        )

    @abstractmethod
    def load_model(self):
        """
        Load the PyTorch model.

        This method should be implemented by subclasses to load the specific model architecture.

        Returns:
            The loaded PyTorch model.
        """

        pass

    def move_model_to_device(self, model):
        """
        Move the model to the appropriate device and handle MPS-specific conversion.

        Args:
            model: The PyTorch model to be moved.

        Returns:
            The model on the appropriate device.
        """

        if self.device.type == "mps":
            return model.to(torch.float32).to(self.device)
        else:
            return model.to(self.device)

    @abstractmethod
    def load_processor(self):
        """
        Load the model's processor.

        This method should be implemented by subclasses to load the specific processor for the model.

        Returns:
            The loaded processor.
        """

        pass

    @abstractmethod
    def get_category_names(self):
        """
        Get the mapping of category IDs to category names.

        This method should be implemented by subclasses to provide the category mapping for the specific model.

        Returns:
            Dict[int, str]: Mapping of category IDs to category names.
        """

        pass

    @abstractmethod
    def process_inputs(self, inputs):
        """
        Process the input data for the model.

        This method should be implemented by subclasses to handle the specific input processing required by the model.

        Args:
            inputs: The input data to be processed.

        Returns:
            The processed inputs ready for the model.
        """

        pass

    @torch.no_grad()
    def process_tile(self, tile):
        """
        Process a single image tile through the segmentation model.

        This method handles the forward pass of the model and post-processing of the output.

        Args:
            tile: The image tile to be processed.

        Returns:
            The processed segmentation map for the tile.
        """
        self.logger.debug("Processing tile", tile_size=tile.size)
        inputs = self.process_inputs(tile)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.device.type == "cuda":
            with autocast(enabled=self.mixed_precision):
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)

        result = self.post_process(outputs, tile.size)
        return result

    @abstractmethod
    def post_process(self, outputs, target_size):
        """
        Post-process the model outputs.

        This method should be implemented by subclasses to handle the specific post-processing required by the model.

        Args:
            outputs: The raw outputs from the model.
            target_size: The target size for the output segmentation map.

        Returns:
            The post-processed segmentation map.
        """

        pass

    def get_palette(self) -> List[Tuple[int, int, int]]:
        """
        Get the color palette for the current dataset.

        Returns:
            List[Tuple[int, int, int]]: List of RGB color tuples for visualization.
        """

        if self.dataset == "cityscapes":
            return CITYSCAPES_PALETTE
        elif self.dataset == "ade20k":
            return ADE20K_PALETTE
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

    def colorize_segmentation(self, seg_map):
        """
        Colorize a segmentation map using the model's color palette.

        Args:
            seg_map: The segmentation map to be colorized.

        Returns:
            The colorized segmentation map.
        """

        return torch.tensor(self.palette, device=self.device)[
            seg_map.to(torch.long) % len(self.palette)
        ]


class BeitSegmentationModel(SegmentationModelBase):
    """
    Implementation of SegmentationModelBase for BEiT models.

    This class provides the specific implementations required for BEiT segmentation models.
    """

    def load_model(self):
        self.logger.info("Loading BEiT model", model_name=self.model_config.name)
        try:
            model = BeitForSemanticSegmentation.from_pretrained(self.model_config.name)
            return self.move_model_to_device(model)
        except Exception as e:
            self.logger.error("Error loading BEiT model", error=str(e))
            raise ModelError(f"Error loading BEiT model: {str(e)}")

    def load_processor(self):
        return BeitImageProcessor.from_pretrained(self.model_config.name)

    def get_category_names(self):
        return {int(k): v for k, v in self.model.config.id2label.items()}

    def process_inputs(self, tile):
        return self.processor(images=tile, return_tensors="pt")

    def post_process(self, outputs, target_size):
        seg_map = outputs.logits.argmax(dim=1)[0]
        return self.processor.post_process_semantic_segmentation(
            seg_map, target_sizes=[target_size[::-1]]
        )[0]


class OneFormerSegmentationModel(SegmentationModelBase):
    """
    Implementation of SegmentationModelBase for OneFormer models.

    This class provides the specific implementations required for OneFormer segmentation models.
    """

    def load_model(self):
        self.logger.info("Loading OneFormer model", model_name=self.model_config.name)
        if "dinat" in self.model_config.name and self.device.type == "mps":
            self.device = get_device("cpu")
            self.logger.info("Forcing CPU for DINAT model")
        try:
            model = OneFormerForUniversalSegmentation.from_pretrained(
                self.model_config.name
            )
            return self.move_model_to_device(model)
        except Exception as e:
            self.logger.error("Error loading OneFormer model", error=str(e))
            raise ModelError(f"Error loading OneFormer model: {str(e)}")

    def load_processor(self):
        return OneFormerProcessor.from_pretrained(self.model_config.name)

    def get_category_names(self):
        return {int(k): v for k, v in self.model.config.id2label.items()}

    def process_inputs(self, tile):
        return self.processor(
            images=tile, task_inputs=["semantic"], return_tensors="pt"
        )

    def post_process(self, outputs, target_size):
        return self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[target_size[::-1]]
        )[0]


def create_segmentation_model(model_config: ModelConfig) -> SegmentationModelBase:
    """
    Factory function to create the appropriate segmentation model based on the configuration.

    Args:
        model_config (Dict[str, Any]): Configuration dictionary for the model.

    Returns:
        SegmentationModelBase: An instance of the appropriate segmentation model.

    Raises:
        ValueError: If an unsupported model type is specified in the configuration.
    """

    model_type = model_config.type.lower()
    logger.info("Creating segmentation model", model_type=model_type)
    try:
        if model_type == "beit":
            return BeitSegmentationModel(model_config)
        elif model_type == "oneformer":
            return OneFormerSegmentationModel(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    except Exception as e:
        logger.error("Error creating segmentation model", error=str(e))
        raise ModelError(f"Error creating segmentation model: {str(e)}")
