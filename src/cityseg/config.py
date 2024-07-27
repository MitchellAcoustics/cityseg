"""
This module defines the configuration classes and utilities for the semantic segmentation pipeline.

It includes classes for input type enumeration, model configuration, visualization configuration,
and the main configuration class that encapsulates all settings for the segmentation process.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class InputType(Enum):
    """Enumeration of supported input types for the segmentation pipeline."""

    SINGLE_IMAGE = "single_image"
    SINGLE_VIDEO = "single_video"
    DIRECTORY = "directory"


@dataclass
class ModelConfig:
    """
    Configuration class for the segmentation model.

    Attributes:
        name (str): The name or path of the pre-trained model to use.
        model_type (Optional[str]): The type of the model (e.g., 'oneformer', 'mask2former').
        max_size (Optional[int]): The maximum size for input image resizing.
        device (Optional[str]): The device to use for processing (e.g., 'cpu', 'cuda').
    """

    name: str
    model_type: Optional[str] = (
        None  # Can be 'oneformer', 'mask2former', or None for auto-detection
    )
    max_size: Optional[int] = None
    device: Optional[str] = None

    # TODO: impelement model_type auto-detection
    # TODO: implement device auto-detection


@dataclass
class VisualizationConfig:
    """
    Configuration class for visualization settings.

    Attributes:
        alpha (float): The alpha value for blending the segmentation mask with the original image.
        colormap (str): The colormap to use for visualizing the segmentation mask.
    """

    alpha: float = 0.5
    colormap: str = "default"


@dataclass
class Config:
    """
    Main configuration class for the segmentation pipeline.

    This class encapsulates all settings required for the segmentation process,
    including input/output paths, model configuration, processing parameters,
    and visualization settings.

    Attributes:
        input (Union[Path, str]): The input path (file or directory) for processing.
        output_dir (Optional[Path]): The output directory for saving results.
        output_prefix (Optional[str]): The prefix for output file names.
        model (ModelConfig): The model configuration.
        frame_step (int): The frame step for video processing.
        batch_size (int): The batch size for processing.
        output_fps (Optional[float]): The output FPS for processed videos.
        save_raw_segmentation (bool): Whether to save raw segmentation maps.
        save_colored_segmentation (bool): Whether to save colored segmentation maps.
        save_overlay (bool): Whether to save overlay visualizations.
        visualization (VisualizationConfig): The visualization configuration.
        input_type (InputType): The type of input (automatically determined).
        force_reprocess (bool): Whether to force reprocessing of existing results.
    """

    input: Union[Path, str]
    output_dir: Optional[Path]
    output_prefix: Optional[str]
    model: ModelConfig
    frame_step: int = 1
    batch_size: int = 16
    output_fps: Optional[float] = None
    save_raw_segmentation: bool = True
    save_colored_segmentation: bool = False
    save_overlay: bool = True
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    input_type: InputType = field(init=False)
    force_reprocess: bool = False

    def __post_init__(self):
        """
        Post-initialization method to set up the input path and determine the input type.

        Raises:
            ValueError: If the input path does not exist.
        """
        self.input = Path(self.input)
        if not self.input.exists():
            raise ValueError(f"Input path does not exist: {self.input}")
        self.input_type = self._determine_input_type()

    def _determine_input_type(self) -> InputType:
        """
        Determine the type of input based on the input path.

        Returns:
            InputType: The determined input type.

        Raises:
            ValueError: If the input type is not supported.
        """
        if self.input.is_dir():
            return InputType.DIRECTORY
        elif self.input.suffix.lower() in [".mp4", ".avi", ".mov"]:
            return InputType.SINGLE_VIDEO
        elif self.input.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        ]:
            return InputType.SINGLE_IMAGE
        else:
            raise ValueError(f"Unsupported input type: {self.input}")

    def generate_output_prefix(self) -> str:
        """
        Generate an output prefix based on the input file name and model configuration.

        Returns:
            str: The generated output prefix.
        """
        if self.input_type == InputType.DIRECTORY:
            name = self.input.name
        else:
            name = self.input.stem.split("_")[
                0
            ]  # Use only the first part of the filename

        model_name = self.model.name.split("/")[-1]
        base_name = f"{name}_{model_name}_step{self.frame_step}"

        return base_name

    def get_output_path(self) -> Path:
        """
        Get the full output path based on the configuration.

        This method determines the appropriate output directory and file name
        based on the input type and configuration settings.

        Returns:
            Path: The full output path.
        """
        if self.output_dir is None:
            self.output_dir = self.input.parent / "output"
        elif not Path(self.output_dir).is_absolute():
            self.output_dir = self.input.parent / self.output_dir

        self.output_dir = self.output_dir.resolve()

        if self.input_type == InputType.DIRECTORY:
            model_name = self.model.name.split("/")[-1]
            subdir_name = f"{model_name}_step{self.frame_step}"
            self.output_dir = self.output_dir / subdir_name

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.input_type == InputType.DIRECTORY:
            return self.output_dir

        prefix = self.output_prefix or self.generate_output_prefix()
        if self.input_type == InputType.SINGLE_IMAGE:
            return self.output_dir / f"{prefix}{self.input.suffix}"
        else:  # SINGLE_VIDEO
            return self.output_dir / f"{prefix}.mp4"

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """
        Create a Config instance from a YAML file.

        Args:
            config_path (Path): Path to the YAML configuration file.

        Returns:
            Config: An instance of the Config class.
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert string paths back to Path objects
        if "input" in config_dict:
            config_dict["input"] = Path(config_dict["input"])
        if "output_dir" in config_dict:
            config_dict["output_dir"] = Path(config_dict["output_dir"])

        model_config = ModelConfig(**config_dict.get("model", {}))
        vis_config = VisualizationConfig(**config_dict.get("visualization", {}))

        return cls(
            input=config_dict["input"],
            output_dir=config_dict.get("output_dir"),
            output_prefix=config_dict.get("output_prefix"),
            model=model_config,
            frame_step=config_dict.get("frame_step", 1),
            save_raw_segmentation=config_dict.get("save_raw_segmentation", True),
            save_colored_segmentation=config_dict.get(
                "save_colored_segmentation", False
            ),
            save_overlay=config_dict.get("save_overlay", True),
            visualization=vis_config,
            force_reprocess=config_dict.get("force_reprocess", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Config instance to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the Config instance.
        """
        return {
            "input": str(self.input),
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "output_prefix": self.output_prefix,
            "model": asdict(self.model),
            "frame_step": self.frame_step,
            "save_raw_segmentation": self.save_raw_segmentation,
            "save_colored_segmentation": self.save_colored_segmentation,
            "save_overlay": self.save_overlay,
            "visualization": asdict(self.visualization),
            "input_type": self.input_type.value,
            "force_reprocess": self.force_reprocess,
        }
