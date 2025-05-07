"""
This module defines the configuration classes and utilities for the semantic segmentation pipeline.

It includes classes for input type enumeration, model configuration, visualization configuration,
and the main configuration class that encapsulates all settings for the segmentation process.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import yaml
from loguru import logger


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
        name: The name or path of the pre-trained model to use.
        model_type: The type of the model (e.g., 'oneformer', 'mask2former').
        max_size: The maximum size for input image resizing.
        device: The device to use for processing (e.g., 'cpu', 'cuda').
    """

    name: str
    model_type: str | None = None
    max_size: int | None = None
    device: str | None = None
    dataset: str | None = None
    num_workers: int | None = 8
    pipe_batch: int | None = 1

    def to_dict(self) -> dict[str, object]:
        """
        Convert the ModelConfig to a dictionary.

        Returns:
            dict[str, object]: Dictionary representation of the ModelConfig.
        """
        return {
            "name": self.name,
            "model_type": self.model_type,
            "max_size": self.max_size,
            "device": self.device,
            "dataset": self.dataset,
            "num_workers": self.num_workers,
            "pipe_batch": self.pipe_batch,
        }

    def __post_init__(self):
        """
        Post-initialization method to set up the model type if not provided.
        """
        self.auto_detect_model_type()
        # Fix: Only compare num_workers if not None
        if self.device == "mps" and (self.num_workers is None or self.num_workers > 0):
            logger.warning(
                "MPS is not compatible with multiple workers in pytorch. Setting num_workers to 0."
            )
            self.num_workers = 0

    def auto_detect_model_type(self):
        """
        Automatically detect the model type from the model name if not provided.
        """

        def auto_model_type(model_name: str) -> str:
            return model_name.split("/")[-1].split("-")[0]

        if self.model_type is None:
            try:
                self.model_type = auto_model_type(self.name)
            except IndexError:
                logger.warning(
                    "Unable to auto-detect model type from the model name and none provided."
                )
                return
            logger.info(f"Auto-detected model type: {self.model_type}")
        elif self.model_type != auto_model_type(self.name):
            logger.warning(
                f"Model type does not match auto-detected model type. Using provided model type: {self.model_type}"
            )


@dataclass
class VisualizationConfig:
    """
    Configuration class for visualization settings.

    Attributes:
        alpha: The alpha value for blending the segmentation mask with the original image.
        colormap: The colormap to use for visualizing the segmentation mask.
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
        input: The input path (file or directory) for processing.
        output_dir: The output directory for saving results.
        output_prefix: The prefix for output file names.
        model: The model configuration.
        frame_step: The frame step for video processing.
        batch_size: The batch size for processing.
        output_fps: The output FPS for processed videos.
        save_raw_segmentation: Whether to save raw segmentation maps.
        save_colored_segmentation: Whether to save colored segmentation maps.
        save_overlay: Whether to save overlay visualizations.
        visualization: The visualization configuration.
        input_type: The type of input (automatically determined).
        force_reprocess: Whether to force reprocessing of existing results.
        disable_tqdm: Whether to disable the progress bar display.
    """

    input: Path | str
    output_dir: Path | None
    output_prefix: str | None
    model: ModelConfig
    ignore_files: list[str] | None = None
    frame_step: int = 1
    batch_size: int = 16
    output_fps: float | None = None
    save_raw_segmentation: bool = True
    save_colored_segmentation: bool = False
    save_overlay: bool = True
    analyze_results: bool = True
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    input_type: InputType = field(init=False)
    force_reprocess: bool = False
    disable_tqdm: bool = False

    def __post_init__(self):
        """
        Post-initialization method to set up the input path and determine the input type.

        Raises:
            ValueError: If the input path does not exist.
        """
        # Convert input to Path if it's a string
        if not isinstance(self.input, Path):
            self.input = Path(self.input)
        if not self.input.exists():
            raise ValueError(f"Input path does not exist: {self.input}")
        self.input_type = self._determine_input_type()
        self.ignore_files = self.ignore_files or []

    @property
    def input_path(self) -> Path:
        """
        Return the input path, guaranteed to be a Path object.

        This property ensures type checkers understand that
        the value is always a Path object after initialization.

        Returns:
            Path: The input path as a Path object
        """
        # We know self.input is a Path after __post_init__, but the type checker doesn't
        # Use typing.cast to tell the type checker this is definitely a Path
        from typing import cast

        return cast(Path, self.input)

    def _determine_input_type(self) -> InputType:
        """
        Determine the type of input based on the input path.

        Returns:
            InputType: The determined input type.

        Raises:
            ValueError: If the input type is not supported.
        """
        # Use typing.cast to tell the type checker this is definitely a Path
        from typing import cast

        input_path = cast(Path, self.input)

        if input_path.is_dir():
            return InputType.DIRECTORY
        elif input_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
            return InputType.SINGLE_VIDEO
        elif input_path.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        ]:
            return InputType.SINGLE_IMAGE
        else:
            raise ValueError(f"Unsupported input type: {input_path}")

    def generate_output_prefix(self) -> str:
        """
        Generate an output prefix based on the input file name and model configuration.

        Returns:
            str: The generated output prefix.
        """
        # Use typing.cast to tell the type checker this is definitely a Path
        from typing import cast

        input_path = cast(Path, self.input)

        if self.input_type == InputType.DIRECTORY:
            name = input_path.name
        else:
            name = input_path.stem

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
        # Use typing.cast to tell the type checker this is definitely a Path
        from typing import cast

        input_path = cast(Path, self.input)

        if self.output_dir is None:
            self.output_dir = input_path.parent / "output"
        elif not Path(self.output_dir).is_absolute():
            self.output_dir = input_path.parent / self.output_dir

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
            return self.output_dir / f"{prefix}{input_path.suffix}"
        else:  # SINGLE_VIDEO
            return self.output_dir / f"{prefix}.mp4"

    @classmethod
    def from_yaml(cls, config_path: Path) -> Config:
        """
        Create a Config instance from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

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
            ignore_files=config_dict.get("ignore_files", []),
            frame_step=config_dict.get("frame_step", 1),
            batch_size=config_dict.get("batch_size", 16),
            output_fps=config_dict.get("output_fps"),
            save_raw_segmentation=config_dict.get("save_raw_segmentation", True),
            save_colored_segmentation=config_dict.get(
                "save_colored_segmentation", False
            ),
            save_overlay=config_dict.get("save_overlay", True),
            analyze_results=config_dict.get("analyze_results", True),
            visualization=vis_config,
            force_reprocess=config_dict.get("force_reprocess", False),
            disable_tqdm=config_dict.get("disable_tqdm", False),
        )

    def to_dict(self) -> dict[str, object]:
        """
        Convert the Config instance to a dictionary.

        Returns:
            dict[str, object]: A dictionary representation of the Config instance.
        """
        return {
            "input": str(self.input),
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "output_prefix": self.output_prefix,
            "model": asdict(self.model),
            "ignore_files": self.ignore_files,
            "frame_step": self.frame_step,
            "batch_size": self.batch_size,
            "output_fps": self.output_fps,
            "save_raw_segmentation": self.save_raw_segmentation,
            "save_colored_segmentation": self.save_colored_segmentation,
            "save_overlay": self.save_overlay,
            "analyze_results": self.analyze_results,
            "visualization": asdict(self.visualization),
            "input_type": self.input_type.value,
            "force_reprocess": self.force_reprocess,
            "disable_tqdm": self.disable_tqdm,
        }


class ConfigHasher:
    """
    A utility class for generating hashes of relevant configuration settings.
    """

    @staticmethod
    def get_relevant_config(config: Config) -> dict[str, object]:
        """
        Extract the relevant configuration settings for hashing.

        This method filters out configuration settings that don't affect the
        analysis results or output format, focusing only on settings that would
        require reprocessing if changed.

        Args:
            config: The full configuration object.

        Returns:
            dict[str, object]: A dictionary of relevant configuration settings.
        """
        return {
            "model": {
                "name": config.model.name,
                "model_type": config.model.model_type,
                "max_size": config.model.max_size,
            },
            "frame_step": config.frame_step,
            "save_raw_segmentation": config.save_raw_segmentation,
            "save_colored_segmentation": config.save_colored_segmentation,
            "save_overlay": config.save_overlay,
            "visualization": config.visualization.alpha,  # Assuming this is the relevant part
        }

    @staticmethod
    def calculate_hash(config: Config) -> str:
        """
        Calculate a hash of the relevant configuration settings.

        This method creates a deterministic hash of the configuration settings
        that affect the analysis results or output format.

        Args:
            config: The full configuration object.

        Returns:
            str: A hexadecimal string representing the hash of the relevant config.
        """
        relevant_config = ConfigHasher.get_relevant_config(config)
        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
