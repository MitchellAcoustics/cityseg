from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class InputType(Enum):
    SINGLE_IMAGE = "single_image"
    SINGLE_VIDEO = "single_video"
    DIRECTORY = "directory"


@dataclass
class ModelConfig:
    """
    Configuration for the segmentation model.

    Attributes:
        type (str): Type of the segmentation model (e.g., 'beit', 'oneformer').
        name (str): Name or identifier of the specific model.
        dataset (str): Dataset the model was trained on (e.g., 'cityscapes', 'ade20k').
        max_size (Optional[int]): Maximum size for input image resizing. If None, no resizing is applied.
        tile_size (Optional[int]): Size of tiles for processing large images. If None, no tiling is applied.
        mixed_precision (bool): Flag to enable mixed precision processing.
    """

    type: str
    name: str
    dataset: str
    max_size: Optional[int] = None
    tile_size: Optional[int] = None
    mixed_precision: bool = False


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization settings.

    Attributes:
        alpha (float): Transparency of the segmentation overlay (0.0 to 1.0).
        colormap (str): Colormap to use for visualization. Options: "default", "cityscapes", "ade20k".
    """

    alpha: float = 0.5
    colormap: str = "default"


@dataclass
class Config:
    """
    Main configuration class for the segmentation pipeline.

    Attributes:
        input (Union[Path, str]): Path to the input image, video, or directory.
        output_dir (Optional[Path]): Directory for output files.
        output_prefix (Optional[str]): Prefix for output file names.
        model (ModelConfig): Configuration for the segmentation model.
        frame_step (int): Number of frames to skip in video processing.
        save_raw_segmentation (bool): Whether to save the raw segmentation maps.
        save_colored_segmentation (bool): Whether to save the colored segmentation video/images.
        save_overlay (bool): Whether to save the overlay video/images.
        visualization (VisualizationConfig): Configuration for visualization settings.
    """

    input: Union[Path, str]
    output_dir: Optional[Path]
    output_prefix: Optional[str]
    model: ModelConfig
    frame_step: int = 1
    save_raw_segmentation: bool = True
    save_colored_segmentation: bool = False
    save_overlay: bool = True
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    input_type: InputType = field(init=False)

    def __post_init__(self):
        self.input = Path(self.input)
        self.input_type = self._determine_input_type()

    def _determine_input_type(self) -> InputType:
        if self.input.is_dir():
            return InputType.DIRECTORY
        elif self.input.suffix.lower() in [".mp4", ".avi", ".mov"]:
            return InputType.SINGLE_VIDEO
        elif self.input.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            return InputType.SINGLE_IMAGE
        else:
            raise ValueError(f"Unsupported input type: {self.input}")

    def generate_output_prefix(self) -> str:
        """
        Generate a default output prefix based on input file and model configuration.

        Returns:
            str: Generated output prefix.
        """
        if self.input_type == InputType.DIRECTORY:
            name = self.input.name
        else:
            name = self.input.stem.split("_")[
                0
            ]  # Use only the first part of the filename

        model_type = self.model.type
        dataset = self.model.dataset
        base_name = f"{name}_{model_type}_{dataset}_step{self.frame_step}"

        if self.model.tile_size:
            base_name += f"_tile{self.model.tile_size}"

        return base_name

    def get_output_path(self) -> Path:
        """
        Get the full output path based on output_dir, output_prefix, and input type.

        Returns:
            Path: Full output path.
        """
        if self.output_dir is None:
            self.output_dir = self.input.parent / "output"
        elif not Path(self.output_dir).is_absolute():
            self.output_dir = self.input.parent / self.output_dir

        self.output_dir = self.output_dir.resolve()

        # For directory processing, create a single subdirectory for all outputs
        if self.input_type == InputType.DIRECTORY:
            model_type = self.model.type
            dataset = self.model.dataset
            subdir_name = f"{model_type}_{dataset}_step{self.frame_step}"
            if self.model.tile_size:
                subdir_name += f"_tile{self.model.tile_size}"
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
            Config: Instantiated Config object.
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        model_config = ModelConfig(**config_dict["model"])
        vis_config = VisualizationConfig(**config_dict.get("visualization", {}))

        return cls(
            input=Path(config_dict["input"]),
            output_dir=Path(config_dict.get("output_dir", ""))
            if config_dict.get("output_dir")
            else None,
            output_prefix=config_dict.get("output_prefix"),
            model=model_config,
            frame_step=config_dict.get("frame_step", 1),
            save_raw_segmentation=config_dict.get("save_raw_segmentation", True),
            save_colored_segmentation=config_dict.get(
                "save_colored_segmentation", True
            ),
            save_overlay=config_dict.get("save_overlay", True),
            visualization=vis_config,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Config instance to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the Config.
        """
        config_dict = {
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
        }
        return config_dict
