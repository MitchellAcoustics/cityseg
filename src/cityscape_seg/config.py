from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


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
        output_prefix (Path): Prefix for output file paths.
        model (ModelConfig): Configuration for the segmentation model.
        frame_step (int): Number of frames to skip in video processing.
        num_workers (int): Number of parallel workers for directory processing.
        generate_overlay (bool): Whether to generate an overlay of the segmentation on the original image.
        save_colored_segmentation (bool): Whether to save the colored segmentation map.
        save_raw_segmentation (bool): Whether to save the raw segmentation map (as numpy array).
        visualization (VisualizationConfig): Configuration for visualization settings.
    """

    input: Union[Path, str]
    output_prefix: Path
    model: ModelConfig
    frame_step: int = 1
    num_workers: int = 1
    generate_overlay: bool = True
    save_colored_segmentation: bool = True
    save_raw_segmentation: bool = True
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self):
        """
        Post-initialization method to set default output prefix if not provided.
        """
        if self.output_prefix is None:
            self.output_prefix = self.generate_output_prefix()

    def generate_output_prefix(self) -> Path:
        """
        Generate a default output prefix based on input file and model configuration.

        Returns:
            Path: Generated output prefix path.
        """
        # TODO: Still not creating output dir correctly for dir processing
        input_path = Path(self.input)
        name = input_path.name if input_path.is_dir() else input_path.stem.split("_")[0]
        model_type = self.model.type
        dataset = self.model.dataset
        base_name = f"{name}_{model_type}_{dataset}_step{self.frame_step}"

        if self.model.tile_size:
            base_name += f"_tile{self.model.tile_size}"

        output_dir = input_path.parent / "output" / base_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

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
            output_prefix=Path(config_dict.get("output_prefix", "")),
            model=model_config,
            frame_step=config_dict.get("frame_step", 1),
            num_workers=config_dict.get("num_workers", 1),
            generate_overlay=config_dict.get("generate_overlay", True),
            save_colored_segmentation=config_dict.get(
                "save_colored_segmentation", True
            ),
            save_raw_segmentation=config_dict.get("save_raw_segmentation", True),
            visualization=vis_config,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Config instance to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the Config.
        """
        return {
            "input": str(self.input),
            "output_prefix": str(self.output_prefix),
            "model": asdict(self.model),
            "frame_step": self.frame_step,
            "num_workers": self.num_workers,
            "generate_overlay": self.generate_overlay,
            "save_colored_segmentation": self.save_colored_segmentation,
            "save_raw_segmentation": self.save_raw_segmentation,
            "visualization": asdict(self.visualization),
        }
