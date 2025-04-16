"""
This module provides Hamilton workflow integration for the CitySeg pipeline.

It includes:
1. Hamilton driver setup and configuration
2. Functions that follow Hamilton's conventions for use in data flow pipelines
3. Data transformation and processing utilities
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image
from hamilton import driver, function_modifiers as fm
from loguru import logger

from ..components.video import VideoProcessor
from ..components.image import ImageProcessor
from ..components.segmentation import SegmentationProcessor
from ..components.dataset import DatasetBuilder
from ..analysis.analyzer import SegmentationAnalyzer
from ..analysis.visualization import VisualizationHandler
from ..core import Config, InputType, ModelConfig


# --- Driver Creation Functions ---


def create_video_driver(config: Config, cache_dir: Path | None = None) -> driver.Driver:
    """
    Create a Hamilton driver for video processing.

    Args:
        config: Configuration for the processing pipeline
        cache_dir: Optional directory for caching results

    Returns:
        Configured Hamilton driver for video processing
    """
    # Import Hamilton functions
    import sys

    sys.modules["cityseg.workflow.hamilton"] = sys.modules[__name__]

    # Set up driver builder
    builder = driver.Builder()
    builder = builder.with_modules(sys.modules[__name__])

    # Add caching if provided
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        builder = builder.enable_dynamic_execution(allow_experimental_mode=True)
        builder = builder.with_config({"results_dir": str(cache_dir)})

    # Build the driver
    drv = builder.build()

    return drv


def create_image_driver(config: Config, cache_dir: Path | None = None) -> driver.Driver:
    """
    Create a Hamilton driver for image processing.

    Args:
        config: Configuration for the processing pipeline
        cache_dir: Optional directory for caching results

    Returns:
        Configured Hamilton driver for image processing
    """
    # Import Hamilton functions
    import sys

    sys.modules["cityseg.workflow.hamilton"] = sys.modules[__name__]

    # Set up driver builder
    builder = driver.Builder()
    builder = builder.with_modules(sys.modules[__name__])

    # Add caching if provided
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        builder = builder.enable_dynamic_execution(allow_experimental_mode=True)
        builder = builder.with_config({"results_dir": str(cache_dir)})

    # Build the driver
    drv = builder.build()

    return drv


def create_directory_driver(
    config: Config, cache_dir: Path | None = None
) -> driver.Driver:
    """
    Create a Hamilton driver for directory processing.

    Args:
        config: Configuration for the processing pipeline
        cache_dir: Optional directory for caching results

    Returns:
        Configured Hamilton driver for directory processing
    """
    # Import Hamilton functions
    import sys

    sys.modules["cityseg.workflow.hamilton"] = sys.modules[__name__]

    # Set up driver builder
    builder = driver.Builder()
    builder = builder.with_modules(sys.modules[__name__])

    # Add caching if provided
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        builder = builder.enable_dynamic_execution(allow_experimental_mode=True)
        builder = builder.with_config({"results_dir": str(cache_dir)})

    # Build the driver
    drv = builder.build()

    return drv


def process(config: Config, cache_dir: Path | None = None) -> dict[str, object]:
    """
    Process input based on configuration using the appropriate Hamilton driver.

    Args:
        config: Configuration for the processing pipeline
        cache_dir: Optional directory for caching results

    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing {config.input_type.value} with Hamilton")

    if config.input_type == InputType.SINGLE_VIDEO:
        drv = create_video_driver(config, cache_dir)
        desired_outputs = ["segmentation_dataset", "analysis_path", "overlay_path"]

    elif config.input_type == InputType.SINGLE_IMAGE:
        drv = create_image_driver(config, cache_dir)
        desired_outputs = ["segmentation_dataset", "analysis_path", "overlay_path"]

    elif config.input_type == InputType.DIRECTORY:
        drv = create_directory_driver(config, cache_dir)
        desired_outputs = ["processed_videos"]

    else:
        raise ValueError(f"Unsupported input type: {config.input_type}")

    # Execute the driver
    result = drv.execute(desired_outputs)

    return result


# --- Video Processing Functions ---


def video_metadata(video_path: str) -> dict[str, object]:
    """
    Video metadata extracted from the video file.

    Args:
        video_path: Path to the video file

    Returns:
        Metadata about the video including dimensions, frame count, and fps
    """
    return VideoProcessor.get_metadata(Path(video_path))


@fm.config.when(source="video_metadata")
def frame_count(video_metadata: dict[str, object]) -> int:
    """
    Number of frames in the video.

    Args:
        video_metadata: Metadata dictionary from the video

    Returns:
        Total frame count
    """
    return video_metadata["frame_count"]


def frame_indices(frame_count: int, frame_step: int) -> list[int]:
    """
    Indices of the frames to extract based on the frame step.

    Args:
        frame_count: Total number of frames in the video
        frame_step: Step size for frame extraction

    Returns:
        List of frame indices to extract
    """
    return VideoProcessor.get_frame_indices(frame_count, frame_step)


@fm.config.when(source="video_metadata")
def video_dimensions(video_metadata: dict[str, object]) -> tuple[int, int]:
    """
    Dimensions of the video frames (width, height).

    Args:
        video_metadata: Metadata dictionary from the video

    Returns:
        Tuple of (width, height)
    """
    return (video_metadata["width"], video_metadata["height"])


def video_frames(video_path: str, frame_indices: list[int]) -> list[Image.Image]:
    """
    Extracted frames from the video at specified indices.

    Args:
        video_path: Path to the video file
        frame_indices: List of frame indices to extract

    Returns:
        List of PIL Image objects
    """
    return VideoProcessor.get_frames(Path(video_path), frame_indices)


def resized_frames(
    video_frames: list[Image.Image], model_max_size: int | None
) -> list[Image.Image]:
    """
    Resized frames for processing by the segmentation model.

    Args:
        video_frames: List of original video frames
        model_max_size: Maximum dimension for resizing

    Returns:
        List of resized PIL Image objects
    """
    return [
        ImageProcessor.resize_image(frame, model_max_size) for frame in video_frames
    ]


# --- Segmentation Functions ---


def segmentation_pipeline(
    model_name: str,
    model_type: str,
    model_device: str | None = None,
    model_num_workers: int = 1,
) -> object:
    """
    Segmentation pipeline for processing images.

    Args:
        model_name: Name of the model to use
        model_type: Type of the model
        model_device: Device to run the model on
        model_num_workers: Number of workers for the pipeline

    Returns:
        Segmentation pipeline object
    """
    model_config = ModelConfig(
        name=model_name,
        model_type=model_type,
        device=model_device,
        num_workers=model_num_workers,
    )

    return SegmentationProcessor.create_pipeline(model_config)


def segmentation_results(
    resized_frames: list[Image.Image], segmentation_pipeline: object
) -> list[dict[str, object]]:
    """
    Segmentation results for the frames.

    Args:
        resized_frames: List of resized frames to process
        segmentation_pipeline: Pipeline for segmentation

    Returns:
        List of segmentation results
    """
    return SegmentationProcessor.process_batch(resized_frames, segmentation_pipeline)


def segmentation_maps(
    segmentation_results: list[dict[str, object]],
) -> list[np.ndarray]:
    """
    Segmentation maps extracted from the results.

    Args:
        segmentation_results: List of segmentation results

    Returns:
        List of segmentation maps
    """
    return SegmentationProcessor.extract_segmentation_maps(segmentation_results)


def segmentation_metadata(
    segmentation_results: list[dict[str, object]],
) -> dict[str, object]:
    """
    Metadata from the segmentation results.

    Args:
        segmentation_results: List of segmentation results

    Returns:
        Dictionary of metadata
    """
    return SegmentationProcessor.extract_metadata(segmentation_results)


# --- Dataset and Storage Functions ---


def segmentation_dataset(
    segmentation_maps: list[np.ndarray],
    video_metadata: dict[str, object],
    frame_indices: list[int],
    model_metadata: dict[str, object],
    segmentation_metadata: dict[str, object],
) -> xr.Dataset:
    """
    XArray Dataset containing the segmentation data.

    Args:
        segmentation_maps: List of segmentation maps
        video_metadata: Metadata from the video
        frame_indices: Indices of the extracted frames
        model_metadata: Metadata about the model
        segmentation_metadata: Metadata from the segmentation

    Returns:
        XArray Dataset with segmentation data
    """
    return DatasetBuilder.create_video_dataset(
        segmentation_maps,
        video_metadata,
        frame_indices,
        model_metadata,
        segmentation_metadata,
    )


def saved_segmentation(segmentation_dataset: xr.Dataset, output_path: str) -> Path:
    """
    Save the segmentation dataset to file.

    Args:
        segmentation_dataset: XArray Dataset with segmentation data
        output_path: Path to save the dataset

    Returns:
        Path to the saved dataset
    """
    return DatasetBuilder.save_segmentation(segmentation_dataset, Path(output_path))


def analysis_path(segmentation_dataset: xr.Dataset, output_path: str) -> Path:
    """
    Analyze and save the segmentation results.

    Args:
        segmentation_dataset: XArray Dataset with segmentation data
        output_path: Path to save the analysis

    Returns:
        Path to the saved analysis
    """
    return SegmentationAnalyzer.analyze_segmentation_dataset(
        segmentation_dataset, Path(output_path)
    )


# --- Visualization Functions ---


def overlay_images(
    video_frames: list[Image.Image],
    segmentation_maps: list[np.ndarray],
    segmentation_metadata: dict[str, object],
) -> list[np.ndarray]:
    """
    Create overlay visualizations of the segmentation.

    Args:
        video_frames: Original video frames
        segmentation_maps: Segmentation maps
        segmentation_metadata: Metadata including palette

    Returns:
        List of overlay images
    """
    # Convert PIL Images to numpy arrays
    np_frames = [np.array(frame) for frame in video_frames]

    # Get palette from metadata
    palette = segmentation_metadata.get("palette")

    # Create overlays
    return VisualizationHandler.visualize_segmentation(
        np_frames, segmentation_maps, palette, colored_only=False, alpha=0.5
    )


def save_overlay_video(
    overlay_images: list[np.ndarray],
    video_metadata: dict[str, object],
    output_path: str,
) -> Path:
    """
    Save overlay images as a video.

    Args:
        overlay_images: List of overlay images
        video_metadata: Metadata from the original video
        output_path: Path to save the video

    Returns:
        Path to the saved video
    """
    import cv2

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine video properties
    height, width = overlay_images[0].shape[:2]
    fps = video_metadata.get("fps", 30)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    # Write frames
    for image in overlay_images:
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(bgr_image)

    # Release resources
    out.release()

    return output_file


def overlay_path(
    overlay_images: list[np.ndarray],
    video_metadata: dict[str, object],
    output_path: str,
) -> Path:
    """
    Save the overlay visualization.

    Args:
        overlay_images: List of overlay images
        video_metadata: Metadata from the original video
        output_path: Path to save the visualization

    Returns:
        Path to the saved visualization
    """
    return save_overlay_video(overlay_images, video_metadata, output_path)


# --- Image Processing Functions ---


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from file.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image object
    """
    return ImageProcessor.load_image(Path(image_path))


def resized_image(image: Image.Image, model_max_size: int | None) -> Image.Image:
    """
    Resize image for processing by the segmentation model.

    Args:
        image: Original image
        model_max_size: Maximum dimension for resizing

    Returns:
        Resized PIL Image object
    """
    return ImageProcessor.resize_image(image, model_max_size)


def image_segmentation_result(
    resized_image: Image.Image, segmentation_pipeline: object
) -> dict[str, object]:
    """
    Segmentation result for the image.

    Args:
        resized_image: Resized image to process
        segmentation_pipeline: Pipeline for segmentation

    Returns:
        Segmentation result
    """
    return SegmentationProcessor.process_image(resized_image, segmentation_pipeline)


def image_segmentation_map(image_segmentation_result: dict[str, object]) -> np.ndarray:
    """
    Segmentation map extracted from the result.

    Args:
        image_segmentation_result: Segmentation result

    Returns:
        Segmentation map
    """
    return image_segmentation_result["seg_map"]


def image_segmentation_metadata(
    image_segmentation_result: dict[str, object],
) -> dict[str, object]:
    """
    Metadata from the segmentation result.

    Args:
        image_segmentation_result: Segmentation result

    Returns:
        Dictionary of metadata
    """
    return {
        "label2id": image_segmentation_result.get("label2id", {}),
        "id2label": image_segmentation_result.get("id2label", {}),
        "palette": image_segmentation_result.get("palette", None),
    }


def image_dataset(
    image_segmentation_map: np.ndarray,
    model_metadata: dict[str, object],
    image_segmentation_metadata: dict[str, object],
) -> xr.Dataset:
    """
    XArray Dataset containing the image segmentation data.

    Args:
        image_segmentation_map: Segmentation map
        model_metadata: Metadata about the model
        image_segmentation_metadata: Metadata from the segmentation

    Returns:
        XArray Dataset with segmentation data
    """
    return DatasetBuilder.create_image_dataset(
        image_segmentation_map, model_metadata, image_segmentation_metadata
    )
