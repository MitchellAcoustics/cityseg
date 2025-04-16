"""
This module provides Hamilton-style functions for the CitySeg pipeline.

It implements functions that follow Hamilton's conventions and best practices
while leveraging the underlying CitySeg classes. These functions can be used
to create data flow pipelines with Hamilton.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from PIL import Image
from loguru import logger
from hamilton import function_modifiers as fm

from .config import ModelConfig, Config
from .pipeline import create_segmentation_pipeline
from .video_processor import VideoProcessor
from .image_processor import ImageProcessor
from .segmentation_processor import SegmentationProcessor
from .dataset_builder import DatasetBuilder
from .segmentation_analyzer import SegmentationAnalyzer
from .visualization_handler import VisualizationHandler


# --- Video Processing Functions ---

def video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Video metadata extracted from the video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Metadata about the video including dimensions, frame count, and fps
    """
    return VideoProcessor.get_metadata(Path(video_path))


@fm.config.when(source="video_metadata")
def frame_count(video_metadata: Dict[str, Any]) -> int:
    """
    Number of frames in the video.
    
    Args:
        video_metadata: Metadata dictionary from the video
        
    Returns:
        Total number of frames in the video
    """
    return video_metadata["frame_count"]


@fm.config.when(source="video_metadata")
def fps(video_metadata: Dict[str, Any]) -> float:
    """
    Frames per second of the video.
    
    Args:
        video_metadata: Metadata dictionary from the video
        
    Returns:
        Video frame rate in frames per second
    """
    return video_metadata["fps"]


@fm.config.when(source="video_metadata")
def video_dimensions(video_metadata: Dict[str, Any]) -> Dict[str, int]:
    """
    Dimensions of the video frames.
    
    Args:
        video_metadata: Metadata dictionary from the video
        
    Returns:
        Dictionary with width and height of the video
    """
    return {
        "width": video_metadata["width"], 
        "height": video_metadata["height"]
    }


def video_frame_indices(frame_count: int, frame_step: int) -> List[int]:
    """
    Frame indices to extract based on frame step.
    
    Args:
        frame_count: Total number of frames in the video
        frame_step: Interval between frames to sample
        
    Returns:
        List of frame indices to extract
    """
    return VideoProcessor.get_frame_indices(frame_count, frame_step)


@fm.config.when(transform=fm.parametrized)
def video_frames(video_path: str, frame_indices: List[int]) -> List[Image.Image]:
    """
    Video frames extracted at the specified indices.
    
    Args:
        video_path: Path to the video file
        frame_indices: List of frame indices to extract
        
    Returns:
        List of PIL Image objects corresponding to the requested frames
    """
    return VideoProcessor.get_frames(Path(video_path), frame_indices)


# --- Image Processing Functions ---

@fm.config.when(transform=fm.parametrized)
def image_data(image_path: str, max_size: Optional[int] = None) -> Image.Image:
    """
    Image loaded from path, optionally resized.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum size for the longest dimension (if specified)
        
    Returns:
        Loaded PIL Image
    """
    image = ImageProcessor.load_image(Path(image_path))
    if max_size:
        image = ImageProcessor.resize_image(image, max_size)
    return image


# --- Segmentation Functions ---

@fm.config.when_in(["model_name", "model_type", "model_device", "model_max_size", "model_num_workers"])
def pipeline(
    model_name: str,
    model_type: Optional[str] = None,
    model_device: Optional[str] = None,
    model_max_size: Optional[int] = None,
    model_num_workers: Optional[int] = None
) -> Any:
    """
    Segmentation pipeline ready for inference.
    
    Args:
        model_name: Name of the model to use
        model_type: Type of the model (e.g., "segformer")
        model_device: Device to run the model on (e.g., "cuda", "cpu")
        model_max_size: Maximum size for image processing
        model_num_workers: Number of workers for processing
        
    Returns:
        Configured segmentation pipeline
    """
    config = ModelConfig(
        name=model_name,
        model_type=model_type,
        device=model_device,
        max_size=model_max_size,
        num_workers=model_num_workers
    )
    return SegmentationProcessor.create_pipeline(config)


def batch_segmentation_results(video_frames: List[Image.Image], pipeline: Any) -> List[Dict[str, Any]]:
    """
    Segmentation results for a batch of video frames.
    
    Args:
        video_frames: List of frames to process
        pipeline: Segmentation pipeline
        
    Returns:
        List of segmentation results
    """
    return SegmentationProcessor.process_batch(video_frames, pipeline)


def image_segmentation_result(image_data: Image.Image, pipeline: Any) -> Dict[str, Any]:
    """
    Segmentation result for a single image.
    
    Args:
        image_data: Image to process
        pipeline: Segmentation pipeline
        
    Returns:
        Segmentation result dictionary
    """
    return SegmentationProcessor.process_image(image_data, pipeline)


def segmentation_maps(batch_segmentation_results: List[Dict[str, Any]]) -> List[np.ndarray]:
    """
    Segmentation maps extracted from results.
    
    Args:
        batch_segmentation_results: Segmentation results from pipeline
        
    Returns:
        List of segmentation maps as numpy arrays
    """
    return SegmentationProcessor.extract_segmentation_maps(batch_segmentation_results)


def segmentation_map(image_segmentation_result: Dict[str, Any]) -> np.ndarray:
    """
    Single segmentation map from an image result.
    
    Args:
        image_segmentation_result: Segmentation result for a single image
        
    Returns:
        Segmentation map as a numpy array
    """
    return image_segmentation_result["seg_map"]


def segmentation_metadata(batch_segmentation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Metadata extracted from segmentation results.
    
    Args:
        batch_segmentation_results: Segmentation results from pipeline
        
    Returns:
        Dictionary with label mappings and palette
    """
    return SegmentationProcessor.extract_metadata(batch_segmentation_results)


def single_segmentation_metadata(image_segmentation_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Metadata extracted from a single segmentation result.
    
    Args:
        image_segmentation_result: Segmentation result for a single image
        
    Returns:
        Dictionary with label mappings and palette
    """
    return {
        "label2id": image_segmentation_result.get("label2id", {}),
        "id2label": image_segmentation_result.get("id2label", {}),
        "palette": image_segmentation_result.get("palette", None)
    }


# --- Dataset Functions ---

def video_segmentation_dataset(
    segmentation_maps: List[np.ndarray],
    video_metadata: Dict[str, Any],
    video_frame_indices: List[int],
    model_metadata: Dict[str, Any],
    segmentation_metadata: Dict[str, Any]
) -> xr.Dataset:
    """
    Dataset containing video segmentation results.
    
    Args:
        segmentation_maps: List of segmentation maps
        video_metadata: Video metadata dictionary
        video_frame_indices: Indices of processed frames
        model_metadata: Model metadata
        segmentation_metadata: Segmentation metadata
        
    Returns:
        xarray Dataset with segmentation data and metadata
    """
    return DatasetBuilder.create_video_dataset(
        segmentation_maps,
        video_metadata,
        video_frame_indices,
        model_metadata,
        segmentation_metadata
    )


def image_segmentation_dataset(
    segmentation_map: np.ndarray,
    model_metadata: Dict[str, Any],
    single_segmentation_metadata: Dict[str, Any]
) -> xr.Dataset:
    """
    Dataset containing single image segmentation result.
    
    Args:
        segmentation_map: Segmentation map for the image
        model_metadata: Model metadata
        single_segmentation_metadata: Segmentation metadata
        
    Returns:
        xarray Dataset with segmentation data and metadata
    """
    return DatasetBuilder.create_image_dataset(
        segmentation_map,
        model_metadata,
        single_segmentation_metadata
    )


# --- Storage Functions ---

def saved_segmentation_path(
    video_segmentation_dataset: xr.Dataset, 
    output_path: str
) -> str:
    """
    Path to the saved video segmentation dataset.
    
    Args:
        video_segmentation_dataset: Dataset to save
        output_path: Base output path
        
    Returns:
        Path to the saved Zarr store
    """
    saved_path = DatasetBuilder.save_segmentation(
        video_segmentation_dataset, 
        Path(output_path)
    )
    return str(saved_path)


def saved_image_segmentation_path(
    image_segmentation_dataset: xr.Dataset, 
    output_path: str
) -> str:
    """
    Path to the saved image segmentation dataset.
    
    Args:
        image_segmentation_dataset: Dataset to save
        output_path: Base output path
        
    Returns:
        Path to the saved Zarr store
    """
    saved_path = DatasetBuilder.save_segmentation(
        image_segmentation_dataset, 
        Path(output_path)
    )
    return str(saved_path)


def saved_analysis_path(
    video_segmentation_dataset: xr.Dataset, 
    output_path: str
) -> str:
    """
    Path to the saved segmentation analysis.
    
    Args:
        video_segmentation_dataset: Dataset to analyze
        output_path: Base output path
        
    Returns:
        Path to the saved analysis file
    """
    saved_path = DatasetBuilder.save_analysis(
        video_segmentation_dataset, 
        Path(output_path)
    )
    return str(saved_path)


# --- Visualization Functions ---

def colored_segmentation(
    image_data: Image.Image,
    segmentation_map: np.ndarray,
    single_segmentation_metadata: Dict[str, Any]
) -> np.ndarray:
    """
    Colored segmentation visualization.
    
    Args:
        image_data: Original image
        segmentation_map: Segmentation map
        single_segmentation_metadata: Metadata containing palette
        
    Returns:
        Colored segmentation as numpy array
    """
    palette = single_segmentation_metadata.get("palette")
    visualizer = VisualizationHandler()
    return visualizer.visualize_segmentation(
        np.array(image_data), segmentation_map, palette, colored_only=True
    )


def segmentation_overlay(
    image_data: Image.Image,
    segmentation_map: np.ndarray,
    single_segmentation_metadata: Dict[str, Any]
) -> np.ndarray:
    """
    Segmentation overlay on the original image.
    
    Args:
        image_data: Original image
        segmentation_map: Segmentation map
        single_segmentation_metadata: Metadata containing palette
        
    Returns:
        Segmentation overlay as numpy array
    """
    palette = single_segmentation_metadata.get("palette")
    visualizer = VisualizationHandler()
    return visualizer.visualize_segmentation(
        np.array(image_data), segmentation_map, palette, colored_only=False
    )


def saved_visualization_path(
    visualization: np.ndarray,
    output_path: str,
    suffix: str
) -> str:
    """
    Path to the saved visualization.
    
    Args:
        visualization: Visualization array to save
        output_path: Base output path
        suffix: Suffix to add to the filename
        
    Returns:
        Path to the saved image
    """
    saved_path = ImageProcessor.save_image(
        visualization,
        Path(output_path).with_name(f"{Path(output_path).stem}_{suffix}.png")
    )
    return str(saved_path)


# --- Analysis Functions ---

def category_analysis(
    segmentation_map: np.ndarray,
    single_segmentation_metadata: Dict[str, Any]
) -> Dict[int, Tuple[int, float]]:
    """
    Analysis of category distribution in a segmentation map.
    
    Args:
        segmentation_map: Segmentation map to analyze
        single_segmentation_metadata: Metadata containing label info
        
    Returns:
        Dictionary mapping category IDs to (count, percentage)
    """
    num_categories = len(single_segmentation_metadata.get("id2label", {}))
    analyzer = SegmentationAnalyzer()
    return analyzer.analyze_segmentation_map(segmentation_map, num_categories)


def saved_category_analysis_path(
    category_analysis: Dict[int, Tuple[int, float]],
    output_path: str
) -> str:
    """
    Path to the saved category analysis.
    
    Args:
        category_analysis: Analysis results
        output_path: Base output path
        
    Returns:
        Path to the saved analysis
    """
    # Extract counts and percentages
    counts = {category_id: count for category_id, (count, _) in category_analysis.items()}
    percentages = {category_id: percentage for category_id, (_, percentage) in category_analysis.items()}
    
    storage = ParquetAnalysisStorage()
    saved_path = storage.save_category_analysis(
        counts,
        percentages,
        Path(output_path).with_name(f"{Path(output_path).stem}_category_analysis")
    )
    return str(saved_path)


# --- Result Collection Functions ---

def video_process_results(
    saved_segmentation_path: str,
    saved_analysis_path: Optional[str] = None
) -> Dict[str, str]:
    """
    Combined results of video processing.
    
    Args:
        saved_segmentation_path: Path to the saved segmentation data
        saved_analysis_path: Path to the saved analysis data
        
    Returns:
        Dictionary of result paths
    """
    results = {"segmentation_path": saved_segmentation_path}
    if saved_analysis_path:
        results["analysis_path"] = saved_analysis_path
    return results


def image_process_results(
    saved_image_segmentation_path: Optional[str] = None,
    saved_category_analysis_path: Optional[str] = None,
    saved_visualization_paths: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Combined results of image processing.
    
    Args:
        saved_image_segmentation_path: Path to the saved segmentation data
        saved_category_analysis_path: Path to the saved category analysis
        saved_visualization_paths: Dictionary of visualization paths
        
    Returns:
        Dictionary of result paths
    """
    results = {}
    if saved_image_segmentation_path:
        results["segmentation_path"] = saved_image_segmentation_path
    if saved_category_analysis_path:
        results["analysis_path"] = saved_category_analysis_path
    if saved_visualization_paths:
        results.update(saved_visualization_paths)
    return results