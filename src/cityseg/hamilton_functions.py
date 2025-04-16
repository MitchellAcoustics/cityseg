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

from .config import ModelConfig, Config
from .pipeline import create_segmentation_pipeline
from .video_resource import VideoResource
from .storage_adapter import ZarrSegmentationStorage, ParquetAnalysisStorage
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
    with VideoResource(Path(video_path)) as resource:
        return resource.get_metadata()


def frame_indices(frame_count: int, frame_step: int) -> List[int]:
    """
    Frame indices to extract from video based on frame step.
    
    Args:
        frame_count: Total number of frames in the video
        frame_step: Interval between frames to sample
        
    Returns:
        List of frame indices to extract
    """
    return list(range(0, frame_count, frame_step))


def video_frames(video_path: str, frame_indices: List[int]) -> List[Image.Image]:
    """
    Video frames extracted at the specified indices.
    
    Args:
        video_path: Path to the video file
        frame_indices: List of frame indices to extract
        
    Returns:
        List of PIL Image objects corresponding to the requested frames
    """
    with VideoResource(Path(video_path)) as resource:
        return resource.get_frame_batch(frame_indices)


def segmentation_pipeline(model_config: Dict[str, Any]) -> Any:
    """
    Segmentation pipeline initialized with model configuration.
    
    Args:
        model_config: Dictionary containing model configuration parameters
        
    Returns:
        Initialized segmentation pipeline ready for inference
    """
    config = ModelConfig(
        name=model_config["name"],
        model_type=model_config.get("model_type"),
        max_size=model_config.get("max_size"),
        device=model_config.get("device"),
        num_workers=model_config.get("num_workers", 1)
    )
    return create_segmentation_pipeline(config)


def segmentation_results(frames: List[Image.Image], pipeline: Any) -> List[Dict[str, Any]]:
    """
    Segmentation results for the input frames.
    
    Args:
        frames: List of frames to segment
        pipeline: Segmentation pipeline to use
        
    Returns:
        List of segmentation results for each frame
    """
    return pipeline(frames)


def segmentation_maps(results: List[Dict[str, Any]]) -> List[np.ndarray]:
    """
    Segmentation maps extracted from segmentation results.
    
    Args:
        results: List of segmentation results from the pipeline
        
    Returns:
        List of segmentation maps as numpy arrays
    """
    return [result["seg_map"] for result in results]


def segmentation_metadata(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Metadata from segmentation results including labels and palette.
    
    Args:
        results: List of segmentation results from the pipeline
        
    Returns:
        Dictionary containing label mappings and color palette
    """
    if not results:
        return {}
    
    result = results[0]
    return {
        "label2id": result.get("label2id", {}),
        "id2label": result.get("id2label", {}),
        "palette": result.get("palette", None)
    }


def segmentation_dataset(
    segmentation_maps: List[np.ndarray],
    video_metadata: Dict[str, Any],
    frame_indices: List[int],
    model_metadata: Dict[str, Any],
    segmentation_metadata: Dict[str, Any]
) -> xr.Dataset:
    """
    Xarray dataset containing segmentation results and metadata.
    
    Args:
        segmentation_maps: List of segmentation maps
        video_metadata: Metadata about the source video
        frame_indices: Indices of frames that were processed
        model_metadata: Information about the model used
        segmentation_metadata: Additional metadata from segmentation
        
    Returns:
        Xarray dataset with segmentation data and metadata
    """
    # Stack segmentation maps into a 3D array
    segmentation_array = np.stack(segmentation_maps)
    
    # Create time coordinates based on frame indices and fps
    time_coords = np.array(frame_indices) / video_metadata['fps'] if frame_indices else np.array([])
    
    # Create xarray DataArray with named dimensions
    segmentation_data = xr.DataArray(
        segmentation_array,
        dims=["time", "y", "x"],
        coords={
            "time": time_coords,
            "y": np.arange(video_metadata['height']),
            "x": np.arange(video_metadata['width'])
        }
    )
    
    # Combine all metadata
    attrs = {
        "model_name": model_metadata.get("name", ""),
        "model_type": model_metadata.get("model_type", ""),
        "fps": video_metadata['fps'],
        "frame_step": video_metadata.get('frame_step', 1),
        "original_width": video_metadata['width'],
        "original_height": video_metadata['height'],
        "codec": video_metadata.get('codec', None)
    }
    
    # Add segmentation metadata if available
    if segmentation_metadata:
        palette = segmentation_metadata.get('palette')
        if palette is not None:
            # Convert numpy array to list for serialization if needed
            attrs['palette'] = palette.tolist() if isinstance(palette, np.ndarray) else palette
            
        if 'id2label' in segmentation_metadata:
            attrs['id2label'] = segmentation_metadata['id2label']
    
    # Create the dataset
    dataset = xr.Dataset(
        data_vars={"segmentation": segmentation_data},
        attrs=attrs
    )
    
    return dataset


def saved_segmentation_path(
    dataset: xr.Dataset, 
    output_path: Union[str, Path]
) -> Path:
    """
    Path to the saved segmentation dataset.
    
    Args:
        dataset: Xarray dataset to save
        output_path: Base path for output files
        
    Returns:
        Path to the saved Zarr dataset
    """
    storage = ZarrSegmentationStorage()
    path = Path(output_path) if isinstance(output_path, str) else output_path
    return storage.save_segmentation_data(
        dataset,
        dict(dataset.attrs),
        path.with_name(f"{path.stem}_segmentation")
    )


def saved_analysis_path(
    dataset: xr.Dataset, 
    output_path: Union[str, Path]
) -> Path:
    """
    Path to the saved analysis data.
    
    Args:
        dataset: Xarray dataset to analyze
        output_path: Base path for output files
        
    Returns:
        Path to the saved analysis file
    """
    storage = ParquetAnalysisStorage()
    path = Path(output_path) if isinstance(output_path, str) else output_path
    return storage.save_video_analysis(
        dataset,
        dict(dataset.attrs),
        path.with_name(f"{path.stem}_analysis")
    )


# --- Image Processing Functions ---

def image_data(image_path: str, max_size: Optional[int] = None) -> Image.Image:
    """
    Image loaded from path, optionally resized.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum size for the longest dimension (if specified)
        
    Returns:
        Loaded PIL Image
    """
    image = Image.open(Path(image_path)).convert("RGB")
    
    if max_size:
        image.thumbnail((max_size, max_size))
        
    return image


def image_segmentation_result(image: Image.Image, pipeline: Any) -> Dict[str, Any]:
    """
    Segmentation result for a single image.
    
    Args:
        image: PIL Image to segment
        pipeline: Segmentation pipeline to use
        
    Returns:
        Dictionary containing segmentation result
    """
    return pipeline([image])[0]


def colored_segmentation(
    image: Union[np.ndarray, Image.Image],
    segmentation_map: np.ndarray,
    palette: np.ndarray
) -> np.ndarray:
    """
    Colored segmentation visualization.
    
    Args:
        image: Original image
        segmentation_map: Segmentation map
        palette: Color palette for visualization
        
    Returns:
        Colored segmentation as a numpy array
    """
    img_array = np.array(image) if isinstance(image, Image.Image) else image
    visualizer = VisualizationHandler()
    return visualizer.visualize_segmentation(
        img_array, segmentation_map, palette, colored_only=True
    )


def segmentation_overlay(
    image: Union[np.ndarray, Image.Image],
    segmentation_map: np.ndarray,
    palette: np.ndarray
) -> np.ndarray:
    """
    Segmentation overlay on the original image.
    
    Args:
        image: Original image
        segmentation_map: Segmentation map
        palette: Color palette for visualization
        
    Returns:
        Overlay visualization as a numpy array
    """
    img_array = np.array(image) if isinstance(image, Image.Image) else image
    visualizer = VisualizationHandler()
    return visualizer.visualize_segmentation(
        img_array, segmentation_map, palette, colored_only=False
    )


def saved_visualization_path(
    visualization: np.ndarray,
    output_path: Union[str, Path],
    suffix: str
) -> Path:
    """
    Path to the saved visualization image.
    
    Args:
        visualization: Visualization data to save
        output_path: Base path for output files
        suffix: Suffix to add to the filename
        
    Returns:
        Path to the saved image file
    """
    path = Path(output_path) if isinstance(output_path, str) else output_path
    save_path = path.with_name(f"{path.stem}_{suffix}.png")
    Image.fromarray(visualization).save(save_path)
    return save_path


def category_analysis(
    segmentation_map: np.ndarray,
    num_categories: int
) -> Dict[int, Tuple[int, float]]:
    """
    Analysis of category distribution in segmentation map.
    
    Args:
        segmentation_map: Segmentation map to analyze
        num_categories: Number of categories in the segmentation
        
    Returns:
        Dictionary mapping category IDs to (pixel_count, percentage)
    """
    analyzer = SegmentationAnalyzer()
    return analyzer.analyze_segmentation_map(segmentation_map, num_categories)


def saved_category_analysis_path(
    analysis: Dict[int, Tuple[int, float]],
    output_path: Union[str, Path]
) -> Path:
    """
    Path to the saved category analysis file.
    
    Args:
        analysis: Analysis results
        output_path: Base path for output files
        
    Returns:
        Path to the saved analysis file
    """
    # Extract counts and percentages
    counts = {category_id: count for category_id, (count, _) in analysis.items()}
    percentages = {category_id: percentage for category_id, (_, percentage) in analysis.items()}
    
    # Save using ParquetAnalysisStorage
    storage = ParquetAnalysisStorage()
    path = Path(output_path) if isinstance(output_path, str) else output_path
    return storage.save_category_analysis(
        counts,
        percentages,
        path.with_name(f"{path.stem}_category_analysis")
    )