"""
This module provides a workflow implementation using Hamilton for the CitySeg pipeline.

It defines the functions and data flow for processing images and videos through
the segmentation pipeline, with proper caching and resource management.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import xarray as xr
from PIL import Image
from loguru import logger
from hamilton import driver
from hamilton.function_modifiers import extract_columns, parameterize, config

from .config import Config, ModelConfig
from .pipeline import create_segmentation_pipeline
from .video_resource import VideoResource
from .storage_adapter import ZarrSegmentationStorage, ParquetAnalysisStorage


# Hamilton workflow functions
def video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a video file.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        Dict[str, Any]: Dictionary containing video metadata.
    """
    video_resource = VideoResource(Path(video_path))
    return video_resource.get_metadata()


@extract_columns(dict_path='video_metadata')
def frame_count(frame_count: int) -> int:
    """
    Extract the frame count from video metadata.
    
    Args:
        frame_count (int): Frame count from the metadata.
        
    Returns:
        int: Frame count.
    """
    return frame_count


@extract_columns(dict_path='video_metadata')
def fps(fps: float) -> float:
    """
    Extract frames per second from video metadata.
    
    Args:
        fps (float): Frames per second from the metadata.
        
    Returns:
        float: Frames per second.
    """
    return fps


@extract_columns(dict_path='video_metadata')
def video_dimensions(width: int, height: int) -> Dict[str, int]:
    """
    Extract video dimensions from metadata.
    
    Args:
        width (int): Video width from the metadata.
        height (int): Video height from the metadata.
        
    Returns:
        Dict[str, int]: Dictionary with video dimensions.
    """
    return {"width": width, "height": height}


def frame_indices(frame_count: int, frame_step: int) -> List[int]:
    """
    Generate frame indices to process based on frame step.
    
    Args:
        frame_count (int): Total number of frames in the video.
        frame_step (int): Number of frames to skip between captures.
        
    Returns:
        List[int]: List of frame indices to process.
    """
    return list(range(0, frame_count, frame_step))


def video_frames(video_path: str, frame_indices: List[int]) -> List[Image.Image]:
    """
    Load frames from the video at specified indices.
    
    Args:
        video_path (str): Path to the video file.
        frame_indices (List[int]): List of frame indices to load.
        
    Returns:
        List[Image.Image]: List of loaded frames as PIL images.
    """
    video_resource = VideoResource(Path(video_path))
    return video_resource.get_frame_batch(frame_indices)


@config('model')
def segmentation_pipeline(
    model: Dict[str, Any]
) -> Any:
    """
    Create a segmentation pipeline from model configuration.
    
    Args:
        model (Dict[str, Any]): Model configuration dictionary.
        
    Returns:
        Any: Segmentation pipeline.
    """
    model_config = ModelConfig(
        name=model['name'],
        model_type=model.get('model_type', None),
        max_size=model.get('max_size', None),
        device=model.get('device', 'cuda')
    )
    return create_segmentation_pipeline(model_config)


def segmentation_maps(video_frames: List[Image.Image], segmentation_pipeline: Any) -> List[np.ndarray]:
    """
    Process frames through segmentation pipeline to get segmentation maps.
    
    Args:
        video_frames (List[Image.Image]): List of frames to process.
        segmentation_pipeline: Segmentation pipeline.
        
    Returns:
        List[np.ndarray]: Segmentation maps for each input frame.
    """
    results = segmentation_pipeline(video_frames)
    return [result["seg_map"] for result in results]


def segmentation_dataset(
    segmentation_maps: List[np.ndarray],
    video_metadata: Dict[str, Any],
    frame_indices: List[int],
    model: Dict[str, Any]
) -> xr.Dataset:
    """
    Create an xarray Dataset from segmentation maps and metadata.
    
    Args:
        segmentation_maps (List[np.ndarray]): List of segmentation maps.
        video_metadata (Dict[str, Any]): Video metadata.
        frame_indices (List[int]): List of frame indices.
        model (Dict[str, Any]): Model configuration.
        
    Returns:
        xr.Dataset: Dataset containing segmentation data and metadata.
    """
    # Stack segmentation maps into a 3D array
    segmentation_array = np.stack(segmentation_maps)
    
    # Create time coordinates based on frame indices and fps
    if len(frame_indices) > 0:
        time_coords = np.array(frame_indices) / video_metadata['fps']
    else:
        time_coords = np.array([])
    
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
    
    # Create dataset with metadata
    dataset = xr.Dataset(
        data_vars={"segmentation": segmentation_data},
        attrs={
            "model_name": model['name'],
            "model_type": model.get('model_type', None),
            "fps": video_metadata['fps'],
            "frame_step": len(frame_indices) / (video_metadata['frame_count'] or 1),
            "original_width": video_metadata['width'],
            "original_height": video_metadata['height'],
            "codec": video_metadata.get('codec', None)
        }
    )
    
    return dataset


def save_segmentation(
    segmentation_dataset: xr.Dataset,
    output_path: str
) -> str:
    """
    Save segmentation dataset to storage.
    
    Args:
        segmentation_dataset (xr.Dataset): Dataset to save.
        output_path (str): Path to save the dataset.
        
    Returns:
        str: Path to the saved dataset.
    """
    storage = ZarrSegmentationStorage()
    metadata = dict(segmentation_dataset.attrs)
    saved_path = storage.save_segmentation_data(
        segmentation_dataset,
        metadata,
        Path(output_path)
    )
    return str(saved_path)


def save_analysis(
    segmentation_dataset: xr.Dataset,
    output_path: str
) -> str:
    """
    Generate and save analysis of segmentation results.
    
    Args:
        segmentation_dataset (xr.Dataset): Dataset to analyze.
        output_path (str): Path to save the analysis.
        
    Returns:
        str: Path to the saved analysis.
    """
    storage = ParquetAnalysisStorage()
    saved_path = storage.save_video_analysis(
        segmentation_dataset,
        dict(segmentation_dataset.attrs),
        Path(output_path).with_name(f"{Path(output_path).stem}_analysis")
    )
    return str(saved_path)


class CitysegWorkflow:
    """
    Workflow manager for the CitySeg pipeline using Hamilton.
    
    This class provides methods for creating and executing the segmentation
    workflow with proper caching and resource management.
    """
    
    def __init__(self, config: Config, cache_dir: Optional[Path] = None):
        """
        Initialize the workflow manager.
        
        Args:
            config (Config): Configuration object.
            cache_dir (Optional[Path]): Path to the cache directory. If None, caching is disabled.
        """
        self.config = config
        self.cache_dir = cache_dir
        self._driver = self._create_driver()
    
    def _create_driver(self) -> driver.Driver:
        """
        Create a Hamilton driver with workflow functions.
        
        Returns:
            driver.Driver: Hamilton driver.
        """
        modules = [
            sys.modules[__name__]  # This module contains the workflow functions
        ]
        
        # Create appropriate adapter for caching if needed
        adapter = None
        if self.cache_dir is not None:
            try:
                from hamilton.experimental.h_cache import CacheManager
                
                # Setup cache manager
                cache_manager = CacheManager(
                    cache_dir=str(self.cache_dir),
                    eager_mode=False,  # Only cache when requested
                    strategy='overwrite'  # Overwrite existing cache
                )
                
                # Configure cache for specific functions
                cache_config = {
                    'segmentation_maps': True,  # Cache this function's outputs
                    'video_frames': True,       # Cache frames
                }
                
                adapter = cache_manager.build_cache_adapter(cache_config)
                logger.info(f"Caching enabled for workflow, using directory: {self.cache_dir}")
            except ImportError:
                logger.warning("Hamilton caching not available, proceeding without cache")
        
        return driver.Driver(modules, adapter=adapter)
    
    def process_video(self) -> Dict[str, Any]:
        """
        Process a video through the segmentation pipeline.
        
        Returns:
            Dict[str, Any]: Dictionary containing workflow results.
        """
        # Prepare inputs for the workflow
        inputs = {
            'video_path': str(self.config.input),
            'frame_step': self.config.frame_step,
            'model': self.config.model.to_dict(),
            'output_path': str(self.config.get_output_path())
        }
        
        # Define desired outputs
        outputs = [
            'segmentation_dataset',
            'save_segmentation',
            'save_analysis'
        ]
        
        # Execute the workflow
        logger.info(f"Processing video: {self.config.input}")
        result = self._driver.execute(outputs, inputs=inputs)
        logger.info(f"Processing complete, results saved to: {result['save_segmentation']}")
        
        return result
    
    def process_image(self) -> Dict[str, Any]:
        """
        Process an image through the segmentation pipeline.
        
        Returns:
            Dict[str, Any]: Dictionary containing workflow results.
        """
        # TODO: Implement image processing workflow
        raise NotImplementedError("Image processing workflow not yet implemented")
    
    def process(self) -> Dict[str, Any]:
        """
        Process the input based on its type.
        
        Returns:
            Dict[str, Any]: Dictionary containing workflow results.
        """
        from .config import InputType
        
        if self.config.input_type == InputType.SINGLE_VIDEO:
            return self.process_video()
        elif self.config.input_type == InputType.SINGLE_IMAGE:
            return self.process_image()
        elif self.config.input_type == InputType.DIRECTORY:
            # TODO: Implement directory processing workflow
            raise NotImplementedError("Directory processing workflow not yet implemented")
        else:
            raise ValueError(f"Unsupported input type: {self.config.input_type}")


def create_workflow(config: Config, cache_dir: Optional[Path] = None) -> CitysegWorkflow:
    """
    Create a CitySeg workflow for the given configuration.
    
    Args:
        config (Config): Configuration object.
        cache_dir (Optional[Path]): Path to the cache directory. If None, caching is disabled.
        
    Returns:
        CitysegWorkflow: Workflow manager.
    """
    return CitysegWorkflow(config, cache_dir)