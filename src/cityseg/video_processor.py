"""
This module provides functionality for processing video files.

It encapsulates methods for extracting metadata and frames from video files,
with proper resource management using context managers.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from loguru import logger

from .video_resource import VideoResource


class VideoProcessor:
    """
    A class for processing video files and extracting frames.
    
    This class provides methods to extract metadata and frames from video files,
    efficiently managing resources through context managers.
    
    Methods:
        get_metadata: Extract metadata from a video file
        get_frame_indices: Generate frame indices based on frame count and step
        get_frames: Extract frames from a video at specified indices
    """
    
    @staticmethod
    def get_metadata(video_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing metadata like dimensions, frame count, fps
        """
        resource = VideoResource(video_path)
        return resource.get_metadata()
    
    @staticmethod
    def get_frame_indices(frame_count: int, frame_step: int) -> List[int]:
        """
        Generate frame indices based on frame count and step.
        
        Args:
            frame_count: Total number of frames in the video
            frame_step: Interval between frames to extract
            
        Returns:
            List of frame indices to extract
        """
        return list(range(0, frame_count, frame_step))
    
    @staticmethod
    def get_frames(video_path: Path, frame_indices: List[int]) -> List[Image.Image]:
        """
        Extract frames from a video at specified indices.
        
        Args:
            video_path: Path to the video file
            frame_indices: List of frame indices to extract
            
        Returns:
            List of PIL Image objects representing the extracted frames
        """
        resource = VideoResource(video_path)
        frames = resource.get_frame_batch(frame_indices)
        logger.debug(f"Extracted {len(frames)} frames from {video_path}")
        return frames