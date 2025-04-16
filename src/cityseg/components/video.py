"""
This module provides functionality for processing video files and directories.

It includes classes and utilities for:
1. Video resource management with proper context handling (VideoResource)
2. Video processing, frame extraction, and metadata handling (VideoProcessor)
3. Iterating over video files in a directory (VideoFileIterator)
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import cv2
from PIL import Image
from loguru import logger


class VideoResource:
    """
    Resource manager for video operations.

    This class provides a context manager for video capture objects,
    ensuring proper resource cleanup and utilities for common operations.

    Attributes:
        video_path (str): Path to the video file.
    """

    def __init__(self, video_path: Path):
        """
        Initialize the video resource manager.

        Args:
            video_path (Path): Path to the video file.
        """
        self.video_path = str(video_path)
        self._cap = None

    def __enter__(self):
        """
        Enter the context manager and open the video file.

        Returns:
            cv2.VideoCapture: The opened video capture object.

        Raises:
            IOError: If the video file cannot be opened.
        """
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise IOError(f"Failed to open video: {self.video_path}")
        return self._cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and release the video resource.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get video metadata without keeping the resource open.

        Returns:
            Dict[str, Any]: Video metadata including frame count, fps, width, and height.
        """
        with self as cap:
            metadata = {
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "codec": self._get_codec(cap),
            }
        return metadata

    def _get_codec(self, cap) -> str:
        """
        Get the video codec as a string.

        Args:
            cap (cv2.VideoCapture): Video capture object.

        Returns:
            str: Video codec as a string.
        """
        codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_bytes = bytes(
            [
                codec_int & 0xFF,
                (codec_int >> 8) & 0xFF,
                (codec_int >> 16) & 0xFF,
                (codec_int >> 24) & 0xFF,
            ]
        )
        return codec_bytes.decode("ascii", errors="replace")

    def get_frame_batch(self, indices: List[int]) -> List[Image.Image]:
        """
        Get a batch of frames at the specified indices.

        Args:
            indices (List[int]): List of frame indices to retrieve.

        Returns:
            List[Image.Image]: List of PIL Image objects for the requested frames.
        """
        frames = []
        with self as cap:
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB for PIL
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb_frame))
                else:
                    logger.warning(f"Failed to read frame at index {idx}")

        return frames

    def get_frames_by_step(
        self, frame_step: int, max_frames: int = None
    ) -> List[Image.Image]:
        """
        Get frames from the video using a step size.

        Args:
            frame_step (int): Number of frames to skip between each captured frame.
            max_frames (int, optional): Maximum number of frames to capture.
                                        If None, all frames are captured.

        Returns:
            List[Image.Image]: List of PIL Image objects for the captured frames.
        """
        frames = []
        with self as cap:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            # Reset to beginning of video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while True:
                # Check if we've reached the maximum number of frames
                if max_frames is not None and len(frames) >= max_frames:
                    break

                # Read the current frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Only keep frames at the specified step
                if frame_count % frame_step == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb_frame))

                frame_count += 1

                # Skip frames if necessary
                if frame_step > 1:
                    # Skip to the next frame we want to capture
                    next_frame = ((frame_count // frame_step) + 1) * frame_step
                    if next_frame < total_frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
                        frame_count = next_frame

        return frames

    def get_frame_batch_generator(self, frame_step: int, batch_size: int):
        """
        Get a generator that yields batches of frames.

        Args:
            frame_step (int): Number of frames to skip between each captured frame.
            batch_size (int): Number of frames to include in each batch.

        Yields:
            List[Image.Image]: Batch of PIL Image objects.
        """
        with self as cap:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for start_idx in range(0, total_frames, frame_step * batch_size):
                batch_frames = []

                for i in range(batch_size):
                    frame_idx = start_idx + (i * frame_step)
                    if frame_idx >= total_frames:
                        break

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                    if ret:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        batch_frames.append(Image.fromarray(rgb_frame))

                if batch_frames:
                    yield batch_frames


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


class VideoFileIterator:
    """
    An iterator class for iterating over video files in a specified directory.

    This class retrieves and stores video files from the given input path and
    provides an iterator interface to access these files.

    Attributes:
        input_path (Path): The path to the directory containing video files.
        video_files (List[Path]): A list of video file paths found in the input directory.
    """

    def __init__(self, input_path: Path, ignore_files: Optional[List[str]] = None):
        """
        Initializes the VideoFileIterator with the specified input path.

        Args:
            input_path (Path): The path to the directory containing video files.
            ignore_files (Optional[List[str]]): List of filenames to ignore.
        """
        self.input_path = input_path
        self.ignore_files = ignore_files or []
        self.video_files = self._get_video_files()

    def _get_video_files(self) -> List[Path]:
        """
        Retrieves a list of video files from the input directory.

        Returns:
            List[Path]: A list of paths to the video files found in the input directory.
        """
        video_extensions = [".mp4", ".avi", ".mov"]
        video_files = [
            f
            for f in self.input_path.glob("*")
            if f.suffix.lower() in video_extensions and f.name not in self.ignore_files
        ]
        logger.info(f"Found {len(video_files)} video files in {self.input_path}")
        return list(video_files)

    def __iter__(self) -> Iterator[Path]:
        """
        Returns an iterator over the video files.

        Returns:
            Iterator[Path]: An iterator over the video file paths.
        """
        return iter(self.video_files)
