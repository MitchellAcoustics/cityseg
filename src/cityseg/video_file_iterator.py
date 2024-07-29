"""
This module provides an iterator class for iterating over video files in a specified directory.

It retrieves and stores video files from the given input path and provides an iterator
interface to access these files.

Classes:
    VideoFileIterator: An iterator class for iterating over video files in a specified directory.
"""

from pathlib import Path
from typing import Iterator, List

from loguru import logger

from .utils import get_video_files


class VideoFileIterator:
    """
    An iterator class for iterating over video files in a specified directory.

    This class retrieves and stores video files from the given input path and
    provides an iterator interface to access these files.

    Attributes:
        input_path (Path): The path to the directory containing video files.
        video_files (List[Path]): A list of video file paths found in the input directory.
    """

    def __init__(self, input_path: Path):
        """
        Initializes the VideoFileIterator with the specified input path.

        Args:
            input_path (Path): The path to the directory containing video files.
        """
        self.input_path = input_path
        self.video_files = self._get_video_files()

    def _get_video_files(self) -> List[Path]:
        """
        Retrieves a list of video files from the input directory.

        This method uses the utility function `get_video_files` to find all video files
        in the specified input path and logs the number of files found.

        Returns:
            List[Path]: A list of paths to the video files found in the input directory.
        """
        video_files = get_video_files(self.input_path)
        logger.info(f"Found {len(video_files)} video files in {self.input_path}")
        return video_files

    def __iter__(self) -> Iterator[Path]:
        """
        Returns an iterator over the video files.

        This method allows the VideoFileIterator to be used in a for-loop or any context
        that requires an iterable.

        Returns:
            Iterator[Path]: An iterator over the video file paths.
        """
        return iter(self.video_files)
