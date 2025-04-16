"""
This module provides a class for handling file operations related to segmentation data.

It includes functionalities for verifying the integrity of video files and data files,
and checking the validity of analysis files.

Classes:
    FileHandler: A class for handling file operations related to segmentation data and metadata.
"""

import json
from pathlib import Path
from typing import Any, Dict, Union

import cv2
import pandas as pd
import xarray as xr
from loguru import logger

from .config import Config
from .storage_adapter import ZarrSegmentationStorage


class FileHandler:
    """
    A class for handling file operations related to segmentation data and metadata.

    This class provides methods for verifying files and checking their integrity.

    Methods:
        verify_zarr_file: Verifies the integrity of a Zarr segmentation file.
        verify_video_file: Verifies the integrity of a video file.
        verify_analysis_files: Verifies the analysis files for counts and percentages.
        verify_parquet_file: Verifies the integrity of a Parquet analysis file.
    """

    @staticmethod
    def verify_zarr_file(file_path: Path, config: Config) -> bool:
        """
        Verifies the integrity of a Zarr segmentation file.

        Args:
            file_path (Path): Path to the Zarr file.
            config (Config): Configuration object for comparison.

        Returns:
            bool: True if the Zarr file is valid and up-to-date, False otherwise.
        """
        try:
            # Ensure file path has .zarr extension
            zarr_path = file_path.with_suffix('.zarr')
            if not zarr_path.exists():
                logger.warning(f"Zarr file at {zarr_path} does not exist")
                return False

            # Load the dataset
            storage = ZarrSegmentationStorage()
            dataset, metadata = storage.load_segmentation_data(zarr_path)

            # Check metadata
            if metadata.get("frame_step") != config.frame_step:
                logger.warning(
                    f"Zarr file frame step ({metadata.get('frame_step')}) does not match current config ({config.frame_step})"
                )
                return False

            # Check for segmentation data
            if "segmentation" not in dataset:
                logger.warning(f"Zarr file at {zarr_path} is missing segmentation data")
                return False

            # Check data integrity
            segmentation_data = dataset.segmentation
            if segmentation_data.size == 0:
                logger.warning(f"Zarr file at {zarr_path} contains no segmentation data")
                return False

            # Check data shape consistency
            if segmentation_data.dims != ('time', 'y', 'x'):
                logger.warning(f"Zarr file at {zarr_path} has unexpected dimensions")
                return False

            logger.debug(f"Zarr file at {zarr_path} is valid and up-to-date")
            return True
        except Exception as e:
            logger.error(f"Error verifying Zarr file at {file_path}: {str(e)}")
            return False

    @staticmethod
    def verify_video_file(file_path: Path) -> bool:
        """
        Verifies the integrity of a video file.

        Args:
            file_path (Path): Path to the video file.

        Returns:
            bool: True if the video file is valid and up-to-date, False otherwise.
        """
        try:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                logger.warning(f"Unable to open video file at {file_path}")
                return False

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, first_frame = cap.read()
            if not ret:
                logger.warning(
                    f"Unable to read first frame from video file at {file_path}"
                )
                return False

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            ret, last_frame = cap.read()
            if not ret:
                logger.warning(
                    f"Unable to read last frame from video file at {file_path}"
                )
                return False

            cap.release()
            logger.debug(f"Video file at {file_path} is valid and up-to-date")
            return True
        except Exception as e:
            logger.error(f"Error verifying video file at {file_path}: {str(e)}")
            return False

    @staticmethod
    def verify_analysis_files(counts_file: Path, percentages_file: Path) -> bool:
        """
        Verifies the analysis files for counts and percentages.

        Args:
            counts_file (Path): Path to the counts file.
            percentages_file (Path): Path to the percentages file.

        Returns:
            bool: True if the analysis files are valid, False otherwise.
        """
        try:
            if counts_file.stat().st_size == 0 or percentages_file.stat().st_size == 0:
                logger.info("One or both analysis files are empty")
                return False
            return True
        except Exception as e:
            logger.error(f"Error verifying analysis files: {str(e)}")
            return False

    @staticmethod
    def verify_parquet_file(file_path: Path) -> bool:
        """
        Verifies the integrity of a Parquet analysis file.

        Args:
            file_path (Path): Path to the Parquet file.

        Returns:
            bool: True if the Parquet file is valid, False otherwise.
        """
        try:
            # Ensure file path has .parquet extension
            parquet_path = file_path.with_suffix('.parquet')
            if not parquet_path.exists():
                logger.warning(f"Parquet file at {parquet_path} does not exist")
                return False

            # Try to load the DataFrame
            df = pd.read_parquet(parquet_path)
            
            # Check for required columns
            required_columns = {"category_id", "pixel_count", "percentage"}
            if not required_columns.issubset(set(df.columns)):
                logger.warning(f"Parquet file at {parquet_path} is missing required columns")
                return False
                
            # Ensure DataFrame has data
            if df.empty:
                logger.warning(f"Parquet file at {parquet_path} contains no data")
                return False
                
            logger.debug(f"Parquet file at {parquet_path} is valid")
            return True
        except Exception as e:
            logger.error(f"Error verifying Parquet file at {file_path}: {str(e)}")
            return False
