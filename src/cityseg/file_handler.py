"""
This module provides a class for handling file operations related to segmentation data.

It includes functionalities for saving and loading segmentation data in HDF files,
verifying the integrity of HDF and video files, and checking the validity of analysis files.

Classes:
    FileHandler: A class for handling file operations related to segmentation data and metadata.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import h5py
import numpy as np
from loguru import logger

from .config import Config


class FileHandler:
    """
    A class for handling file operations related to segmentation data and metadata.

    This class provides methods for saving and loading segmentation data in HDF files,
    verifying the integrity of HDF and video files, and checking analysis files.

    Methods:
        save_hdf_file: Saves segmentation data and metadata to an HDF file.
        load_hdf_file: Loads segmentation data and metadata from an HDF file.
        verify_hdf_file: Verifies the integrity of an HDF file.
        verify_video_file: Verifies the integrity of a video file.
        verify_analysis_files: Verifies the analysis files for counts and percentages.
    """

    @staticmethod
    def save_hdf_file(
        file_path: Path, segmentation_data: np.ndarray, metadata: Dict[str, Any]
    ) -> None:
        """
        Saves segmentation data and metadata to an HDF file.

        Args:
            file_path (Path): Path to the HDF file.
            segmentation_data (np.ndarray): Segmentation data to be saved.
            metadata (Dict[str, Any]): Metadata associated with the segmentation data.
        """
        with h5py.File(file_path, "w") as f:
            f.create_dataset("segmentation", data=segmentation_data, compression="gzip")
            if "palette" in metadata and isinstance(metadata["palette"], np.ndarray):
                metadata["palette"] = metadata["palette"].tolist()
            json_metadata = json.dumps(metadata)
            f.create_dataset("metadata", data=json_metadata)

    @staticmethod
    def load_hdf_file(file_path: Path) -> Tuple[h5py.File, Dict[str, Any]]:
        """
        Loads segmentation data and metadata from an HDF file.

        Args:
            file_path (Path): Path to the HDF file.

        Returns:
            Tuple[h5py.File, Dict[str, Any]]: Loaded HDF file and metadata.
        """
        hdf_file = h5py.File(file_path, "r")
        json_metadata = hdf_file["metadata"][()]
        metadata = json.loads(json_metadata)
        if "palette" in metadata:
            metadata["palette"] = np.array(metadata["palette"], np.uint8)
        return hdf_file, metadata

    @staticmethod
    def verify_hdf_file(file_path: Path, config: Config) -> bool:
        """
        Verifies the integrity of an HDF file.

        Args:
            file_path (Path): Path to the HDF file.
            config (Config): Configuration object for comparison.

        Returns:
            bool: True if the HDF file is valid and up-to-date, False otherwise.
        """
        try:
            with h5py.File(file_path, "r") as f:
                if "segmentation" not in f or "metadata" not in f:
                    logger.warning(
                        f"HDF file at {file_path} is missing required datasets"
                    )
                    return False

                json_metadata = f["metadata"][()]
                metadata = json.loads(json_metadata)

                if metadata.get("frame_step") != config.frame_step:
                    logger.warning(
                        f"HDF file frame step ({metadata.get('frame_step')}) does not match current config ({config.frame_step})"
                    )
                    return False

                segmentation_data = f["segmentation"]
                if len(segmentation_data) == 0:
                    logger.warning(
                        f"HDF file at {file_path} contains no segmentation data"
                    )
                    return False

                first_frame = segmentation_data[0]
                last_frame = segmentation_data[-1]
                if first_frame.shape != last_frame.shape:
                    logger.warning(
                        f"Inconsistent frame shapes in HDF file at {file_path}"
                    )
                    return False

            logger.debug(f"HDF file at {file_path} is valid and up-to-date")
            return True
        except Exception as e:
            logger.error(f"Error verifying HDF file at {file_path}: {str(e)}")
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
