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
        update_hdf_file: Updates an existing HDF file with new segmentation data.
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
            # Convert segmentation data to integer type before saving
            if not np.issubdtype(segmentation_data.dtype, np.integer):
                segmentation_data = np.round(segmentation_data).astype(np.int32)

            f.create_dataset("segmentation", data=segmentation_data, compression="gzip")
            if "palette" in metadata and isinstance(metadata["palette"], np.ndarray):
                metadata["palette"] = metadata["palette"].tolist()
            json_metadata = json.dumps(metadata)
            f.create_dataset("metadata", data=json_metadata)

    @staticmethod
    def update_hdf_file(
        file_path: Path,
        new_segmentation_data: np.ndarray,
        current_frame_count: int,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Updates an existing HDF file with new segmentation data incrementally.

        This method is optimized for batch processing by only adding the new data
        and updating metadata without rewriting the entire file.

        Args:
            file_path (Path): Path to the HDF file.
            new_segmentation_data (np.ndarray): New segmentation data to be added.
            current_frame_count (int): Total number of frames including the new data.
            metadata (Dict[str, Any]): Updated metadata associated with the segmentation data.
        """
        file_exists = file_path.exists()
        metadata_to_save = metadata.copy()

        # Handle palette conversion for metadata
        if "palette" in metadata_to_save and isinstance(
            metadata_to_save["palette"], np.ndarray
        ):
            metadata_to_save["palette"] = metadata_to_save["palette"].tolist()

        # Update the frame count in metadata
        metadata_to_save["frame_count"] = current_frame_count

        # Convert new segmentation data to integer type before saving
        if not np.issubdtype(new_segmentation_data.dtype, np.integer):
            new_segmentation_data = np.round(new_segmentation_data).astype(np.int32)

        if file_exists:
            try:
                with h5py.File(str(file_path), "a") as f:
                    # If the file exists but doesn't have the datasets yet, create them
                    if "segmentation" not in f:
                        f.create_dataset(
                            "segmentation",
                            data=new_segmentation_data,
                            maxshape=(
                                None,
                                *new_segmentation_data.shape[1:],
                            ),  # None allows unlimited growth
                            compression="gzip",
                        )
                    else:
                        # Get the current dataset
                        dset = f["segmentation"]

                        # Make sure we're working with a dataset, not a group or datatype
                        if isinstance(dset, h5py.Dataset):
                            old_size = dset.shape[0]
                            new_size = current_frame_count

                            # Check if the dataset needs resizing
                            if new_size > old_size:
                                # If the dataset has a fixed maxshape, recreate it with unlimited first dimension
                                if (
                                    dset.maxshape[0] is not None
                                    and new_size > dset.maxshape[0]
                                ):
                                    logger.debug(
                                        f"Dataset maxshape ({dset.maxshape[0]}) is too small for new size ({new_size}), recreating dataset"
                                    )

                                    # Read the existing data
                                    existing_data = dset[:]

                                    # Delete the existing dataset
                                    del f["segmentation"]

                                    # Create a new dataset with unlimited first dimension
                                    new_dset = f.create_dataset(
                                        "segmentation",
                                        shape=(old_size, *existing_data.shape[1:]),
                                        maxshape=(None, *existing_data.shape[1:]),
                                        compression="gzip",
                                    )

                                    # Copy the existing data
                                    new_dset[:old_size] = existing_data

                                    # Update our reference
                                    dset = new_dset

                                # Now resize the dataset (with proper maxshape)
                                dset.resize((new_size, *dset.shape[1:]))

                            # Add new data
                            dset[old_size:new_size] = new_segmentation_data
                        else:
                            logger.error("'segmentation' is not a Dataset in HDF file")
                            return

                    # Update metadata
                    if "metadata" in f:
                        del f["metadata"]

                    # Create metadata as a string
                    json_str = json.dumps(metadata_to_save)
                    f.create_dataset("metadata", data=json_str)
            except Exception as e:
                logger.error(f"Error updating HDF file: {str(e)}")
                # If update fails, try recreating the file
                logger.info("Attempting to recreate HDF file with all data")
                FileHandler.save_hdf_file(
                    file_path,
                    new_segmentation_data,  # This would lose previous data, but prevents total failure
                    metadata_to_save,
                )
        else:
            # If the file doesn't exist, create it
            with h5py.File(str(file_path), "w") as f:
                f.create_dataset(
                    "segmentation",
                    data=new_segmentation_data,
                    maxshape=(
                        None,
                        *new_segmentation_data.shape[1:],
                    ),  # None allows unlimited growth
                    compression="gzip",
                )
                # Create metadata as a string
                json_str = json.dumps(metadata_to_save)
                f.create_dataset("metadata", data=json_str)

        logger.debug(
            f"HDF file updated at {file_path} with {len(new_segmentation_data)} new frames"
        )

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
        if "palette" in metadata and isinstance(metadata["palette"], list):
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
