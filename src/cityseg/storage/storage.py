"""
This module provides storage adapters and file handling utilities for CitySeg.

It includes:
1. Abstract SegmentationStorage interface
2. ZarrSegmentationStorage adapter for segmentation data
3. ParquetAnalysisStorage adapter for analysis results
4. FileHandler for verifying files
5. StorageFactory for creating storage adapters
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from loguru import logger

from ..core import Config


class SegmentationStorage:
    """
    Abstract base class for segmentation data storage.

    This class defines the interface for segmentation storage adapters.
    """

    def save_segmentation_data(
        self,
        data: np.ndarray | xr.DataArray | xr.Dataset,
        metadata: dict[str, object],
        output_path: Path,
    ) -> Path:
        """
        Save segmentation data and metadata to storage.

        Args:
            data: Segmentation data.
            metadata (dict[str, object]): Metadata about the segmentation.
            output_path (Path): Path to save the data.

        Returns:
            Path: Path to the saved data.
        """
        raise NotImplementedError("Subclasses must implement save_segmentation_data")

    def load_segmentation_data(
        self, input_path: Path
    ) -> tuple[xr.Dataset, dict[str, object]]:
        """
        Load segmentation data and metadata from storage.

        Args:
            input_path (Path): Path to the saved data.

        Returns:
            tuple[xr.Dataset, dict[str, object]]: Tuple of segmentation data and metadata.
        """
        raise NotImplementedError("Subclasses must implement load_segmentation_data")

    def load_segmentation_batch(
        self, input_path: Path, start: int, end: int
    ) -> xr.Dataset:
        """
        Load a batch of segmentation data.

        Args:
            input_path (Path): Path to the saved data.
            start (int): Start index of the batch.
            end (int): End index of the batch.

        Returns:
            xr.Dataset: Batch of segmentation data.
        """
        raise NotImplementedError("Subclasses must implement load_segmentation_batch")


class ZarrSegmentationStorage(SegmentationStorage):
    """
    Segmentation storage adapter using Zarr format.

    This class handles storing and retrieving segmentation data using Zarr,
    which provides efficient chunked storage for multi-dimensional arrays.
    """

    def save_segmentation_data(
        self,
        data: np.ndarray | xr.DataArray | xr.Dataset,
        metadata: dict[str, object],
        output_path: Path,
    ) -> Path:
        """
        Save segmentation data and metadata to Zarr storage.

        Args:
            data: Segmentation data as numpy array or xarray DataArray/Dataset.
            metadata (dict[str, object]): Metadata about the segmentation.
            output_path (Path): Path to save the data.

        Returns:
            Path: Path to the saved Zarr store.
        """
        # Ensure output path has .zarr extension
        zarr_path = output_path.with_suffix(".zarr")

        # Convert numpy arrays to xarray if needed
        if isinstance(data, np.ndarray):
            if data.ndim == 3:
                # Assuming shape (frames, height, width)
                frames, height, width = data.shape
                data = xr.DataArray(
                    data,
                    dims=["time", "y", "x"],
                    coords={
                        "time": np.arange(frames),
                        "y": np.arange(height),
                        "x": np.arange(width),
                    },
                )
            else:
                raise ValueError(f"Unsupported array shape: {data.shape}")

        # If it's a DataArray, convert to Dataset
        if isinstance(data, xr.DataArray):
            data = xr.Dataset({"segmentation": data})

        # Add metadata as attributes
        if isinstance(data, xr.Dataset):
            for key, value in metadata.items():
                # Convert numpy arrays to lists for JSON serialization
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                # Convert dictionaries with integer keys to string keys for JSON serialization
                elif isinstance(value, dict) and any(
                    isinstance(k, int) for k in value.keys()
                ):
                    value = {str(k): v for k, v in value.items()}

                data.attrs[key] = value

        # Determine optimal chunking for video data
        if isinstance(data, xr.Dataset) and "segmentation" in data:
            time_chunks = min(100, data.sizes.get("time", 1))
            y_chunks = min(256, data.sizes.get("y", 1))
            x_chunks = min(256, data.sizes.get("x", 1))
            chunked_data = data.chunk(
                {"time": time_chunks, "y": y_chunks, "x": x_chunks}
            )
        else:
            chunked_data = data

        # Save to Zarr format with compression
        encoding = {"segmentation": {"compressor": zarr.Blosc(cname="zstd", clevel=3)}}
        chunked_data.to_zarr(zarr_path, mode="w", encoding=encoding)

        logger.info(f"Saved segmentation data to {zarr_path}")
        return zarr_path

    def load_segmentation_data(
        self, input_path: Path
    ) -> tuple[xr.Dataset, dict[str, object]]:
        """
        Load segmentation data and metadata from Zarr storage.

        Args:
            input_path (Path): Path to the Zarr store.

        Returns:
            tuple[xr.Dataset, dict[str, object]]: Tuple of segmentation dataset and metadata.
        """
        # Ensure input path has .zarr extension
        zarr_path = input_path.with_suffix(".zarr")

        # Load the dataset
        dataset = xr.open_zarr(zarr_path)

        # Extract metadata from attributes
        metadata = dict(dataset.attrs)

        return dataset, metadata

    def load_segmentation_batch(
        self, input_path: Path, start: int, end: int
    ) -> xr.Dataset:
        """
        Load a batch of segmentation data from Zarr storage.

        Args:
            input_path (Path): Path to the Zarr store.
            start (int): Start index of the batch.
            end (int): End index of the batch.

        Returns:
            xr.Dataset: Batch of segmentation data.
        """
        # Ensure input path has .zarr extension
        zarr_path = input_path.with_suffix(".zarr")

        # Open the dataset
        dataset = xr.open_zarr(zarr_path)

        # Extract the specified time slice
        batch = dataset.isel(time=slice(start, end))

        return batch


class ParquetAnalysisStorage:
    """
    Analysis results storage adapter using Parquet format.

    This class handles storing and retrieving analysis results using Parquet,
    which provides efficient columnar storage for tabular data.
    """

    def save_category_analysis(
        self,
        counts: dict[int, int],
        percentages: dict[int, float],
        output_path: Path,
        frame_idx: int | None = None,
    ) -> Path:
        """
        Save category counts and percentages to Parquet storage.

        Args:
            counts (dict[int, int]): Dictionary mapping category IDs to pixel counts.
            percentages (dict[int, float]): Dictionary mapping category IDs to percentages.
            output_path (Path): Path to save the data.
            frame_idx (int, optional): Frame index, for video analysis.

        Returns:
            Path: Path to the saved Parquet file.
        """
        # Create a DataFrame with category analysis
        data = []
        for category_id in counts.keys():
            row = {
                "category_id": category_id,
                "pixel_count": counts[category_id],
                "percentage": percentages[category_id],
            }
            if frame_idx is not None:
                row["frame_idx"] = frame_idx
            data.append(row)

        df = pd.DataFrame(data)

        # Ensure output path has .parquet extension
        parquet_path = output_path.with_suffix(".parquet")

        # Save to Parquet format
        df.to_parquet(parquet_path, index=False)

        logger.info(f"Saved category analysis to {parquet_path}")
        return parquet_path

    def save_video_analysis(
        self,
        segmentation_data: np.ndarray | xr.DataArray | xr.Dataset,
        metadata: dict[str, object],
        output_path: Path,
    ) -> Path:
        """
        Save video analysis results to Parquet storage.

        Args:
            segmentation_data: Segmentation data.
            metadata (dict[str, object]): Metadata about the segmentation.
            output_path (Path): Path to save the data.

        Returns:
            Path: Path to the saved Parquet file.
        """
        # Initialize list to hold all frame statistics
        all_stats = []

        # Get the appropriate data array
        if (
            isinstance(segmentation_data, xr.Dataset)
            and "segmentation" in segmentation_data
        ):
            data_array = segmentation_data.segmentation
        elif isinstance(segmentation_data, xr.DataArray):
            data_array = segmentation_data
        elif isinstance(segmentation_data, np.ndarray):
            # Assuming shape (frames, height, width)
            data_array = segmentation_data
        else:
            raise ValueError("Unsupported data type for segmentation_data")

        # Analyze each frame
        if isinstance(data_array, (xr.DataArray, xr.Dataset)):
            frame_count = data_array.sizes.get("time", 1)
            for frame_idx in range(frame_count):
                frame = (
                    data_array.isel(time=frame_idx).values
                    if hasattr(data_array, "isel")
                    else data_array[frame_idx]
                )
                frame_stats = self._analyze_frame(frame, frame_idx)
                all_stats.extend(frame_stats)
        else:
            # Numpy array
            for frame_idx, frame in enumerate(data_array):
                frame_stats = self._analyze_frame(frame, frame_idx)
                all_stats.extend(frame_stats)

        # Create DataFrame
        df = pd.DataFrame(all_stats)

        # Ensure output path has .parquet extension
        parquet_path = output_path.with_suffix(".parquet")

        # Save to Parquet format
        df.to_parquet(parquet_path, index=False)

        logger.info(f"Saved video analysis to {parquet_path}")
        return parquet_path

    def _analyze_frame(
        self, frame: np.ndarray, frame_idx: int
    ) -> list[dict[str, object]]:
        """
        Analyze a single frame and return statistics.

        Args:
            frame (np.ndarray): Frame data.
            frame_idx (int): Frame index.

        Returns:
            list[dict[str, object]]: List of statistics dictionaries.
        """
        # Count pixels per category
        unique_values, counts = np.unique(frame, return_counts=True)
        total_pixels = frame.size

        # Create statistics for each category
        frame_stats = []
        for value, count in zip(unique_values, counts):
            percentage = (count / total_pixels) * 100
            frame_stats.append(
                {
                    "frame_idx": frame_idx,
                    "category_id": int(value),
                    "pixel_count": int(count),
                    "percentage": float(percentage),
                }
            )

        return frame_stats

    def load_category_analysis(self, input_path: Path) -> pd.DataFrame:
        """
        Load category analysis from Parquet storage.

        Args:
            input_path (Path): Path to the Parquet file.

        Returns:
            pd.DataFrame: DataFrame with category analysis.
        """
        # Ensure input path has .parquet extension
        parquet_path = input_path.with_suffix(".parquet")

        # Load the DataFrame
        df = pd.read_parquet(parquet_path)

        return df


class FileHandler:
    """
    A class for handling file operations related to segmentation data and metadata.

    This class provides methods for verifying files and checking their integrity.

    Methods:
        verify_zarr_file: Verifies the integrity of a Zarr segmentation file.
        verify_video_file: Verifies the integrity of a video file.
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
            zarr_path = file_path.with_suffix(".zarr")
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
                logger.warning(
                    f"Zarr file at {zarr_path} contains no segmentation data"
                )
                return False

            # Check data shape consistency
            if segmentation_data.dims != ("time", "y", "x"):
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
            parquet_path = file_path.with_suffix(".parquet")
            if not parquet_path.exists():
                logger.warning(f"Parquet file at {parquet_path} does not exist")
                return False

            # Try to load the DataFrame
            df = pd.read_parquet(parquet_path)

            # Check for required columns
            required_columns = {"category_id", "pixel_count", "percentage"}
            if not required_columns.issubset(set(df.columns)):
                logger.warning(
                    f"Parquet file at {parquet_path} is missing required columns"
                )
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


class StorageFactory:
    """
    Factory class for creating storage adapters.

    This class provides methods for creating appropriate storage adapters
    based on configuration options.
    """

    @staticmethod
    def create_segmentation_storage(config: Config) -> SegmentationStorage:
        """
        Create a segmentation storage adapter based on configuration.

        Args:
            config (Config): Configuration object.

        Returns:
            SegmentationStorage: Appropriate segmentation storage adapter.
        """
        # For now, we're using Zarr for all segmentation storage
        return ZarrSegmentationStorage()

    @staticmethod
    def create_analysis_storage(config: Config) -> ParquetAnalysisStorage:
        """
        Create an analysis storage adapter based on configuration.

        Args:
            config (Config): Configuration object.

        Returns:
            ParquetAnalysisStorage: Appropriate analysis storage adapter.
        """
        return ParquetAnalysisStorage()
