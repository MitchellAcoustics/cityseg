"""
This module provides functionality for building datasets from segmentation results.

It handles creating and saving xarray datasets with proper metadata for
both video segmentation and single image segmentation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr
from loguru import logger

from ..storage.storage import ZarrSegmentationStorage, ParquetAnalysisStorage


class DatasetBuilder:
    """
    A class for building datasets from segmentation results.

    This class provides methods to create, save, and analyze xarray datasets
    containing segmentation results.

    Methods:
        create_video_dataset: Create an xarray Dataset from video segmentation results
        create_image_dataset: Create an xarray Dataset from a single image segmentation
        save_segmentation: Save a segmentation dataset using ZarrSegmentationStorage
        save_analysis: Save analysis of segmentation results using ParquetAnalysisStorage
    """

    @staticmethod
    def create_video_dataset(
        segmentation_maps: List[np.ndarray],
        video_metadata: Dict[str, Any],
        frame_indices: List[int],
        model_metadata: Dict[str, Any],
        segmentation_metadata: Optional[Dict[str, Any]] = None,
    ) -> xr.Dataset:
        """
        Create an xarray Dataset from video segmentation results.

        Args:
            segmentation_maps: List of segmentation maps
            video_metadata: Metadata about the video
            frame_indices: Indices of the frames that were processed
            model_metadata: Metadata about the segmentation model
            segmentation_metadata: Additional metadata from segmentation results

        Returns:
            xarray Dataset containing segmentation data with metadata
        """
        # Stack segmentation maps into a 3D array
        segmentation_array = np.stack(segmentation_maps)

        # Create time coordinates based on frame indices and fps
        if frame_indices:
            time_coords = np.array(frame_indices) / video_metadata["fps"]
        else:
            time_coords = np.array([])

        # Create xarray DataArray with named dimensions
        segmentation_data = xr.DataArray(
            segmentation_array,
            dims=["time", "y", "x"],
            coords={
                "time": time_coords,
                "y": np.arange(video_metadata["height"]),
                "x": np.arange(video_metadata["width"]),
            },
        )

        # Combine all metadata
        attrs = {
            "model_name": model_metadata.get("name", ""),
            "model_type": model_metadata.get("model_type", ""),
            "fps": video_metadata["fps"],
            "frame_step": video_metadata.get("frame_step", 1),
            "original_width": video_metadata["width"],
            "original_height": video_metadata["height"],
            "codec": video_metadata.get("codec", None),
        }

        # Add segmentation metadata if available
        if segmentation_metadata:
            palette = segmentation_metadata.get("palette")
            if palette is not None:
                # Convert numpy array to list for serialization
                attrs["palette"] = (
                    palette.tolist() if isinstance(palette, np.ndarray) else palette
                )

            if "id2label" in segmentation_metadata:
                attrs["id2label"] = segmentation_metadata["id2label"]

            if "label2id" in segmentation_metadata:
                attrs["label2id"] = segmentation_metadata["label2id"]

        # Create dataset with data variables and attributes
        dataset = xr.Dataset(data_vars={"segmentation": segmentation_data}, attrs=attrs)

        logger.debug(f"Created video dataset with {len(segmentation_maps)} frames")
        return dataset

    @staticmethod
    def create_image_dataset(
        segmentation_map: np.ndarray,
        model_metadata: Dict[str, Any],
        segmentation_metadata: Optional[Dict[str, Any]] = None,
    ) -> xr.Dataset:
        """
        Create an xarray Dataset from a single image segmentation.

        Args:
            segmentation_map: Segmentation map for a single image
            model_metadata: Metadata about the segmentation model
            segmentation_metadata: Additional metadata from segmentation results

        Returns:
            xarray Dataset containing segmentation data with metadata
        """
        # Create xarray DataArray with named dimensions
        segmentation_data = xr.DataArray(
            segmentation_map,
            dims=["y", "x"],
            coords={
                "y": np.arange(segmentation_map.shape[0]),
                "x": np.arange(segmentation_map.shape[1]),
            },
        )

        # Combine all metadata
        attrs = {
            "model_name": model_metadata.get("name", ""),
            "model_type": model_metadata.get("model_type", ""),
            "original_width": segmentation_map.shape[1],
            "original_height": segmentation_map.shape[0],
        }

        # Add segmentation metadata if available
        if segmentation_metadata:
            palette = segmentation_metadata.get("palette")
            if palette is not None:
                # Convert numpy array to list for serialization
                attrs["palette"] = (
                    palette.tolist() if isinstance(palette, np.ndarray) else palette
                )

            if "id2label" in segmentation_metadata:
                attrs["id2label"] = segmentation_metadata["id2label"]

            if "label2id" in segmentation_metadata:
                attrs["label2id"] = segmentation_metadata["label2id"]

        # Create dataset with data variables and attributes
        dataset = xr.Dataset(data_vars={"segmentation": segmentation_data}, attrs=attrs)

        logger.debug(f"Created image dataset with shape {segmentation_map.shape}")
        return dataset

    @staticmethod
    def save_segmentation(dataset: xr.Dataset, output_path: Path) -> Path:
        """
        Save a segmentation dataset using ZarrSegmentationStorage.

        Args:
            dataset: xarray Dataset to save
            output_path: Base path for the output file

        Returns:
            Path to the saved dataset
        """
        storage = ZarrSegmentationStorage()
        metadata = dict(dataset.attrs)

        # Ensure the output path has a proper name
        if not output_path.stem.endswith("_segmentation"):
            output_path = output_path.with_name(f"{output_path.stem}_segmentation")

        saved_path = storage.save_segmentation_data(dataset, metadata, output_path)
        logger.info(f"Saved segmentation dataset to {saved_path}")
        return saved_path

    @staticmethod
    def save_analysis(dataset: xr.Dataset, output_path: Path) -> Path:
        """
        Save analysis of segmentation results using ParquetAnalysisStorage.

        Args:
            dataset: xarray Dataset to analyze
            output_path: Base path for the output file

        Returns:
            Path to the saved analysis
        """
        storage = ParquetAnalysisStorage()

        # Ensure the output path has a proper name
        if not output_path.stem.endswith("_analysis"):
            output_path = output_path.with_name(f"{output_path.stem}_analysis")

        saved_path = storage.save_video_analysis(
            dataset.segmentation, dict(dataset.attrs), output_path
        )

        logger.info(f"Saved segmentation analysis to {saved_path}")
        return saved_path
