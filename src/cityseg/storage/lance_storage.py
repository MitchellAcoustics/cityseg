"""
Lance storage adapter for CitySeg.
Provides efficient storage for segmentation maps using the Lance format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

try:
    import lance

    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False
    logging.warning(
        "Lance is not installed. To use LanceSegmentationStorage, install lance: "
        "pip install lance pyarrow"
    )

from .storage import SegmentationStorage
from loguru import logger


class LanceSegmentationStorage(SegmentationStorage):
    """Storage adapter for segmentation data using Lance format.

    Lance provides efficient storage for tensor data with fast random access,
    versioning, and good compression.

    Attributes:
        version: Storage format version
    """

    version = "0.1.0"

    def __init__(self):
        """Initialize the Lance storage adapter."""
        if not LANCE_AVAILABLE:
            raise ImportError(
                "Lance is required for LanceSegmentationStorage. "
                "Install with: pip install lance pyarrow"
            )
        super().__init__()

    def save_segmentation_data(
        self,
        data: Union[np.ndarray, xr.DataArray, xr.Dataset],
        metadata: Dict[str, Any],
        output_path: Union[str, Path],
    ) -> Path:
        """Save segmentation data to Lance format.

        Args:
            data: Segmentation data as numpy array or xarray DataArray/Dataset.
            metadata: Dictionary of metadata
            output_path: Base path for output files

        Returns:
            Path to the saved Lance dataset
        """
        # Ensure path exists
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # Create Lance output path
        lance_path = str(output_path) + ".lance"

        # Get the segmentation data
        if isinstance(data, xr.Dataset) and "segmentation" in data:
            seg_data = data["segmentation"].values
            time_coords = data["segmentation"].coords["time"].values
        elif isinstance(data, xr.DataArray):
            seg_data = data.values
            time_coords = data.coords["time"].values
        elif isinstance(data, np.ndarray):
            seg_data = data
            time_coords = np.arange(seg_data.shape[0])
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Get dimensions
        num_frames, height, width = seg_data.shape

        # Create records for each frame
        # Using list of dictionaries which will be converted to a PyArrow Table
        records = []

        for i, t in enumerate(time_coords):
            # Store frame index, height, width, and flatten segmentation map
            # Using fixed-size list type for the segmentation data
            records.append(
                {
                    "frame_idx": int(t),
                    "height": int(height),
                    "width": int(width),
                    "segmentation_data": seg_data[i].flatten().tolist(),
                }
            )

        # Convert to PyArrow Table
        # This explicitly defines the schema to ensure the segmentation data is stored as a list
        schema = pa.schema(
            [
                pa.field("frame_idx", pa.int32()),
                pa.field("height", pa.int32()),
                pa.field("width", pa.int32()),
                pa.field("segmentation_data", pa.list_(pa.int32())),
            ]
        )

        table = pa.Table.from_pylist(records, schema=schema)

        # Save metadata as a separate JSON file
        # Lance has limited support for complex metadata
        metadata_path = str(output_path) + "_metadata.json"

        # Clean metadata for JSON serialization
        clean_metadata = self._clean_metadata_for_storage(metadata)

        # Add shape information to metadata
        clean_metadata["tensor_shape"] = {
            "time": len(time_coords),
            "height": int(height),
            "width": int(width),
        }

        # Save metadata to JSON
        with open(metadata_path, "w") as f:
            json.dump(clean_metadata, f, indent=2)

        # Write to Lance dataset
        try:
            if Path(lance_path).exists():
                # If dataset exists, update it (creates a new version)
                ds = lance.dataset(lance_path)
                ds.update(table)
                logger.info(f"Updated existing Lance dataset at {lance_path}")
            else:
                # Create new dataset using the correct function from lance.dataset
                lance.write_dataset(table, lance_path)
                logger.info(f"Created new Lance dataset at {lance_path}")
        except Exception as e:
            logger.error(f"Error saving to Lance format: {str(e)}")
            raise

        logger.info(f"Saved segmentation data to Lance format: {lance_path}")
        logger.info(f"Saved metadata to JSON: {metadata_path}")

        return Path(lance_path)

    def load_segmentation_data(
        self, input_path: Union[str, Path]
    ) -> Tuple[xr.Dataset, Dict[str, Any]]:
        """Load segmentation data from Lance format.

        Args:
            input_path: Path to Lance dataset

        Returns:
            Tuple of xarray Dataset with segmentation data and metadata
        """
        lance_path = str(input_path)

        # Check if path exists
        if not Path(lance_path).exists():
            raise FileNotFoundError(f"Lance dataset not found: {lance_path}")

        # Load metadata from companion file
        metadata_path = lance_path.replace(".lance", "_metadata.json")
        metadata = {}
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

                # Convert category IDs back to integers if stored as strings
                if "id2label" in metadata and all(
                    k.isdigit() for k in metadata["id2label"].keys()
                ):
                    metadata["id2label"] = {
                        int(k): v for k, v in metadata["id2label"].items()
                    }
        except FileNotFoundError:
            logger.warning(f"Metadata file not found: {metadata_path}")

        # Load Lance dataset
        try:
            ds = lance.dataset(lance_path)

            # Read entire dataset as a table
            table = ds.to_table()

            # Convert to pandas DataFrame
            df = table.to_pandas()

            # Sort by frame index
            df = df.sort_values("frame_idx")

            # Get dimensions from the first row
            height = df["height"].iloc[0]
            width = df["width"].iloc[0]

            # Reconstruct the 3D array from the flattened data
            frames = []
            frame_indices = df["frame_idx"].values

            for _, row in df.iterrows():
                # Get segmentation data and reshape to 2D
                flat_data = np.array(row["segmentation_data"], dtype=np.int32)
                seg_map = flat_data.reshape(height, width)
                frames.append(seg_map)

            # Stack all frames into a 3D array
            seg_array = np.stack(frames)

            # Create coordinates for dimensions
            time_coords = frame_indices
            y_coords = np.arange(height)
            x_coords = np.arange(width)

            # Create DataArray
            da = xr.DataArray(
                seg_array,
                dims=["time", "y", "x"],
                coords={"time": time_coords, "y": y_coords, "x": x_coords},
            )

            # Create Dataset
            dataset = xr.Dataset({"segmentation": da})

            # Add metadata to dataset attributes
            for key, value in metadata.items():
                dataset.attrs[key] = value

            logger.info(f"Loaded segmentation data from Lance format: {lance_path}")
            return dataset, metadata

        except Exception as e:
            logger.error(f"Error loading Lance dataset: {str(e)}")
            raise

    def load_segmentation_batch(
        self, input_path: Union[str, Path], start: int, end: int
    ) -> xr.Dataset:
        """Load a batch of segmentation data.

        Args:
            input_path: Path to the saved data.
            start: Start index of the batch.
            end: End index of the batch.

        Returns:
            Batch of segmentation data.
        """
        lance_path = str(input_path)

        # Check if path exists
        if not Path(lance_path).exists():
            raise FileNotFoundError(f"Lance dataset not found: {lance_path}")

        # Load metadata
        metadata_path = lance_path.replace(".lance", "_metadata.json")
        metadata = {}
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Metadata file not found: {metadata_path}")

        # Load Lance dataset
        try:
            ds = lance.dataset(lance_path)

            # Use Lance's filter capability to get only frames in the range
            filtered_ds = ds.filter(f"frame_idx >= {start} AND frame_idx < {end}")

            # Read filtered data
            table = filtered_ds.to_table()

            if table.num_rows == 0:
                logger.warning(f"No frames found in range {start}-{end}")
                return xr.Dataset()

            # Convert to pandas DataFrame
            df = table.to_pandas()

            # Sort by frame index
            df = df.sort_values("frame_idx")

            # Get dimensions
            height = df["height"].iloc[0]
            width = df["width"].iloc[0]

            # Reconstruct frames
            frames = []
            frame_indices = df["frame_idx"].values

            for _, row in df.iterrows():
                flat_data = np.array(row["segmentation_data"], dtype=np.int32)
                seg_map = flat_data.reshape(height, width)
                frames.append(seg_map)

            # Stack into 3D array
            seg_array = np.stack(frames)

            # Create coordinates
            time_coords = frame_indices
            y_coords = np.arange(height)
            x_coords = np.arange(width)

            # Create DataArray
            da = xr.DataArray(
                seg_array,
                dims=["time", "y", "x"],
                coords={"time": time_coords, "y": y_coords, "x": x_coords},
            )

            # Create Dataset
            dataset = xr.Dataset({"segmentation": da})

            # Add metadata to attributes (skipping tensor_shape which we used to get dimensions)
            for key, value in metadata.items():
                if key != "tensor_shape":
                    dataset.attrs[key] = value

            return dataset

        except Exception as e:
            logger.error(f"Error loading batch from Lance dataset: {str(e)}")
            raise

    def get_versions(self, lance_path: Union[str, Path]) -> pd.DataFrame:
        """Get available versions of a Lance dataset.

        Args:
            lance_path: Path to Lance dataset

        Returns:
            DataFrame of version information
        """
        lance_path = str(lance_path)

        if not Path(lance_path).exists():
            raise FileNotFoundError(f"Lance dataset not found: {lance_path}")

        # Load Lance dataset
        ds = lance.dataset(lance_path)

        # Get version information
        versions = []
        for version in ds.versions():
            versions.append(
                {
                    "version": version.version,
                    "timestamp": version.timestamp,
                    "fragment_ids": version.fragment_ids,
                }
            )

        return pd.DataFrame(versions)

    def get_version(
        self, lance_path: Union[str, Path], version: Optional[int] = None
    ) -> xr.Dataset:
        """Load a specific version of a Lance dataset.

        Args:
            lance_path: Path to Lance dataset
            version: Version number to load, defaults to latest

        Returns:
            xarray Dataset for the specified version
        """
        lance_path = str(lance_path)

        if not Path(lance_path).exists():
            raise FileNotFoundError(f"Lance dataset not found: {lance_path}")

        # Load Lance dataset with specific version
        try:
            # Open dataset with specified version
            ds = (
                lance.dataset(lance_path, version=version)
                if version is not None
                else lance.dataset(lance_path)
            )

            # Convert to pandas DataFrame
            df = ds.to_table().to_pandas()

            if df.empty:
                logger.warning(f"No data found in version {version}")
                return xr.Dataset()

            # Sort by frame index
            df = df.sort_values("frame_idx")

            # Get dimensions
            height = df["height"].iloc[0]
            width = df["width"].iloc[0]

            # Reconstruct frames
            frames = []
            frame_indices = df["frame_idx"].values

            for _, row in df.iterrows():
                flat_data = np.array(row["segmentation_data"], dtype=np.int32)
                seg_map = flat_data.reshape(height, width)
                frames.append(seg_map)

            # Stack into 3D array
            seg_array = np.stack(frames)

            # Create coordinates
            time_coords = frame_indices
            y_coords = np.arange(height)
            x_coords = np.arange(width)

            # Create DataArray
            da = xr.DataArray(
                seg_array,
                dims=["time", "y", "x"],
                coords={"time": time_coords, "y": y_coords, "x": x_coords},
            )

            # Create Dataset
            dataset = xr.Dataset({"segmentation": da})

            # Load metadata
            metadata_path = lance_path.replace(".lance", "_metadata.json")
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)

                    # Add metadata to attributes
                    for key, value in metadata.items():
                        if key != "tensor_shape":
                            dataset.attrs[key] = value
            except FileNotFoundError:
                logger.warning(f"Metadata file not found: {metadata_path}")

            # Add version information
            dataset.attrs["lance_version"] = (
                version if version is not None else ds.latest_version()
            )

            return dataset

        except Exception as e:
            logger.error(f"Error loading version from Lance dataset: {str(e)}")
            raise

    def _clean_metadata_for_storage(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata for JSON serialization.

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Cleaned metadata suitable for JSON serialization
        """
        clean_metadata = {}

        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool, list)):
                clean_metadata[k] = v
            elif isinstance(v, dict):
                # Convert dict of ints to dict of strings for JSON compatibility
                if all(isinstance(key, int) for key in v.keys()):
                    clean_metadata[k] = {str(key): value for key, value in v.items()}
                else:
                    clean_metadata[k] = v
            elif isinstance(v, np.ndarray):
                clean_metadata[k] = v.tolist()
            else:
                # Convert other types to string representation
                clean_metadata[k] = str(v)

        return clean_metadata
