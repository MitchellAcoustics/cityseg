"""Core SegmentationDataset structure and creation functions.

This module defines the standard xarray Dataset schema for segmentation data
and provides validation and creation utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


class SegmentationDataset:
    """Standard schema and validation for segmentation xarray Datasets.

    This class defines the expected structure for segmentation datasets:

    Dimensions:
        - time: Time dimension for video data (optional for single images)
        - y: Height dimension (pixels)
        - x: Width dimension (pixels)

    Coordinates:
        - time: Timestamp or frame number
        - y: Pixel row indices (0-based)
        - x: Pixel column indices (0-based)

    Data Variables:
        - seg_map: Integer labels for each pixel (dims: [time], y, x)
        - confidence: Prediction confidence scores (dims: [time], y, x) [optional]

    Attributes:
        - model_name: Name of segmentation model used
        - model_version: Version of the model
        - class_labels: Mapping from label IDs to class names
        - palette: RGB color mapping for visualization
        - source_files: Original input file paths
        - processing_metadata: Additional processing information
    """

    REQUIRED_DIMS = {"y", "x"}
    OPTIONAL_DIMS = {"time", "class_id", "rgb"}
    REQUIRED_COORDS = {"y", "x"}
    OPTIONAL_COORDS = {"time", "class_id"}
    REQUIRED_DATA_VARS = {"seg_map", "palette", "class_label"}
    OPTIONAL_DATA_VARS = {"confidence", "image"}
    REQUIRED_ATTRS = {"model_name", "class_labels"}
    OPTIONAL_ATTRS = {
        "model_version",
        "palette",
        "source_files",
        "processing_metadata",
        "created_at",
        "cityseg_version",
    }

    @classmethod
    def validate_dataset(cls, ds: xr.Dataset) -> None:
        """Validate that a Dataset conforms to the SegmentationDataset schema.

        Args:
            ds: xarray Dataset to validate

        Raises:
            ValueError: If dataset doesn't conform to schema
        """
        # Check required dimensions
        missing_dims = cls.REQUIRED_DIMS - set(ds.dims.keys())  # type: ignore
        if missing_dims:
            raise ValueError(f"Missing required dimensions: {missing_dims}")

        # Check required coordinates
        missing_coords = cls.REQUIRED_COORDS - set(ds.coords.keys())  # type: ignore
        if missing_coords:
            raise ValueError(f"Missing required coordinates: {missing_coords}")

        # Check required data variables
        missing_vars = cls.REQUIRED_DATA_VARS - set(ds.data_vars.keys())
        if missing_vars:
            raise ValueError(f"Missing required data variables: {missing_vars}")

        # Check required attributes
        missing_attrs = cls.REQUIRED_ATTRS - set(ds.attrs.keys())
        if missing_attrs:
            raise ValueError(f"Missing required attributes: {missing_attrs}")

        # Validate seg_map data variable
        seg_var = ds.seg_map
        expected_dims = {"y", "x"}
        if "time" in ds.dims:
            expected_dims.add("time")

        if set(seg_var.dims) != expected_dims:
            raise ValueError(
                f"seg_map variable has dims {seg_var.dims}, expected {expected_dims}"
            )

        # Validate data types
        if not np.issubdtype(seg_var.dtype, np.integer):
            raise ValueError("seg_map data must be integer type")

        if "confidence" in ds.data_vars:
            conf_var = ds.confidence
            if not np.issubdtype(conf_var.dtype, np.floating):
                raise ValueError("Confidence data must be floating point type")

        logger.info("Dataset validation passed")


# def create_segmentation_dataset(
#     segmentation_data: np.ndarray,
#     model_name: str,
#     class_labels: dict[int, str],
#     confidence_data: np.ndarray | None = None,
#     image_data: np.ndarray | None = None,
#     source_files: list[str] | str | None = None,
#     palette: dict[int, tuple[int, int, int]] | None = None,
#     model_version: str | None = None,
#     processing_metadata: dict[str, Any] | None = None,
# ) -> xr.Dataset:
#     """Create a SegmentationDataset from raw segmentation output.

#     Args:
#         segmentation_data: Integer segmentation labels, shape (H, W) or (T, H, W)
#         model_name: Name of the segmentation model used
#         class_labels: Mapping from label IDs to class names
#         confidence_data: Optional confidence scores, same shape as segmentation_data
#         image_data: Optional original image data, shape (H, W, 3) or (T, H, W, 3)
#         source_files: Original input file path(s)
#         palette: RGB color mapping for visualization {label_id: (r, g, b)}
#         model_version: Version of the segmentation model
#         processing_metadata: Additional metadata about processing

#     Returns:
#         xarray Dataset conforming to SegmentationDataset schema

#     Raises:
#         ValueError: If input data doesn't have expected shape or format
#     """
#     # Validate input data
#     if not isinstance(segmentation_data, np.ndarray):
#         raise ValueError("segmentation_data must be a numpy array")

#     if segmentation_data.ndim not in [2, 3]:
#         raise ValueError("segmentation_data must be 2D (H, W) or 3D (T, H, W)")

#     if not np.issubdtype(segmentation_data.dtype, np.integer):
#         raise ValueError("segmentation_data must have integer dtype")

#     # Determine if this is video (3D) or image (2D) data
#     is_video = segmentation_data.ndim == 3

#     if is_video:
#         num_frames, height, width = segmentation_data.shape
#         dims = ["time", "y", "x"]
#         coords = {
#             "time": np.arange(num_frames),
#             "y": np.arange(height),
#             "x": np.arange(width),
#         }
#     else:
#         height, width = segmentation_data.shape
#         dims = ["y", "x"]
#         coords = {"y": np.arange(height), "x": np.arange(width)}

#     # Add class_id coordinate for palette and label mapping
#     class_ids = list(class_labels.keys())
#     coords["class_id"] = pd.Categorical(class_ids, categories=class_ids)

#     # Create data variables
#     data_vars = {"seg_map": (dims, segmentation_data.astype(np.int32))}

#     # Add confidence data if provided
#     if confidence_data is not None:
#         if confidence_data.shape != segmentation_data.shape:
#             raise ValueError(
#                 "confidence_data must have same shape as segmentation_data"
#             )
#         if not np.issubdtype(confidence_data.dtype, np.floating):
#             logger.warning("Converting confidence_data to float32")
#             confidence_data = confidence_data.astype(np.float32)
#         data_vars["confidence"] = (dims, confidence_data)

#     # Add original image data if provided
#     if image_data is not None:
#         if is_video:
#             if image_data.ndim != 4 or image_data.shape[:-1] != segmentation_data.shape:
#                 raise ValueError(
#                     "For video: image_data must have shape (T, H, W, 3) matching segmentation_data (T, H, W)"
#                 )
#             image_dims = ["time", "y", "x", "rgb"]
#         else:
#             if image_data.ndim != 3 or image_data.shape[:-1] != segmentation_data.shape:
#                 raise ValueError(
#                     "For single image: image_data must have shape (H, W, 3) matching segmentation_data (H, W)"
#                 )
#             image_dims = ["y", "x", "rgb"]

#         # Add rgb coordinate if not already present
#         if "rgb" not in coords:
#             coords["rgb"] = ["r", "g", "b"]

#         data_vars["image"] = (image_dims, image_data)

#     # Add class labels as data variable for direct xarray operations
#     class_label_array = np.array([class_labels[cid] for cid in class_ids])
#     data_vars["class_label"] = (["class_id"], class_label_array)

#     # Always create palette as data variable (required by trial version)
#     palette_array = np.zeros((len(class_ids), 3), dtype=np.uint8)

#     # Default colors using matplotlib colormap
#     import matplotlib.pyplot as plt

#     cmap = plt.cm.get_cmap("tab20")

#     for i, cid in enumerate(class_ids):
#         if palette is not None and cid in palette:
#             palette_array[i] = palette[cid]
#         else:
#             # Generate default color
#             color = cmap(i / len(class_ids))
#             r, g, b = color[:3]
#             palette_array[i] = [int(r * 255), int(g * 255), int(b * 255)]

#     # Add rgb coordinate if not already present
#     if "rgb" not in coords:
#         coords["rgb"] = ["r", "g", "b"]

#     data_vars["palette"] = (["class_id", "rgb"], palette_array)

#     # Create attributes (keep class_labels in attrs for backwards compatibility)
#     attrs = {
#         "model_name": model_name,
#         "class_labels": class_labels,
#     }

#     # Add optional attributes
#     if model_version is not None:
#         attrs["model_version"] = model_version

#     if source_files is not None:
#         if isinstance(source_files, str):
#             source_files = [source_files]
#         attrs["source_files"] = source_files

#     # Keep palette in attrs for backwards compatibility
#     if palette is not None:
#         attrs["palette"] = palette

#     if processing_metadata is not None:
#         attrs["processing_metadata"] = processing_metadata

#     # Add creation timestamp and version
#     from datetime import datetime

#     attrs["created_at"] = datetime.now().isoformat()

#     try:
#         from cityseg import __version__

#         attrs["cityseg_version"] = __version__
#     except ImportError:
#         pass

#     # Create the dataset
#     ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

#     # Validate the created dataset
#     SegmentationDataset.validate_dataset(ds)

#     logger.info(f"Created SegmentationDataset with shape {segmentation_data.shape}")
#     return ds


# def load_segmentation_dataset(file_path: str | Path) -> xr.Dataset:
#     """Load a SegmentationDataset from file.

#     Args:
#         file_path: Path to the dataset file (.zarr, .nc, or .h5)

#     Returns:
#         xarray Dataset conforming to SegmentationDataset schema

#     Raises:
#         ValueError: If file format is not supported or dataset is invalid
#         FileNotFoundError: If file doesn't exist
#     """
#     file_path = Path(file_path)

#     if not file_path.exists():
#         raise FileNotFoundError(f"Dataset file not found: {file_path}")

#     # Determine file format and load appropriately
#     if file_path.suffix == ".zarr" or file_path.name.endswith(".zarr"):
#         ds = xr.open_zarr(file_path)
#     elif file_path.suffix == ".nc":
#         ds = xr.open_dataset(file_path)
#     elif file_path.suffix == ".h5":
#         # For backwards compatibility with HDF5 files
#         ds = xr.open_dataset(file_path, engine="h5netcdf")
#     else:
#         raise ValueError(f"Unsupported file format: {file_path.suffix}")

#     # Validate the loaded dataset
#     try:
#         SegmentationDataset.validate_dataset(ds)
#     except ValueError as e:
#         logger.warning(f"Loaded dataset may not conform to schema: {e}")

#     logger.info(f"Loaded SegmentationDataset from {file_path}")
#     return ds


# def save_segmentation_dataset(
#     ds: xr.Dataset,
#     file_path: str | Path,
#     format: Literal["zarr", "netcdf", "hdf5"] = "zarr",
#     compression: str | None = "zstd",
#     chunks: dict[str, int] | None = None,
# ) -> None:
#     """Save a SegmentationDataset to file.

#     Args:
#         ds: SegmentationDataset to save
#         file_path: Output file path
#         format: Output format ("zarr", "netcdf", or "hdf5")
#         compression: Compression algorithm (format-dependent)
#         chunks: Chunking specification for performance

#     Raises:
#         ValueError: If dataset is invalid or format unsupported
#     """
#     # Validate dataset before saving
#     SegmentationDataset.validate_dataset(ds)

#     file_path = Path(file_path)
#     file_path.parent.mkdir(parents=True, exist_ok=True)

#     # Set default chunking for video data
#     if chunks is None and "time" in ds.dims:
#         chunks = {"time": 10, "y": 256, "x": 256}
#     elif chunks is None:
#         chunks = {"y": 512, "x": 512}

#     if format == "zarr":
#         # Zarr format with compression
#         encoding = {}
#         if compression:
#             for var in ds.data_vars:
#                 encoding[var] = {"compressor": compression, "chunks": chunks}
#         ds.to_zarr(file_path, encoding=encoding, mode="w")

#     elif format == "netcdf":
#         # NetCDF format
#         encoding = {}
#         if compression:
#             for var in ds.data_vars:
#                 encoding[var] = {
#                     "zlib": True,
#                     "complevel": 6,
#                     "chunksizes": tuple(chunks.values()),
#                 }
#         ds.to_netcdf(file_path, encoding=encoding)

#     elif format == "hdf5":
#         # HDF5 format via h5netcdf
#         encoding = {}
#         if compression:
#             for var in ds.data_vars:
#                 encoding[var] = {
#                     "compression": compression,
#                     "chunks": tuple(chunks.values()),
#                 }
#         ds.to_netcdf(file_path, engine="h5netcdf", encoding=encoding)

#     else:
#         raise ValueError(f"Unsupported format: {format}")

#     logger.info(f"Saved SegmentationDataset to {file_path} in {format} format")
