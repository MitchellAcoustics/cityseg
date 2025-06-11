"""Core MediaDataset structure and media loading functions.

This module defines the standard xarray Dataset schema for media data (images/videos)
and provides loading utilities for various media formats.
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image
from datetime import datetime


import numpy as np
import xarray as xr
from loguru import logger


class MediaDataset:
    """Standard schema and validation for media xarray Datasets.

    This class defines the expected structure for media datasets:

    Dimensions:
        - time: Time dimension for video data (optional for single images)
        - y: Height dimension (pixels)
        - x: Width dimension (pixels)
        - rgb: Color channels (r, g, b)

    Coordinates:
        - time: Timestamp or frame number (for videos)
        - y: Pixel row indices (0-based)
        - x: Pixel column indices (0-based)
        - rgb: Color channel names ["r", "g", "b"]

    Data Variables:
        - image: RGB image data (dims: [time], y, x, rgb)

    Attributes:
        - source_file: Original media file path
        - media_type: "image" or "video"
        - created_at: Dataset creation timestamp
        - cityseg_version: Version of CitySeg used
        - width: Image/video width in pixels
        - height: Image/video height in pixels
        - fps: Frames per second (video only)
        - duration: Duration in seconds (video only)
    """

    REQUIRED_DIMS = {"y", "x", "rgb"}
    OPTIONAL_DIMS = {"time"}
    REQUIRED_COORDS = {"y", "x", "rgb"}
    OPTIONAL_COORDS = {"time"}
    REQUIRED_DATA_VARS = {"image"}
    OPTIONAL_DATA_VARS = {"metadata"}
    REQUIRED_ATTRS = {"source_file", "media_type"}
    OPTIONAL_ATTRS = {
        "created_at",
        "cityseg_version",
        "width",
        "height",
        "fps",
        "duration",
        "processing_metadata",
    }

    @classmethod
    def validate_dataset(cls, ds: xr.Dataset) -> None:
        """Validate that a Dataset conforms to the MediaDataset schema.

        Args:
            ds: xarray Dataset to validate

        Raises:
            ValueError: If dataset doesn't conform to schema
        """
        # Check required dimensions
        missing_dims = cls.REQUIRED_DIMS - set(str(dim) for dim in ds.sizes.keys())
        if missing_dims:
            raise ValueError(f"Missing required dimensions: {missing_dims}")

        # Check required coordinates
        missing_coords = cls.REQUIRED_COORDS - set(
            str(coord) for coord in ds.coords.keys()
        )
        if missing_coords:
            raise ValueError(f"Missing required coordinates: {missing_coords}")

        # Check required data variables
        missing_vars = cls.REQUIRED_DATA_VARS - set(
            str(var) for var in ds.data_vars.keys()
        )
        if missing_vars:
            raise ValueError(f"Missing required data variables: {missing_vars}")

        # Check required attributes
        missing_attrs = cls.REQUIRED_ATTRS - set(str(attr) for attr in ds.attrs.keys())
        if missing_attrs:
            raise ValueError(f"Missing required attributes: {missing_attrs}")

        # Validate image data variable
        image_var = ds.image
        expected_dims = {"y", "x", "rgb"}
        if "time" in ds.dims:
            expected_dims.add("time")

        if set(image_var.dims) != expected_dims:
            raise ValueError(
                f"Image variable has dims {image_var.dims}, expected {expected_dims}"
            )

        # Validate data types
        if not np.issubdtype(image_var.dtype, np.integer) and not np.issubdtype(
            image_var.dtype, np.floating
        ):
            raise ValueError("Image data must be numeric type")

        # Validate media type
        if ds.attrs["media_type"] not in ["image", "video"]:
            raise ValueError("media_type must be 'image' or 'video'")

        # Video-specific validation
        if ds.attrs["media_type"] == "video":
            if "time" not in ds.dims:
                raise ValueError("Video datasets must have time dimension")
        else:  # image
            if "time" in ds.dims:
                logger.warning("Single image dataset has time dimension")

        logger.info("MediaDataset validation passed")

    @classmethod
    def is_video(cls, ds: xr.Dataset) -> bool:
        """Check if dataset represents video data."""
        return "time" in ds.dims and ds.attrs.get("media_type") == "video"

    @classmethod
    def is_image(cls, ds: xr.Dataset) -> bool:
        """Check if dataset represents image data."""
        return "time" not in ds.dims and ds.attrs.get("media_type") == "image"


def load_image(file_path: str | Path) -> xr.Dataset:
    """Load a single image into MediaDataset format.

    Args:
        file_path: Path to image file

    Returns:
        xarray Dataset conforming to MediaDataset schema

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is not supported or corrupted
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    try:
        with Image.open(file_path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Convert to numpy array
            image_array = np.array(img, dtype=np.uint8)
            height, width = image_array.shape[:2]

    except Exception as e:
        raise ValueError(f"Failed to load image {file_path}: {e}")

    # Create coordinates
    coords = {
        "y": np.arange(height),
        "x": np.arange(width),
        "rgb": ["r", "g", "b"],
    }

    # Create data variables
    data_vars = {"image": (["y", "x", "rgb"], image_array)}

    # Create attributes
    attrs = {
        "source_file": str(file_path.absolute()),
        "media_type": "image",
        "width": width,
        "height": height,
        "created_at": datetime.now().isoformat(),
    }

    # Add CitySeg version if available
    try:
        from cityseg import __version__

        attrs["cityseg_version"] = __version__
    except ImportError:
        pass

    # Create the dataset
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    # Validate the created dataset
    MediaDataset.validate_dataset(ds)

    logger.info(f"Loaded image: {file_path} ({width}x{height})")
    return ds


def load_video(file_path: str | Path, max_frames: int | None = None) -> xr.Dataset:
    """Load a video into MediaDataset format.

    Args:
        file_path: Path to video file
        max_frames: Maximum number of frames to load (None for all)

    Returns:
        xarray Dataset conforming to MediaDataset schema

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video format is not supported or corrupted
        ImportError: If required video processing library is not available
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    # Try to load video using OpenCV first, fall back to imageio
    video_array = None
    fps = None
    duration = None

    try:
        import cv2

        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError("Could not open video with OpenCV")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else None

        if max_frames is not None:
            frame_count = min(frame_count, max_frames)

        frames = []
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if frames:
            video_array = np.array(frames, dtype=np.uint8)

    except ImportError:
        logger.warning("OpenCV not available, trying imageio")

    # Fallback to imageio if OpenCV failed
    if video_array is None:
        try:
            import imageio.v3 as iio  # type: ignore

            # Read video
            frames = []
            reader = iio.imopen(file_path, "r")

            for i, frame in enumerate(reader):
                if max_frames is not None and i >= max_frames:
                    break
                frames.append(frame)

            if frames:
                video_array = np.array(frames, dtype=np.uint8)

            # Try to get metadata
            try:
                meta = reader.metadata()
                fps = meta.get("fps", 30.0)  # Default to 30 fps
                duration = len(frames) / fps
            except Exception:
                fps = 30.0
                duration = len(frames) / fps

        except ImportError:
            raise ImportError(
                "Either OpenCV (cv2) or imageio is required for video loading"
            )
        except Exception as e:
            raise ValueError(f"Failed to load video {file_path}: {e}")

    if video_array is None or len(video_array) == 0:
        raise ValueError(f"No frames could be read from video: {file_path}")

    num_frames, height, width = video_array.shape[:3]

    # Create coordinates
    coords = {
        "time": np.arange(num_frames),
        "y": np.arange(height),
        "x": np.arange(width),
        "rgb": ["r", "g", "b"],
    }

    # Create data variables
    data_vars = {"image": (["time", "y", "x", "rgb"], video_array)}

    # Create attributes
    attrs = {
        "source_file": str(file_path.absolute()),
        "media_type": "video",
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "created_at": datetime.now().isoformat(),
    }

    # Add CitySeg version if available
    try:
        from cityseg import __version__

        attrs["cityseg_version"] = __version__
    except ImportError:
        pass

    # Create the dataset
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    # Validate the created dataset
    MediaDataset.validate_dataset(ds)

    logger.info(
        f"Loaded video: {file_path} ({num_frames} frames, {width}x{height}, {fps:.1f} fps)"
    )
    return ds


def save_media_dataset(
    ds: xr.Dataset,
    file_path: str | Path,
    format: str = "zarr",
    compression: str | None = "zstd",
    chunks: dict[str, int] | None = None,
) -> None:
    """Save a MediaDataset to file.

    Args:
        ds: MediaDataset to save
        file_path: Output file path
        format: Output format ("zarr", "netcdf", or "hdf5")
        compression: Compression algorithm (format-dependent)
        chunks: Chunking specification for performance

    Raises:
        ValueError: If dataset is invalid or format unsupported
    """
    # Validate dataset before saving
    MediaDataset.validate_dataset(ds)

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Set default chunking for video data
    if chunks is None and "time" in ds.dims:
        chunks = {"time": 10, "y": 256, "x": 256, "rgb": 3}
    elif chunks is None:
        chunks = {"y": 512, "x": 512, "rgb": 3}

    if format == "zarr":
        # Zarr format with compression
        encoding = {}
        if compression:
            for var in ds.data_vars:
                encoding[var] = {"compressor": compression, "chunks": chunks}
        ds.to_zarr(file_path, encoding=encoding, mode="w")

    elif format == "netcdf":
        # NetCDF format
        encoding = {}
        if compression:
            for var in ds.data_vars:
                encoding[var] = {
                    "zlib": True,
                    "complevel": 6,
                    "chunksizes": tuple(chunks.values()),
                }
        ds.to_netcdf(file_path, encoding=encoding)

    elif format == "hdf5":
        # HDF5 format via h5netcdf
        encoding = {}
        if compression:
            for var in ds.data_vars:
                encoding[var] = {
                    "compression": compression,
                    "chunks": tuple(chunks.values()),
                }
        ds.to_netcdf(file_path, engine="h5netcdf", encoding=encoding)

    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved MediaDataset to {file_path} in {format} format")


def load_media_dataset(file_path: str | Path) -> xr.Dataset:
    """Load a MediaDataset from file.

    Args:
        file_path: Path to the dataset file (.zarr, .nc, or .h5)

    Returns:
        xarray Dataset conforming to MediaDataset schema

    Raises:
        ValueError: If file format is not supported or dataset is invalid
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    # Determine file format and load appropriately
    if file_path.suffix == ".zarr" or file_path.name.endswith(".zarr"):
        ds = xr.open_zarr(file_path)
    elif file_path.suffix == ".nc":
        ds = xr.open_dataset(file_path)
    elif file_path.suffix == ".h5":
        # For backwards compatibility with HDF5 files
        ds = xr.open_dataset(file_path, engine="h5netcdf")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Validate the loaded dataset
    try:
        MediaDataset.validate_dataset(ds)
    except ValueError as e:
        logger.warning(f"Loaded dataset may not conform to MediaDataset schema: {e}")

    logger.info(f"Loaded MediaDataset from {file_path}")
    return ds
