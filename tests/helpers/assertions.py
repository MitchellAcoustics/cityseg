"""Custom assertion helpers for testing CitySeg components."""

import os
import numpy as np
import xarray as xr
import pandas as pd
import cv2
from PIL import Image
import json


def assert_segmentation_data_valid(data, min_frames=1, expected_classes=None):
    """
    Assert that the segmentation data is valid.

    Args:
        data: xarray.Dataset containing segmentation data
        min_frames: Minimum number of frames expected
        expected_classes: Optional list of class IDs expected in the data
    """
    # Check it's an xarray Dataset
    assert isinstance(data, xr.Dataset), "Data is not an xarray Dataset"

    # Check for required variables
    assert "segmentation" in data, "Missing segmentation variable"

    # Check dimensions
    assert "frames" in data.dims, "Missing frames dimension"
    assert "height" in data.dims, "Missing height dimension"
    assert "width" in data.dims, "Missing width dimension"

    # Check frame count
    assert data.dims["frames"] >= min_frames, (
        f"Not enough frames. Expected at least {min_frames}, got {data.dims['frames']}"
    )

    # Check data type
    assert np.issubdtype(data.segmentation.dtype, np.integer), (
        "Segmentation data should be integer type"
    )

    # Check value range
    assert data.segmentation.min() >= 0, "Segmentation contains negative values"

    # Check for expected classes if provided
    if expected_classes is not None:
        unique_classes = np.unique(data.segmentation.values)
        for class_id in expected_classes:
            assert class_id in unique_classes, (
                f"Expected class {class_id} not found in segmentation data"
            )

    # Check metadata attributes
    assert "model_name" in data.attrs, "Missing model_name in attributes"
    assert "processing_date" in data.attrs, "Missing processing_date in attributes"


def assert_analysis_data_valid(csv_path, expected_columns=None):
    """
    Assert that the analysis CSV data is valid.

    Args:
        csv_path: Path to the CSV file
        expected_columns: List of column names expected in the CSV
    """
    # Check file exists
    assert os.path.exists(csv_path), f"CSV file not found: {csv_path}"

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Check it's not empty
    assert len(df) > 0, f"CSV file is empty: {csv_path}"

    # Check for expected columns
    if expected_columns:
        for col in expected_columns:
            assert col in df.columns, f"Missing expected column: {col}"

    # If this is a percentages file, check values are between 0-100
    if "percentage" in df.columns:
        assert df["percentage"].min() >= 0, "Negative percentage found"
        assert df["percentage"].max() <= 100, "Percentage > 100% found"

    # If this is a counts file, check values are non-negative
    if "count" in df.columns:
        assert df["count"].min() >= 0, "Negative count found"

    return df


def assert_image_valid(image_path, min_width=10, min_height=10):
    """
    Assert that the image file is valid.

    Args:
        image_path: Path to the image file
        min_width: Minimum expected width
        min_height: Minimum expected height
    """
    # Check file exists
    assert os.path.exists(image_path), f"Image file not found: {image_path}"

    # Try to open with PIL
    try:
        img = Image.open(image_path)
        img.verify()  # Verify the image is valid

        # Check dimensions
        width, height = img.size
        assert width >= min_width, f"Image width too small: {width} < {min_width}"
        assert height >= min_height, f"Image height too small: {height} < {min_height}"

        # Check mode
        assert img.mode in ("RGB", "RGBA"), f"Unexpected image mode: {img.mode}"

    except Exception as e:
        assert False, f"Invalid image file {image_path}: {str(e)}"


def assert_video_valid(video_path, min_frames=1):
    """
    Assert that the video file is valid.

    Args:
        video_path: Path to the video file
        min_frames: Minimum number of frames expected
    """
    # Check file exists
    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    # Try to open with OpenCV
    cap = cv2.VideoCapture(str(video_path))
    try:
        assert cap.isOpened(), f"Failed to open video: {video_path}"

        # Check frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count >= min_frames, (
            f"Not enough frames: {frame_count} < {min_frames}"
        )

        # Check dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert width > 0, "Invalid video width"
        assert height > 0, "Invalid video height"

        # Check FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert fps > 0, "Invalid FPS"

    finally:
        cap.release()


def assert_processing_history_valid(history_path):
    """
    Assert that the processing history JSON file is valid.

    Args:
        history_path: Path to the processing history JSON file
    """
    # Check file exists
    assert os.path.exists(history_path), f"History file not found: {history_path}"

    # Try to load as JSON
    try:
        with open(history_path, "r") as f:
            history = json.load(f)

        # Check required fields
        assert "model_name" in history, "Missing model_name in history"
        assert "processing_date" in history, "Missing processing_date in history"
        assert "config" in history, "Missing config in history"

    except json.JSONDecodeError:
        assert False, f"Invalid JSON in history file: {history_path}"
