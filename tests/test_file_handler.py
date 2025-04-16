import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from cityseg.config import Config
from cityseg.file_handler import FileHandler
from cityseg.storage_adapter import ZarrSegmentationStorage, ParquetAnalysisStorage


@pytest.fixture
def temp_zarr_file(tmp_path):
    file_path = tmp_path / "test.zarr"
    yield file_path
    if file_path.exists():
        import shutil
        shutil.rmtree(file_path)


@pytest.fixture
def temp_parquet_file(tmp_path):
    file_path = tmp_path / "test.parquet"
    yield file_path
    if file_path.exists():
        file_path.unlink()


@pytest.fixture
def temp_video_file(tmp_path):
    file_path = tmp_path / "test.mp4"
    file_path.touch()
    yield file_path
    if file_path.exists():
        file_path.unlink()


def test_verifies_zarr_file_correctly(temp_zarr_file):
    # Create sample segmentation data and metadata
    frames, height, width = 10, 100, 100
    segmentation_data = np.random.randint(0, 10, size=(frames, height, width), dtype=np.uint8)
    metadata = {"frame_step": 1, "palette": [1, 2, 3]}
    
    # Create xarray dataset
    data_array = xr.DataArray(
        segmentation_data,
        dims=["time", "y", "x"],
        coords={
            "time": np.arange(frames),
            "y": np.arange(height),
            "x": np.arange(width)
        }
    )
    dataset = xr.Dataset({"segmentation": data_array})
    
    # Add metadata as attributes
    for key, value in metadata.items():
        dataset.attrs[key] = value
    
    # Save to Zarr
    dataset.to_zarr(temp_zarr_file, mode='w')
    
    # Verify with FileHandler
    mock_config = MagicMock(spec=Config)
    mock_config.frame_step = 1
    assert FileHandler.verify_zarr_file(temp_zarr_file, mock_config) is True


def test_fails_verification_for_invalid_zarr_file(temp_zarr_file):
    # Create sample segmentation data with mismatched frame_step
    frames, height, width = 10, 100, 100
    segmentation_data = np.random.randint(0, 10, size=(frames, height, width), dtype=np.uint8)
    metadata = {"frame_step": 2}  # Different from config
    
    # Create xarray dataset
    data_array = xr.DataArray(
        segmentation_data,
        dims=["time", "y", "x"],
        coords={
            "time": np.arange(frames),
            "y": np.arange(height),
            "x": np.arange(width)
        }
    )
    dataset = xr.Dataset({"segmentation": data_array})
    
    # Add metadata as attributes
    for key, value in metadata.items():
        dataset.attrs[key] = value
    
    # Save to Zarr
    dataset.to_zarr(temp_zarr_file, mode='w')
    
    # Verify with FileHandler using config with different frame_step
    mock_config = MagicMock(spec=Config)
    mock_config.frame_step = 1
    assert FileHandler.verify_zarr_file(temp_zarr_file, mock_config) is False


def test_verifies_video_file_correctly(temp_video_file):
    with patch("cv2.VideoCapture") as mock_capture:
        mock_capture.return_value.isOpened.return_value = True
        mock_capture.return_value.read.side_effect = [
            (True, np.zeros((10, 10, 3))),
            (True, np.zeros((10, 10, 3))),
        ]
        assert FileHandler.verify_video_file(temp_video_file) is True


def test_fails_verification_for_invalid_video_file(temp_video_file):
    with patch("cv2.VideoCapture") as mock_capture:
        mock_capture.return_value.isOpened.return_value = False
        assert FileHandler.verify_video_file(temp_video_file) is False


def test_verifies_analysis_files_correctly(tmp_path):
    counts_file = tmp_path / "counts.txt"
    percentages_file = tmp_path / "percentages.txt"
    counts_file.write_text("data")
    percentages_file.write_text("data")
    assert FileHandler.verify_analysis_files(counts_file, percentages_file) is True


def test_fails_verification_for_empty_analysis_files(tmp_path):
    counts_file = tmp_path / "counts.txt"
    percentages_file = tmp_path / "percentages.txt"
    counts_file.touch()
    percentages_file.touch()
    assert FileHandler.verify_analysis_files(counts_file, percentages_file) is False


def test_verifies_parquet_file_correctly(temp_parquet_file):
    # Create sample analysis data
    data = [
        {"category_id": 0, "pixel_count": 1000, "percentage": 10.0},
        {"category_id": 1, "pixel_count": 2000, "percentage": 20.0},
        {"category_id": 2, "pixel_count": 7000, "percentage": 70.0},
    ]
    df = pd.DataFrame(data)
    
    # Save to Parquet
    df.to_parquet(temp_parquet_file, index=False)
    
    # Verify with FileHandler
    assert FileHandler.verify_parquet_file(temp_parquet_file) is True


def test_fails_verification_for_invalid_parquet_file(temp_parquet_file):
    # Create sample data with missing required columns
    data = [
        {"category_id": 0, "pixel_count": 1000},  # Missing percentage column
        {"category_id": 1, "pixel_count": 2000},
        {"category_id": 2, "pixel_count": 7000},
    ]
    df = pd.DataFrame(data)
    
    # Save to Parquet
    df.to_parquet(temp_parquet_file, index=False)
    
    # Verify with FileHandler
    assert FileHandler.verify_parquet_file(temp_parquet_file) is False


def test_fails_verification_for_empty_parquet_file(temp_parquet_file):
    # Create empty DataFrame
    df = pd.DataFrame()
    
    # Save to Parquet
    df.to_parquet(temp_parquet_file, index=False)
    
    # Verify with FileHandler
    assert FileHandler.verify_parquet_file(temp_parquet_file) is False