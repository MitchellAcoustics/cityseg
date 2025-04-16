"""
Tests for the storage adapter module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from cityseg.storage_adapter import ZarrSegmentationStorage, ParquetAnalysisStorage


@pytest.fixture
def sample_segmentation_data():
    """Sample segmentation data for testing."""
    # Create a 3D array (frames, height, width)
    return np.random.randint(0, 10, size=(5, 100, 200), dtype=np.uint8)


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "model_name": "test_model",
        "fps": 30.0,
        "frame_step": 1,
        "label_ids": {"0": "background", "1": "car", "2": "person"},
        "palette": np.random.randint(0, 255, size=(10, 3), dtype=np.uint8),
    }


@pytest.fixture
def sample_xarray_dataset(sample_segmentation_data, sample_metadata):
    """Sample xarray dataset for testing."""
    frames, height, width = sample_segmentation_data.shape
    data_array = xr.DataArray(
        sample_segmentation_data,
        dims=["time", "y", "x"],
        coords={
            "time": np.arange(frames),
            "y": np.arange(height),
            "x": np.arange(width),
        },
    )
    dataset = xr.Dataset({"segmentation": data_array})

    # Add metadata as attributes
    for key, value in sample_metadata.items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, dict) and any(isinstance(k, int) for k in value.keys()):
            value = {str(k): v for k, v in value.items()}
        dataset.attrs[key] = value

    return dataset


class TestZarrSegmentationStorage:
    """Tests for ZarrSegmentationStorage."""

    def test_save_and_load_segmentation_data(self, sample_segmentation_data, sample_metadata):
        """Test saving and loading segmentation data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize storage
            storage = ZarrSegmentationStorage()
            output_path = Path(tmpdir) / "test_segmentation"

            # Save data
            saved_path = storage.save_segmentation_data(
                sample_segmentation_data, sample_metadata, output_path
            )

            # Verify file exists
            assert saved_path.exists()
            assert saved_path.suffix == ".zarr"

            # Load data
            loaded_data, loaded_metadata = storage.load_segmentation_data(saved_path)

            # Verify data
            assert isinstance(loaded_data, xr.Dataset)
            assert "segmentation" in loaded_data
            np.testing.assert_array_equal(
                loaded_data.segmentation.values, sample_segmentation_data
            )

            # Verify metadata
            assert loaded_metadata["model_name"] == sample_metadata["model_name"]
            assert loaded_metadata["fps"] == sample_metadata["fps"]
            assert loaded_metadata["frame_step"] == sample_metadata["frame_step"]

    def test_save_and_load_xarray_dataset(self, sample_xarray_dataset):
        """Test saving and loading xarray dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize storage
            storage = ZarrSegmentationStorage()
            output_path = Path(tmpdir) / "test_segmentation"

            # Save data
            saved_path = storage.save_segmentation_data(
                sample_xarray_dataset, dict(sample_xarray_dataset.attrs), output_path
            )

            # Verify file exists
            assert saved_path.exists()
            assert saved_path.suffix == ".zarr"

            # Load data
            loaded_data, loaded_metadata = storage.load_segmentation_data(saved_path)

            # Verify data
            assert isinstance(loaded_data, xr.Dataset)
            assert "segmentation" in loaded_data
            np.testing.assert_array_equal(
                loaded_data.segmentation.values, sample_xarray_dataset.segmentation.values
            )

            # Verify metadata
            assert loaded_metadata["model_name"] == sample_xarray_dataset.attrs["model_name"]
            assert loaded_metadata["fps"] == sample_xarray_dataset.attrs["fps"]

    def test_load_segmentation_batch(self, sample_xarray_dataset):
        """Test loading a batch of segmentation data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize storage
            storage = ZarrSegmentationStorage()
            output_path = Path(tmpdir) / "test_segmentation"

            # Save data
            saved_path = storage.save_segmentation_data(
                sample_xarray_dataset, dict(sample_xarray_dataset.attrs), output_path
            )

            # Load a batch (frames 1-3)
            batch = storage.load_segmentation_batch(saved_path, 1, 3)

            # Verify batch
            assert isinstance(batch, xr.Dataset)
            assert "segmentation" in batch
            assert batch.sizes["time"] == 2  # End index is exclusive
            np.testing.assert_array_equal(
                batch.segmentation.values,
                sample_xarray_dataset.segmentation.isel(time=slice(1, 3)).values,
            )


class TestParquetAnalysisStorage:
    """Tests for ParquetAnalysisStorage."""

    def test_save_and_load_category_analysis(self):
        """Test saving and loading category analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize storage
            storage = ParquetAnalysisStorage()
            output_path = Path(tmpdir) / "test_analysis"

            # Sample data
            counts = {0: 5000, 1: 3000, 2: 2000}
            percentages = {0: 50.0, 1: 30.0, 2: 20.0}

            # Save data
            saved_path = storage.save_category_analysis(
                counts, percentages, output_path, frame_idx=0
            )

            # Verify file exists
            assert saved_path.exists()
            assert saved_path.suffix == ".parquet"

            # Load data
            loaded_df = storage.load_category_analysis(saved_path)

            # Verify data
            assert len(loaded_df) == 3  # 3 categories
            assert set(loaded_df.columns) == {"category_id", "pixel_count", "percentage", "frame_idx"}
            assert set(loaded_df["category_id"]) == {0, 1, 2}
            assert loaded_df.loc[loaded_df["category_id"] == 0, "pixel_count"].iloc[0] == 5000
            assert loaded_df.loc[loaded_df["category_id"] == 1, "percentage"].iloc[0] == 30.0

    def test_save_video_analysis(self, sample_segmentation_data):
        """Test saving video analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize storage
            storage = ParquetAnalysisStorage()
            output_path = Path(tmpdir) / "test_video_analysis"

            # Sample metadata
            metadata = {"model_name": "test_model", "fps": 30.0}

            # Save data
            saved_path = storage.save_video_analysis(
                sample_segmentation_data, metadata, output_path
            )

            # Verify file exists
            assert saved_path.exists()
            assert saved_path.suffix == ".parquet"

            # Load data
            loaded_df = storage.load_category_analysis(saved_path)

            # Verify data structure
            assert "frame_idx" in loaded_df.columns
            assert "category_id" in loaded_df.columns
            assert "pixel_count" in loaded_df.columns
            assert "percentage" in loaded_df.columns

            # Verify all frames are represented
            assert set(loaded_df["frame_idx"]) == set(range(sample_segmentation_data.shape[0]))