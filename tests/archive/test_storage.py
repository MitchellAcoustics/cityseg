"""Tests for storage adapters used in CitySeg."""

import pytest
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# Add parent directory to import path
sys.path.append(str(Path(__file__).parent.parent))

from cityseg.storage.storage import ZarrSegmentationStorage, ParquetAnalysisStorage
from helpers.test_data_generators import create_test_segmentation_data


@pytest.mark.skip(reason="Needs updated implementation for new StorageAdapter API")
def test_zarr_storage(test_temp_dir):
    """Test basic Zarr storage operations."""
    # Create a test dataset
    test_data = create_test_segmentation_data(
        frame_count=3, height=480, width=640, num_classes=19, model_name="test-model"
    )

    # Create storage path
    storage_path = test_temp_dir / "test_segmentation.zarr"

    # Save data
    storage = ZarrSegmentationStorage(storage_path)
    storage.save_segmentation_data(test_data)

    # Verify file exists
    assert os.path.exists(storage_path), "Zarr file not created"

    # Load data back
    loaded_data = storage.load_segmentation_data()

    # Verify data structure
    assert isinstance(loaded_data, xr.Dataset), "Loaded data is not an xarray Dataset"
    assert "segmentation" in loaded_data, "Missing segmentation variable"
    assert loaded_data.dims["frames"] == 3, "Wrong number of frames"
    assert loaded_data.dims["height"] == 480, "Wrong height"
    assert loaded_data.dims["width"] == 640, "Wrong width"

    # Verify data values
    np.testing.assert_array_equal(
        loaded_data.segmentation.values,
        test_data.segmentation.values,
        "Data values don't match",
    )

    # Verify metadata
    assert loaded_data.attrs["model_name"] == "test-model", (
        "Missing model name in attributes"
    )
    assert loaded_data.attrs["frame_count"] == 3, "Wrong frame count in attributes"

    # Test batch loading
    batch = storage.load_segmentation_batch(0, 2)
    assert batch.dims["frames"] == 2, "Wrong batch size"
    np.testing.assert_array_equal(
        batch.segmentation.values,
        test_data.segmentation.values[:2, :, :],
        "Batch values don't match",
    )


@pytest.mark.skip(reason="Needs updated implementation for new StorageAdapter API")
def test_parquet_analysis_storage(test_temp_dir):
    """Test basic Parquet analysis storage operations."""
    # Create test data
    category_counts = pd.DataFrame(
        {
            "frame": [0, 0, 1, 1],
            "category_id": [1, 2, 1, 2],
            "count": [1000, 2000, 1500, 2500],
            "category_name": ["road", "sidewalk", "road", "sidewalk"],
        }
    )

    category_percentages = pd.DataFrame(
        {
            "frame": [0, 0, 1, 1],
            "category_id": [1, 2, 1, 2],
            "percentage": [10.0, 20.0, 15.0, 25.0],
            "category_name": ["road", "sidewalk", "road", "sidewalk"],
        }
    )

    # Create output paths
    counts_path = test_temp_dir / "category_counts.parquet"
    percentages_path = test_temp_dir / "category_percentages.parquet"

    # Save data
    storage = ParquetAnalysisStorage()
    storage.save_category_analysis(category_counts, counts_path)
    storage.save_category_analysis(category_percentages, percentages_path)

    # Verify files exist
    assert os.path.exists(counts_path), "Counts file not created"
    assert os.path.exists(percentages_path), "Percentages file not created"

    # Load data back
    loaded_counts = storage.load_category_analysis(counts_path)
    loaded_percentages = storage.load_category_analysis(percentages_path)

    # Verify data structure
    assert isinstance(loaded_counts, pd.DataFrame), "Loaded counts is not a DataFrame"
    assert isinstance(loaded_percentages, pd.DataFrame), (
        "Loaded percentages is not a DataFrame"
    )

    # Verify data values
    pd.testing.assert_frame_equal(
        loaded_counts, category_counts, "Counts data doesn't match"
    )
    pd.testing.assert_frame_equal(
        loaded_percentages, category_percentages, "Percentages data doesn't match"
    )
