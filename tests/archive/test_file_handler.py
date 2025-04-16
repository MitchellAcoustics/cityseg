"""Tests for FileHandler utility."""

import pytest
import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to import path
sys.path.append(str(Path(__file__).parent.parent))

from cityseg.storage.storage import FileHandler
from helpers.test_data_generators import create_test_segmentation_data


@pytest.mark.skip(reason="Needs updated implementation for new FileHandler API")
def test_zarr_validation(test_temp_dir):
    """Test zarr file validation."""
    # Create a test dataset
    test_data = create_test_segmentation_data(
        frame_count=3, height=480, width=640, num_classes=19, model_name="test-model"
    )

    # Create storage path
    zarr_path = test_temp_dir / "test_validation.zarr"

    # Save data to zarr
    test_data.to_zarr(zarr_path)

    # Verify file exists
    assert os.path.exists(zarr_path), "Zarr file not created"

    # Test valid zarr validation
    result = FileHandler.verify_zarr_file(zarr_path)
    assert result is True, "Valid zarr file failed validation"

    # Test invalid zarr validation
    invalid_path = test_temp_dir / "nonexistent.zarr"
    result = FileHandler.verify_zarr_file(invalid_path)
    assert result is False, "Nonexistent zarr file passed validation"


@pytest.mark.skip(reason="Needs updated implementation for new FileHandler API")
def test_parquet_validation(test_temp_dir):
    """Test parquet file validation."""
    # Create test DataFrame
    df = pd.DataFrame(
        {"frame": [0, 1, 2], "category_id": [1, 2, 3], "count": [100, 200, 300]}
    )

    # Create storage path
    parquet_path = test_temp_dir / "test_validation.parquet"

    # Save data to parquet
    df.to_parquet(parquet_path)

    # Verify file exists
    assert os.path.exists(parquet_path), "Parquet file not created"

    # Test valid parquet validation
    result = FileHandler.verify_parquet_file(parquet_path)
    assert result is True, "Valid parquet file failed validation"

    # Test invalid parquet validation
    invalid_path = test_temp_dir / "nonexistent.parquet"
    result = FileHandler.verify_parquet_file(invalid_path)
    assert result is False, "Nonexistent parquet file passed validation"
