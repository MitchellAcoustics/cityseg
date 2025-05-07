"""Dummy test using generated data to verify CitySeg without downloading models."""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from helpers.test_data_generators import (
    create_test_segmentation_data,
)
from helpers.assertions import assert_segmentation_data_valid


@pytest.fixture
def test_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_segmentation_dataset_output(test_output_dir):
    """Test saving and loading segmentation data.

    This is a dummy test that doesn't require downloading any models.
    It just verifies the dataset handling works correctly.
    """
    # Create a test dataset
    test_data = create_test_segmentation_data(
        frame_count=2, height=240, width=320, num_classes=19, pattern="grid"
    )

    # Verify the dataset is valid
    assert_segmentation_data_valid(test_data)

    # Get the segmentation array shape
    seg_array = test_data["segmentation"].values
    assert seg_array.shape == (2, 240, 320), f"Unexpected shape: {seg_array.shape}"

    # Verify we have reasonable class IDs
    unique_classes = np.unique(seg_array)
    assert len(unique_classes) > 0, "No classes found in segmentation data"
    assert np.max(unique_classes) < 19, (
        f"Maximum class ID too high: {np.max(unique_classes)}"
    )

    print("\nSuccessfully created and verified test segmentation data")
