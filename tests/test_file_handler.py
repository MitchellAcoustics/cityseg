import json
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from cityseg.config import Config
from cityseg.file_handler import FileHandler


@pytest.fixture
def temp_hdf_file(tmp_path):
    file_path = tmp_path / "test.hdf5"
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


def test_saves_hdf_file_correctly(temp_hdf_file):
    segmentation_data = np.random.rand(10, 10)
    metadata = {"frame_step": 1, "palette": np.array([1, 2, 3])}
    FileHandler.save_hdf_file(temp_hdf_file, segmentation_data, metadata)
    with h5py.File(temp_hdf_file, "r") as f:
        assert "segmentation" in f
        assert "metadata" in f
        # Check that the values are correctly rounded
        expected_data = np.round(segmentation_data).astype(np.int32)
        assert np.array_equal(f["segmentation"], expected_data)
        loaded_metadata = json.loads(f["metadata"][()])
        assert loaded_metadata["frame_step"] == 1
        assert loaded_metadata["palette"] == [1, 2, 3]


def test_loads_hdf_file_correctly(temp_hdf_file):
    segmentation_data = np.random.rand(10, 10)
    metadata = {"frame_step": 1, "palette": [1, 2, 3]}
    with h5py.File(temp_hdf_file, "w") as f:
        f.create_dataset("segmentation", data=segmentation_data)
        f.create_dataset("metadata", data=json.dumps(metadata))
    hdf_file, loaded_metadata = FileHandler.load_hdf_file(temp_hdf_file)
    assert np.array_equal(hdf_file["segmentation"], segmentation_data)
    assert loaded_metadata["frame_step"] == 1
    assert np.array_equal(loaded_metadata["palette"], np.array([1, 2, 3]))


def test_verifies_hdf_file_correctly(temp_hdf_file):
    segmentation_data = np.random.rand(10, 10)
    metadata = {"frame_step": 1}
    with h5py.File(temp_hdf_file, "w") as f:
        f.create_dataset("segmentation", data=segmentation_data)
        f.create_dataset("metadata", data=json.dumps(metadata))
    mock_config = MagicMock(spec=Config)
    mock_config.frame_step = 1
    assert FileHandler.verify_hdf_file(temp_hdf_file, mock_config) is True


def test_fails_verification_for_invalid_hdf_file(temp_hdf_file):
    segmentation_data = np.random.rand(10, 10)
    metadata = {"frame_step": 2}
    with h5py.File(temp_hdf_file, "w") as f:
        f.create_dataset("segmentation", data=segmentation_data)
        f.create_dataset("metadata", data=json.dumps(metadata))
    mock_config = MagicMock(spec=Config)
    mock_config.frame_step = 1
    assert FileHandler.verify_hdf_file(temp_hdf_file, mock_config) is False


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


def test_saves_float_segmentation_as_integers(temp_hdf_file):
    """
    Test that floating-point segmentation data is correctly converted to integers when saving.

    This tests the fix for the issue where segmentation maps with floating-point values
    caused errors when used as indices into the palette array during visualization.
    """
    # Create segmentation data with floating-point values
    segmentation_data = np.random.uniform(0, 5, (10, 10)).astype(np.float32)
    metadata = {
        "frame_step": 1,
        "palette": np.array(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [0, 255, 255],
                [255, 0, 255],
            ]
        ),
    }

    # Save the data
    FileHandler.save_hdf_file(temp_hdf_file, segmentation_data, metadata)

    # Verify that the segmentation data was converted to integers
    with h5py.File(temp_hdf_file, "r") as f:
        assert "segmentation" in f
        loaded_data = f["segmentation"][()]

        # Check that the loaded data is an integer type
        assert np.issubdtype(loaded_data.dtype, np.integer)

        # Check that the values are correctly rounded
        expected_data = np.round(segmentation_data).astype(np.int32)
        assert np.array_equal(loaded_data, expected_data)


def test_update_hdf_file_converts_float_to_int(temp_hdf_file):
    """
    Test that the update_hdf_file method correctly converts floating-point segmentation data to integers.
    """
    # Create initial data
    initial_data = np.zeros((5, 10), dtype=np.int32)
    metadata = {"frame_step": 1, "palette": [[255, 0, 0], [0, 255, 0]]}

    # Create the initial file
    with h5py.File(temp_hdf_file, "w") as f:
        f.create_dataset("segmentation", data=initial_data, maxshape=(None, 10))
        f.create_dataset("metadata", data=json.dumps(metadata))

    # Create new floating-point data to add
    new_data = np.random.uniform(0, 5, (5, 10)).astype(np.float32)

    # Update the file with floating-point data
    FileHandler.update_hdf_file(temp_hdf_file, new_data, 10, metadata)

    # Verify that the added data was converted to integers
    with h5py.File(temp_hdf_file, "r") as f:
        assert f["segmentation"].shape == (10, 10)
        added_data = f["segmentation"][5:]

        # Check that the added data is an integer type
        assert np.issubdtype(added_data.dtype, np.integer)

        # Check that the values are correctly rounded
        expected_data = np.round(new_data).astype(np.int32)
        assert np.array_equal(added_data, expected_data)
