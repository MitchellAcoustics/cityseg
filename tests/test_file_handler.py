import json
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from src.cityseg.config import Config
from src.cityseg.file_handler import FileHandler


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
        assert np.array_equal(f["segmentation"], segmentation_data)
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
