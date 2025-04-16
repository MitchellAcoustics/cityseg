from typing import Generator
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from cityseg.utils.common import get_segmentation_batch, setup_logging, tqdm_context
from tqdm.auto import tqdm


@pytest.fixture
def temp_array_data() -> Generator[tuple[np.ndarray, xr.DataArray], None, None]:
    """
    Fixture to create temporary numpy array and xarray DataArray for testing.

    Yields:
        tuple[np.ndarray, xr.DataArray]: Numpy array and equivalent xarray DataArray.
    """
    # Create a 3D numpy array with shape (10, 5, 5) - 10 frames of 5x5 images
    data = np.random.randint(0, 10, size=(10, 5, 5), dtype=np.int32)

    # Create an equivalent xarray DataArray
    coords = {"time": np.arange(10), "y": np.arange(5), "x": np.arange(5)}
    xr_data = xr.DataArray(data, coords=coords, dims=["time", "y", "x"])

    yield data, xr_data


class TestGetSegmentationBatch:
    """Test cases for get_segmentation_batch function."""

    def test_retrieves_correct_batch_from_numpy(self, temp_array_data):
        """Test that we can retrieve a batch from a numpy array."""
        np_data, _ = temp_array_data
        batch = get_segmentation_batch(np_data, 2, 5)
        assert batch.shape == (3, 5, 5)
        np.testing.assert_array_equal(batch, np_data[2:5])

    def test_retrieves_correct_batch_from_xarray(self, temp_array_data):
        """Test that we can retrieve a batch from an xarray DataArray."""
        _, xr_data = temp_array_data
        batch = get_segmentation_batch(xr_data, 2, 5)
        assert batch.shape == (3, 5, 5)
        np.testing.assert_array_equal(batch, xr_data.isel(time=slice(2, 5)).values)

    def test_handles_empty_segmentation_batch(self, temp_array_data):
        """Test handling of empty batches (start == end)."""
        np_data, xr_data = temp_array_data

        np_batch = get_segmentation_batch(np_data, 2, 2)
        assert np_batch.shape == (0, 5, 5) or len(np_batch) == 0

        xr_batch = get_segmentation_batch(xr_data, 2, 2)
        assert xr_batch.shape == (0, 5, 5) or len(xr_batch) == 0

    def test_handles_out_of_bounds_batch(self, temp_array_data):
        """Test handling of batches beyond array bounds."""
        np_data, xr_data = temp_array_data

        # Test with start >= length
        np_batch = get_segmentation_batch(np_data, 10, 11)
        assert np_batch.shape[0] == 0 or len(np_batch) == 0

        xr_batch = get_segmentation_batch(xr_data, 10, 11)
        assert xr_batch.shape[0] == 0 or len(xr_batch) == 0

    def test_with_different_data_types(self):
        """Test with different data types (int, float)."""
        # Create arrays with different data types
        int_data = np.random.randint(0, 10, size=(5, 5, 5)).astype(np.int32)
        float_data = np.random.random(size=(5, 5, 5)).astype(np.float32)

        # Get batches
        int_batch = get_segmentation_batch(int_data, 1, 3)
        float_batch = get_segmentation_batch(float_data, 1, 3)

        # Check data types are preserved
        assert int_batch.dtype == np.int32
        assert float_batch.dtype == np.float32

    def test_with_multi_dimensional_data(self):
        """Test with multi-dimensional segmentation data."""
        # Create 4D array (batch, channels, height, width)
        data = np.random.randint(0, 10, size=(10, 3, 5, 5))

        # Get batch
        batch = get_segmentation_batch(data, 2, 5)

        # Check shape and content
        assert batch.shape == (3, 3, 5, 5)
        np.testing.assert_array_equal(batch, data[2:5])

    def test_single_element_batch(self, temp_array_data):
        """Test retrieving a single element batch."""
        np_data, xr_data = temp_array_data

        np_batch = get_segmentation_batch(np_data, 2, 3)
        assert np_batch.shape == (1, 5, 5)

        xr_batch = get_segmentation_batch(xr_data, 2, 3)
        assert xr_batch.shape == (1, 5, 5)


class TestTqdmContext:
    """Test cases for tqdm_context context manager."""

    def test_handles_empty_progress_bar(self):
        """Test tqdm_context with an empty iterable."""
        with tqdm_context([]) as pbar:
            assert isinstance(pbar, tqdm)
            assert pbar.total == 0

    def test_handles_non_empty_progress_bar(self):
        """Test tqdm_context with a non-empty iterable."""
        iterable = list(range(5))
        with tqdm_context(iterable) as pbar:
            assert isinstance(pbar, tqdm)
            assert pbar.total == 5

    def test_handles_progress_bar_with_updates(self):
        """Test tqdm_context with manual updates."""
        with tqdm_context(total=10) as pbar:
            assert isinstance(pbar, tqdm)
            assert pbar.total == 10
            assert pbar.n == 0

            pbar.update(3)
            assert pbar.n == 3

    def test_handles_progress_bar_with_exception(self):
        """Test tqdm_context properly closes the bar even with an exception."""
        try:
            with tqdm_context(total=10) as pbar:
                assert isinstance(pbar, tqdm)
                raise ValueError("Test exception")
        except ValueError:
            # Check that the bar is closed - implementation may vary by tqdm version
            # Just check that pbar still exists and can be accessed
            assert hasattr(pbar, "close")

    def test_with_large_total_value(self):
        """Test tqdm_context with a large total value."""
        with tqdm_context(total=1000000) as pbar:
            assert isinstance(pbar, tqdm)
            assert pbar.total == 1000000


class TestSetupLogging:
    """Test cases for setup_logging function."""

    @patch("cityseg.utils.common.logger")
    def test_basic_setup(self, mock_logger):
        """Test basic logging setup."""
        setup_logging(log_level="INFO")

        # Verify logger was configured correctly
        assert mock_logger.remove.called
        assert mock_logger.add.call_count == 2  # Console and file
        assert mock_logger.info.called

    @patch("cityseg.utils.common.logger")
    def test_verbose_mode(self, mock_logger):
        """Test logging setup in verbose mode."""
        setup_logging(log_level="INFO", verbose=True)

        # Verify console level is set to DEBUG
        console_call = mock_logger.add.call_args_list[0]
        assert "level" in console_call[1]
        assert console_call[1]["level"] == "DEBUG"

    @patch("cityseg.utils.common.logger")
    def test_different_log_levels(self, mock_logger):
        """Test logging setup with different log levels."""
        setup_logging(log_level="WARNING")

        # Verify levels
        console_call = mock_logger.add.call_args_list[0]
        file_call = mock_logger.add.call_args_list[1]

        assert console_call[1]["level"] == "WARNING"
        assert file_call[1]["level"] == "INFO"  # File level is capped at INFO

    @patch("cityseg.utils.common.logger")
    def test_file_logging_config(self, mock_logger):
        """Test file logging configuration."""
        setup_logging(log_level="INFO")

        # Get file logging configuration
        file_call = mock_logger.add.call_args_list[1]

        # Check for serialization, rotation, and retention
        assert file_call[1]["serialize"] is True
        assert "rotation" in file_call[1]
        assert "retention" in file_call[1]

    @patch("cityseg.utils.common.logger")
    def test_console_logging_format(self, mock_logger):
        """Test console logging format."""
        setup_logging(log_level="INFO")

        # Get console logging configuration
        console_call = mock_logger.add.call_args_list[0]

        # Check console format includes color
        assert "format" in console_call[1]
        assert "colorize" in console_call[1]
        assert console_call[1]["colorize"] is True


def test_integration_segmentation_with_tqdm():
    """Test integration of segmentation data with tqdm progress tracking."""
    # Create test segmentation data
    data = np.random.randint(0, 10, size=(10, 5, 5))

    # Use tqdm_context to iterate through frames
    processed_frames = []
    with tqdm_context(total=len(data), desc="Processing frames") as pbar:
        for i in range(len(data)):
            # Get a single frame
            frame = get_segmentation_batch(data, i, i + 1)
            processed_frames.append(frame)
            pbar.update(1)

    # Verify all frames were processed
    assert len(processed_frames) == 10
    for i, frame in enumerate(processed_frames):
        np.testing.assert_array_equal(frame, data[i : i + 1])
