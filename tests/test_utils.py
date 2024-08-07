import os
import sys
import tempfile
from typing import Generator
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from cityseg.utils import get_segmentation_data_batch, setup_logging, tqdm_context
from tqdm.auto import tqdm


@pytest.fixture
def temp_hdf5_file() -> Generator[str, None, None]:
    """
    Fixture to create a temporary HDF5 file for testing.

    Yields:
        str: Path to the temporary HDF5 file.
    """
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
        tmp_path = tmp.name
    yield tmp_path
    os.unlink(tmp_path)


class TestGetSegmentationDataBatch:
    """Tests for the get_segmentation_data_batch function."""

    def test_retrieves_correct_batch_of_segmentation_data(
        self, temp_hdf5_file: str
    ) -> None:
        """
        Test that the function retrieves the correct batch of segmentation data.

        Args:
            temp_hdf5_file (str): Path to temporary HDF5 file.
        """
        data = np.random.rand(100, 100)
        with h5py.File(temp_hdf5_file, "w") as f:
            dset = f.create_dataset("segmentation", data=data)
            batch = get_segmentation_data_batch(dset, 10, 20)
            assert batch.shape == (10, 100)
            assert np.array_equal(batch, data[10:20])

    def test_handles_empty_segmentation_data_batch(self, temp_hdf5_file: str) -> None:
        """
        Test that the function correctly handles an empty batch request.

        Args:
            temp_hdf5_file (str): Path to temporary HDF5 file.
        """
        data = np.random.rand(100, 100)
        with h5py.File(temp_hdf5_file, "w") as f:
            dset = f.create_dataset("segmentation", data=data)
            batch = get_segmentation_data_batch(dset, 0, 0)
            assert batch.shape == (0, 100)

    def test_handles_out_of_bounds_segmentation_data_batch(
        self, temp_hdf5_file: str
    ) -> None:
        """
        Test that the function correctly handles out-of-bounds batch requests.

        Args:
            temp_hdf5_file (str): Path to temporary HDF5 file.
        """
        data = np.random.rand(100, 100)
        with h5py.File(temp_hdf5_file, "w") as f:
            dset = f.create_dataset("segmentation", data=data)
            batch = get_segmentation_data_batch(dset, 90, 110)
            assert batch.shape == (10, 100)
            assert np.array_equal(batch, data[90:100])

    def test_with_different_data_types(self, temp_hdf5_file: str) -> None:
        """
        Test the function with different data types (int and float).

        Args:
            temp_hdf5_file (str): Path to temporary HDF5 file.
        """
        int_data = np.random.randint(0, 100, (100, 100))
        float_data = np.random.rand(100, 100)

        with h5py.File(temp_hdf5_file, "w") as f:
            int_dset = f.create_dataset("int_segmentation", data=int_data)
            float_dset = f.create_dataset("float_segmentation", data=float_data)

            int_batch = get_segmentation_data_batch(int_dset, 10, 20)
            float_batch = get_segmentation_data_batch(float_dset, 10, 20)

            assert int_batch.dtype == np.int64
            assert float_batch.dtype == np.float64
            assert np.array_equal(int_batch, int_data[10:20])
            assert np.array_equal(float_batch, float_data[10:20])

    def test_with_multi_dimensional_data(self, temp_hdf5_file: str) -> None:
        """
        Test the function with multi-dimensional data.

        Args:
            temp_hdf5_file (str): Path to temporary HDF5 file.
        """
        data = np.random.rand(
            100, 50, 50, 3
        )  # 4D data (frames, height, width, channels)
        with h5py.File(temp_hdf5_file, "w") as f:
            dset = f.create_dataset("multi_dim_segmentation", data=data)
            batch = get_segmentation_data_batch(dset, 10, 20)
            assert batch.shape == (10, 50, 50, 3)
            assert np.array_equal(batch, data[10:20])

    def test_single_element_batch(self, temp_hdf5_file: str) -> None:
        """
        Test the function with a single-element batch.

        Args:
            temp_hdf5_file (str): Path to temporary HDF5 file.
        """
        data = np.random.rand(100, 100)
        with h5py.File(temp_hdf5_file, "w") as f:
            dset = f.create_dataset("segmentation", data=data)
            batch = get_segmentation_data_batch(dset, 10, 11)
            assert batch.shape == (1, 100)
            assert np.array_equal(batch, data[10:11])


class TestTqdmContext:
    """Tests for the tqdm_context function."""

    def test_handles_empty_progress_bar(self) -> None:
        """Test that the function correctly handles an empty progress bar."""
        with tqdm_context(total=0) as progress_bar:
            assert isinstance(progress_bar, tqdm)
            assert progress_bar.total == 0

    def test_handles_non_empty_progress_bar(self) -> None:
        """Test that the function correctly handles a non-empty progress bar."""
        with tqdm_context(total=100) as progress_bar:
            assert isinstance(progress_bar, tqdm)
            assert progress_bar.total == 100

    def test_handles_progress_bar_with_updates(self) -> None:
        """Test that the function correctly handles progress bar updates."""
        with tqdm_context(total=100) as progress_bar:
            progress_bar.update(10)
            assert progress_bar.n == 10
            progress_bar.update(20)
            assert progress_bar.n == 30

    def test_handles_progress_bar_with_exception(self) -> None:
        """Test that the function correctly handles exceptions within the context."""
        with pytest.raises(ValueError, match="Test exception"):
            with tqdm_context(total=100):
                raise ValueError("Test exception")

    def test_with_large_total_value(self) -> None:
        """Test that the function handles a very large total value."""
        large_total = 10**10
        with tqdm_context(total=large_total) as progress_bar:
            assert progress_bar.total == large_total
            progress_bar.update(large_total // 2)
            assert progress_bar.n == large_total // 2


class TestSetupLogging:
    """
    Test suite for the setup_logging function in utils.py.

    This class contains tests to verify the correct behavior of the logging setup,
    including console and file logging configurations, different log levels,
    and formatting options.
    """

    @pytest.fixture
    def mock_logger(self):
        """
        Fixture that provides a mock logger for testing.

        Returns:
            MockLogger: A mock object that simulates the behavior of the loguru logger.
        """

        class MockLogger:
            def __init__(self):
                self.handlers = []

            def remove(self, handler_id=None):
                print("MockLogger: remove() called")

            def add(self, sink, **kwargs):
                print(f"MockLogger: add() called with sink={sink}, kwargs={kwargs}")
                self.handlers.append((sink, kwargs))

            def info(self, message):
                print(f"MockLogger: info() called with message: {message}")

        return MockLogger()

    @patch("cityseg.utils.logger")
    def test_basic_setup(self, mock_logger):
        """
        Test the basic setup of logging with default parameters.

        Args:
            mock_logger: The mocked logger object.
        """
        setup_logging("INFO")
        assert mock_logger.remove.called, "logger.remove() should be called"
        assert (
            len(mock_logger.add.call_args_list) == 2
        ), "Should add console and file handlers"

        # Verify console handler
        console_call = mock_logger.add.call_args_list[0]
        assert console_call[0][0] == sys.stderr, "Console handler should use sys.stderr"
        assert console_call[1]["level"] == "INFO", "Console level should be INFO"

        # Verify file handler
        file_call = mock_logger.add.call_args_list[1]
        assert (
            file_call[0][0] == "segmentation.log"
        ), "File handler should use 'segmentation.log'"
        assert file_call[1]["level"] == "INFO", "File level should be INFO"

    @patch("cityseg.utils.logger")
    def test_verbose_mode(self, mock_logger):
        """
        Test the logging setup in verbose mode.

        Args:
            mock_logger: The mocked logger object.
        """
        setup_logging("INFO", verbose=True)
        console_call = mock_logger.add.call_args_list[0]
        assert (
            console_call[1]["level"] == "DEBUG"
        ), "Console level should be DEBUG in verbose mode"

    @patch("cityseg.utils.logger")
    def test_different_log_levels(self, mock_logger):
        """
        Test the logging setup with different log levels.

        Args:
            mock_logger: The mocked logger object.
        """
        setup_logging("DEBUG")
        console_call = mock_logger.add.call_args_list[0]
        file_call = mock_logger.add.call_args_list[1]
        assert console_call[1]["level"] == "DEBUG", "Console level should be DEBUG"
        assert (
            file_call[1]["level"] == "DEBUG"
        ), "File level should be DEBUG (minimum of input and INFO)"

        setup_logging("WARNING")
        console_call = mock_logger.add.call_args_list[2]
        file_call = mock_logger.add.call_args_list[3]
        assert console_call[1]["level"] == "WARNING", "Console level should be WARNING"
        assert (
            file_call[1]["level"] == "INFO"
        ), "File level should be INFO (minimum of input and INFO)"

    @patch("cityseg.utils.logger")
    def test_file_logging_config(self, mock_logger):
        """
        Test the file logging configuration.

        Args:
            mock_logger: The mocked logger object.
        """
        setup_logging("INFO")
        file_call = mock_logger.add.call_args_list[1]
        assert file_call[1]["rotation"] == "100 MB", "File should rotate at 100 MB"
        assert (
            file_call[1]["retention"] == "1 week"
        ), "File should be retained for 1 week"
        assert file_call[1]["serialize"] is True, "File logging should be serialized"

    @patch("cityseg.utils.logger")
    def test_console_logging_format(self, mock_logger):
        """
        Test the console logging format.

        Args:
            mock_logger: The mocked logger object.
        """
        setup_logging("INFO")
        console_call = mock_logger.add.call_args_list[0]
        format_string = console_call[1]["format"]
        assert "<green>{time:YYYY-MM-DD HH:mm:ss}</green>" in format_string
        assert "<level>{level: <8}</level>" in format_string
        assert (
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
            in format_string
        )


def test_integration_segmentation_with_tqdm(temp_hdf5_file: str) -> None:
    """
    Integration test for using get_segmentation_data_batch within a tqdm_context.

    Args:
        temp_hdf5_file (str): Path to temporary HDF5 file.
    """
    data = np.random.rand(100, 100)
    with h5py.File(temp_hdf5_file, "w") as f:
        dset = f.create_dataset("segmentation", data=data)

        with tqdm_context(total=len(data), desc="Processing") as pbar:
            for i in range(0, len(data), 10):
                batch = get_segmentation_data_batch(dset, i, min(i + 10, len(data)))
                assert batch.shape[0] <= 10
                pbar.update(len(batch))

        assert pbar.n == len(data)
