import csv
from pathlib import Path
from typing import Any, Dict
from unittest.mock import mock_open, patch

import h5py
import numpy as np
import pandas as pd
import pytest
from cityseg.segmentation_analyzer import SegmentationAnalyzer


class TestSegmentationAnalyzer:
    """
    Test suite for the SegmentationAnalyzer class.

    This class contains tests to verify the correct behavior of the SegmentationAnalyzer
    methods, including segmentation map analysis, results analysis, and statistics generation.
    """

    @pytest.fixture
    def sample_segmentation_map(self) -> np.ndarray:
        """
        Fixture that provides a sample segmentation map for testing.

        Returns:
            np.ndarray: A 2D numpy array representing a sample segmentation map.
        """
        return np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])

    @pytest.fixture
    def sample_metadata(self) -> Dict[str, Any]:
        """
        Fixture that provides sample metadata for testing.

        Returns:
            Dict[str, Any]: A dictionary containing sample metadata.
        """
        return {
            "label_ids": {0: "background", 1: "foreground", 2: "edge"},
            "frame_step": 5,
        }

    def test_analyze_segmentation_map(self, sample_segmentation_map):
        """
        Test the analyze_segmentation_map method.

        This test verifies that the method correctly computes pixel counts
        and percentages for each category in the segmentation map.

        Args:
            sample_segmentation_map (np.ndarray): The sample segmentation map fixture.
        """
        result = SegmentationAnalyzer.analyze_segmentation_map(
            sample_segmentation_map, 3
        )
        expected = {0: (3, 100 / 3), 1: (3, 100 / 3), 2: (3, 100 / 3)}

        assert len(result) == len(expected), "Number of categories does not match"
        for category in expected:
            assert category in result, f"Category {category} is missing from the result"
            assert (
                result[category][0] == expected[category][0]
            ), f"Pixel count for category {category} does not match"
            assert np.isclose(
                result[category][1], expected[category][1], rtol=1e-9
            ), f"Percentage for category {category} is not close enough"

    @patch("cityseg.segmentation_analyzer.get_segmentation_data_batch")
    @patch("cityseg.segmentation_analyzer.open", new_callable=mock_open)
    @patch("cityseg.segmentation_analyzer.csv.writer")
    @patch("cityseg.segmentation_analyzer.logger")
    @patch("cityseg.segmentation_analyzer.pd.read_csv")
    @patch("cityseg.segmentation_analyzer.SegmentationAnalyzer.generate_category_stats")
    def test_analyze_results(
        self,
        mock_generate_stats,
        mock_read_csv,
        mock_logger,
        mock_csv_writer,
        mock_open,
        mock_get_data,
        sample_metadata,
        tmp_path,
    ):
        """
        Test the analyze_results method.

        This test verifies that the method correctly processes segmentation data,
        writes results to CSV files, and generates statistics.

        Args:
            mock_generate_stats: Mocked generate_category_stats method.
            mock_read_csv: Mocked pandas read_csv function.
            mock_logger: Mocked logger object.
            mock_csv_writer: Mocked CSV writer object.
            mock_open: Mocked open function.
            mock_get_data: Mocked get_segmentation_data_batch function.
            sample_metadata (Dict[str, Any]): The sample metadata fixture.
            tmp_path (Path): Pytest fixture for a temporary directory path.
        """
        mock_segmentation_data = mock_get_data.return_value
        mock_segmentation_data.__len__.return_value = 10
        mock_get_data.return_value = np.array([[[0, 1], [1, 2]]] * 10)

        # Mock the DataFrame that would be read from the CSV
        mock_df = pd.DataFrame(
            {
                "Frame": [0, 5],
                "background": [50, 60],
                "foreground": [30, 25],
                "edge": [20, 15],
            }
        )
        mock_read_csv.return_value = mock_df

        output_path = tmp_path / "test_output.h5"
        SegmentationAnalyzer.analyze_results(
            mock_segmentation_data, sample_metadata, output_path
        )

        assert mock_open.call_count == 2, "Should open two files for writing"
        assert mock_csv_writer.call_count == 2, "Should create two CSV writers"
        assert (
            mock_generate_stats.call_count == 2
        ), "Should call generate_category_stats twice"

    @patch("cityseg.segmentation_analyzer.pd.read_csv")
    @patch("cityseg.segmentation_analyzer.logger")
    def test_generate_category_stats(self, mock_logger, mock_read_csv, tmp_path):
        """
        Test the generate_category_stats method.

        This test verifies that the method correctly reads input data,
        computes statistics, and saves the results.

        Args:
            mock_logger: Mocked logger object.
            mock_read_csv: Mocked pandas read_csv function.
            tmp_path (Path): Pytest fixture for a temporary directory path.
        """
        mock_df = pd.DataFrame(
            {
                "Frame": [0, 5, 10],
                "background": [50, 60, 70],
                "foreground": [30, 25, 20],
                "edge": [20, 15, 10],
            }
        )
        mock_read_csv.return_value = mock_df

        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"

        SegmentationAnalyzer.generate_category_stats(input_file, output_file)

        mock_read_csv.assert_called_once_with(input_file)
        assert output_file.exists(), "Output file should be created"
        mock_logger.info.assert_called_once()

    def test_generate_category_stats_error_handling(self, tmp_path):
        """
        Test error handling in the generate_category_stats method.

        This test verifies that the method properly handles and logs errors
        when processing fails.

        Args:
            tmp_path (Path): Pytest fixture for a temporary directory path.
        """
        non_existent_file = tmp_path / "non_existent.csv"
        output_file = tmp_path / "output.csv"

        with pytest.raises(Exception):
            SegmentationAnalyzer.generate_category_stats(non_existent_file, output_file)
