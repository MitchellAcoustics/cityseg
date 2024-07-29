"""
This module provides a class for analyzing segmentation results.

It includes methods to analyze segmentation maps, compute pixel counts and percentages
for each category, and generate statistics for the analysis results.

Classes:
    SegmentationAnalyzer: A class for analyzing segmentation results.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from loguru import logger

from cityseg.utils import get_segmentation_data_batch


class SegmentationAnalyzer:
    """
    A class for analyzing segmentation results.

    This class provides methods to analyze segmentation maps, compute pixel counts and
    percentages for each category, and generate statistics for the analysis results.

    Methods:
        analyze_segmentation_map: Analyzes a segmentation map to compute pixel counts and percentages.
        analyze_results: Analyzes segmentation data and saves counts and percentages to CSV files.
        generate_category_stats: Generates statistics for category counts or percentages.
    """

    @staticmethod
    def analyze_segmentation_map(
        seg_map: np.ndarray, num_categories: int
    ) -> Dict[int, tuple[int, float]]:
        """
        Analyzes a segmentation map to compute pixel counts and percentages for each category.

        Args:
            seg_map (np.ndarray): The segmentation map to analyze.
            num_categories (int): The total number of categories in the segmentation.

        Returns:
            Dict[int, Tuple[int, float]]: A dictionary where keys are category IDs and values
            are tuples of (pixel count, percentage) for each category.
        """
        unique, counts = np.unique(seg_map, return_counts=True)
        total_pixels = seg_map.size
        category_analysis = {i: (0, 0.0) for i in range(num_categories)}

        for category_id, pixel_count in zip(unique, counts):
            percentage = (pixel_count / total_pixels) * 100
            category_analysis[int(category_id)] = (int(pixel_count), float(percentage))

        return category_analysis

    @staticmethod
    def analyze_results(
        segmentation_data: h5py.Dataset, metadata: Dict[str, Any], output_path: Path
    ) -> None:
        """
        Analyzes segmentation data and saves counts and percentages to CSV files.

        This method processes the segmentation data in chunks, computes the analysis for each
        frame, and writes the results to separate CSV files for counts and percentages.

        Args:
            segmentation_data (h5py.Dataset): The segmentation data to analyze.
            metadata (Dict[str, Any]): Metadata containing label IDs and frame step.
            output_path (Path): The path where the output CSV files will be saved.
        """
        counts_file = output_path.with_name(f"{output_path.stem}_category_counts.csv")
        percentages_file = output_path.with_name(
            f"{output_path.stem}_category_percentages.csv"
        )

        id2label = metadata["label_ids"]
        headers = ["Frame"] + [id2label[i] for i in sorted(id2label.keys())]

        chunk_size = 100  # Adjust based on memory constraints

        with open(counts_file, "w", newline="") as cf, open(
            percentages_file, "w", newline=""
        ) as pf:
            counts_writer = csv.writer(cf)
            percentages_writer = csv.writer(pf)
            counts_writer.writerow(headers)
            percentages_writer.writerow(headers)

            for chunk_start in range(0, len(segmentation_data), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(segmentation_data))
                seg_chunk = get_segmentation_data_batch(
                    segmentation_data, chunk_start, chunk_end
                )

                for frame_idx, seg_map in enumerate(seg_chunk, start=chunk_start):
                    analysis = SegmentationAnalyzer.analyze_segmentation_map(
                        seg_map, len(id2label)
                    )
                    frame_number = frame_idx * metadata["frame_step"]

                    counts_row = [frame_number] + [
                        analysis[i][0] for i in sorted(analysis.keys())
                    ]
                    percentages_row = [frame_number] + [
                        analysis[i][1] for i in sorted(analysis.keys())
                    ]

                    counts_writer.writerow(counts_row)
                    percentages_writer.writerow(percentages_row)

        logger.info(f"Category counts saved to {counts_file}")
        logger.info(f"Category percentages saved to {percentages_file}")

        SegmentationAnalyzer.generate_category_stats(
            counts_file, output_path.with_name(f"{output_path.stem}_counts_stats.csv")
        )
        SegmentationAnalyzer.generate_category_stats(
            percentages_file,
            output_path.with_name(f"{output_path.stem}_percentages_stats.csv"),
        )

    @staticmethod
    def generate_category_stats(input_file: Path, output_file: Path) -> None:
        """
        Generates statistics for category counts or percentages.

        This method reads the input CSV file, computes statistics (mean, median, std, min, max)
        for each category, and saves the results to the specified output file.

        Args:
            input_file (Path): Path to the input CSV file containing category data.
            output_file (Path): Path to save the generated statistics.
        """
        try:
            df = pd.read_csv(input_file)
            category_columns = df.columns[1:]
            stats = df[category_columns].agg(["mean", "median", "std", "min", "max"])
            stats = stats.transpose()
            stats.to_csv(output_file)
            logger.info(f"Category statistics saved to {output_file}")
        except Exception as e:
            logger.error(f"Error generating category stats: {str(e)}")
            raise
