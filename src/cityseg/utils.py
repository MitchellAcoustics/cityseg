"""
This module provides utility functions for the semantic segmentation pipeline.

It includes functions for analyzing segmentation maps, generating statistics,
handling file operations, and setting up logging. These utilities are used
throughout the segmentation pipeline to support various processing tasks.
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from tqdm.asyncio import tqdm


def analyze_segmentation_map(
    seg_map: np.ndarray, num_categories: int
) -> Dict[int, Tuple[int, float]]:
    """
    Analyze a segmentation map to compute pixel counts and percentages for each category.

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


def generate_category_stats(csv_file_path: Path) -> pd.DataFrame:
    """
    Generate statistical summaries for each category from a CSV file.

    Args:
        csv_file_path (Path): Path to the CSV file containing category data.

    Returns:
        pd.DataFrame: A DataFrame containing statistical summaries for each category.
    """
    df = pd.read_csv(csv_file_path)
    category_columns = df.columns[1:]

    stats = []
    for category in category_columns:
        category_data = df[category]
        stats.append(
            {
                "Category": category,
                "Mean": category_data.mean(),
                "Median": category_data.median(),
                "Std Dev": category_data.std(),
                "Min": category_data.min(),
                "Max": category_data.max(),
            }
        )

    return pd.DataFrame(stats)


def initialize_csv_files(
    output_prefix: Path, category_names: Dict[int, str]
) -> Tuple[Path, Path]:
    """
    Initialize CSV files for storing category counts and percentages.

    Args:
        output_prefix (Path): The prefix for output file names.
        category_names (Dict[int, str]): A dictionary mapping category IDs to names.

    Returns:
        Tuple[Path, Path]: Paths to the created counts and percentages CSV files.
    """
    counts_file = output_prefix.with_name(f"{output_prefix.stem}_category_counts.csv")
    percentages_file = output_prefix.with_name(
        f"{output_prefix.stem}_category_percentages.csv"
    )

    headers = ["Frame"] + [category_names[i] for i in sorted(category_names.keys())]

    pd.DataFrame(columns=headers).to_csv(counts_file, index=False)
    pd.DataFrame(columns=headers).to_csv(percentages_file, index=False)

    return counts_file, percentages_file


def append_to_csv_files(
    counts_file: Path,
    percentages_file: Path,
    frame_count: int,
    analysis: Dict[int, Tuple[int, float]],
) -> None:
    """
    Append analysis results to the counts and percentages CSV files.

    Args:
        counts_file (Path): Path to the counts CSV file.
        percentages_file (Path): Path to the percentages CSV file.
        frame_count (int): The current frame count.
        analysis (Dict[int, Tuple[int, float]]): Analysis results for the current frame.
    """
    counts_row = [frame_count] + [analysis[i][0] for i in sorted(analysis.keys())]
    percentages_row = [frame_count] + [analysis[i][1] for i in sorted(analysis.keys())]

    pd.DataFrame([counts_row]).to_csv(counts_file, mode="a", header=False, index=False)
    pd.DataFrame([percentages_row]).to_csv(
        percentages_file, mode="a", header=False, index=False
    )


def get_video_files(directory: Path) -> List[Path]:
    """
    Get a list of video files in the specified directory.

    Args:
        directory (Path): The directory to search for video files.

    Returns:
        List[Path]: A list of paths to video files found in the directory.
    """
    video_extensions = [".mp4", ".avi", ".mov"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(directory.glob(f"*{ext}"))
    return video_files


def save_segmentation_map(
    seg_map: np.ndarray, output_prefix: Path, frame_count: Optional[int] = None
) -> None:
    """
    Save a segmentation map as a NumPy array file.

    Args:
        seg_map (np.ndarray): The segmentation map to save.
        output_prefix (Path): The prefix for the output file name.
        frame_count (Optional[int]): The frame count, if applicable.
    """
    filename = f"{output_prefix.stem}_segmap{'_' + str(frame_count) if frame_count is not None else ''}.npy"
    np.save(output_prefix.with_name(filename), seg_map)


def save_colored_segmentation(
    colored_seg: np.ndarray, output_prefix: Path, frame_count: Optional[int] = None
) -> None:
    """
    Save a colored segmentation map as an image file.

    Args:
        colored_seg (np.ndarray): The colored segmentation map to save.
        output_prefix (Path): The prefix for the output file name.
        frame_count (Optional[int]): The frame count, if applicable.
    """
    filename = f"{output_prefix.stem}_colored{'_' + str(frame_count) if frame_count is not None else ''}.png"
    cv2.imwrite(
        str(output_prefix.with_name(filename)),
        cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR),
    )


def save_overlay(
    overlay: np.ndarray, output_prefix: Path, frame_count: Optional[int] = None
) -> None:
    """
    Save an overlay image as a file.

    Args:
        overlay (np.ndarray): The overlay image to save.
        output_prefix (Path): The prefix for the output file name.
        frame_count (Optional[int]): The frame count, if applicable.
    """
    filename = f"{output_prefix.stem}_overlay{'_' + str(frame_count) if frame_count is not None else ''}.png"
    cv2.imwrite(str(output_prefix.with_name(filename)), overlay)


def setup_logging(log_level: str, verbose: bool = False) -> None:
    """
    Set up logging configuration for the application.

    This function configures console and file logging with appropriate
    log levels and formats.

    Args:
        log_level (str): The log level for file logging (e.g., "INFO", "DEBUG").
        verbose (bool): If True, set console logging to DEBUG level.
    """
    logger.remove()  # Remove default handler

    # Determine console log level
    console_level = "DEBUG" if verbose else log_level

    # Console logging
    console_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.add(
        sys.stderr,
        format=console_format,
        level=console_level,
        colorize=True,
    )

    # File logging (always at INFO level or lower, JSON format)
    file_level = min(log_level, "INFO")
    logger.add(
        "segmentation.log",
        format="{message}",
        level=file_level,
        rotation="100 MB",
        retention="1 week",
        serialize=True,
    )

    logger.info(
        f"Logging initialized. Console level: {console_level}, File level: {file_level}"
    )


@contextmanager
def tqdm_context(*args: Any, **kwargs: Any) -> Iterator[tqdm]:
    """
    A context manager for tqdm progress bars.

    This context manager ensures that the tqdm progress bar is properly
    initialized and closed, even if an exception occurs.

    Args:
        *args: Positional arguments to pass to tqdm.
        **kwargs: Keyword arguments to pass to tqdm.

    Yields:
        tqdm: The tqdm progress bar object.
    """
    try:
        progress_bar = tqdm(*args, **kwargs)
        yield progress_bar
    finally:
        progress_bar.close()
