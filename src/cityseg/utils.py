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
import h5py
import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm


def get_segmentation_data_batch(
    segmentation_data: h5py.Dataset, start: int, end: int
) -> np.ndarray:
    """
    Get a batch of segmentation data from the HDF5 file.

    Args:
        segmentation_data:
        start (int): Start index of the batch.
        end (int): End index of the batch.

    Returns:
        np.ndarray: A batch of segmentation data.
    """
    return segmentation_data[start:end]


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
