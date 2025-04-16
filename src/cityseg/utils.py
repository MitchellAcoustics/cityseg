"""
This module provides utility functions for the semantic segmentation pipeline.

It includes functions for analyzing segmentation maps, generating statistics,
handling file operations, and setting up logging. These utilities are used
throughout the segmentation pipeline to support various processing tasks.
"""

import sys
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import xarray as xr
from loguru import logger
from tqdm.auto import tqdm


def get_segmentation_batch(
    segmentation_data: np.ndarray | xr.DataArray, start: int, end: int
) -> np.ndarray:
    """
    Get a batch of segmentation data from an array or xarray DataArray.

    Args:
        segmentation_data: The segmentation data, either as numpy array or xarray DataArray.
        start (int): Start index of the batch.
        end (int): End index of the batch.

    Returns:
        np.ndarray: A batch of segmentation data.
    """
    if isinstance(segmentation_data, xr.DataArray):
        return segmentation_data.isel(time=slice(start, end)).values
    else:
        return segmentation_data[start:end]


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
