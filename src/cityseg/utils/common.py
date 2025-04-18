"""
This module provides utility functions for the CitySeg package.

It includes functions for handling segmentation data, color palettes,
progress tracking, and logging setup.
"""

from __future__ import annotations

from collections.abc import Iterator
import sys
from contextlib import contextmanager

import numpy as np
import xarray as xr
from loguru import logger
from tqdm.auto import tqdm


# ---- Data Handling Functions ----


def get_segmentation_batch(
    segmentation_data: np.ndarray | xr.DataArray, start: int, end: int
) -> np.ndarray:
    """
    Get a batch of segmentation data from an array or xarray DataArray.

    Args:
        segmentation_data (np.ndarray | xr.DataArray): The segmentation data, either as numpy array or xarray DataArray.
        start (int): Start index of the batch.
        end (int): End index of the batch.

    Returns:
        np.ndarray: A batch of segmentation data.
    """
    if isinstance(segmentation_data, xr.DataArray):
        return segmentation_data.isel(time=slice(start, end)).values
    else:
        return segmentation_data[start:end]


# ---- Color Palette Functions ----

# Cityscapes color palette
CITYSCAPES_PALETTE: list[tuple[int, int, int]] = [
    (128, 64, 128),  # road
    (244, 35, 232),  # sidewalk
    (70, 70, 70),  # building
    (102, 102, 156),  # wall
    (190, 153, 153),  # fence
    (153, 153, 153),  # pole
    (250, 170, 30),  # traffic light
    (220, 220, 0),  # traffic sign
    (107, 142, 35),  # vegetation
    (152, 251, 152),  # terrain
    (70, 130, 180),  # sky
    (220, 20, 60),  # person
    (255, 0, 0),  # rider
    (0, 0, 142),  # car
    (0, 0, 70),  # truck
    (0, 60, 100),  # bus
    (0, 80, 100),  # train
    (0, 0, 230),  # motorcycle
    (119, 11, 32),  # bicycle
    (0, 0, 0),  # void/ignored
]

# First 20 colors of ADE20K palette (full palette imported only when needed)
ADE20K_PALETTE_SUBSET: list[tuple[int, int, int]] = [
    (120, 120, 120),  # wall
    (180, 120, 120),  # building
    (6, 230, 230),  # sky
    (80, 50, 50),
    (4, 200, 3),
    (120, 120, 80),
    (140, 140, 140),
    (204, 5, 255),
    (230, 230, 230),
    (4, 250, 7),
    (224, 5, 255),
    (235, 255, 7),
    (150, 5, 61),
    (120, 120, 70),
    (8, 255, 51),
    (255, 6, 82),
    (143, 255, 140),
    (204, 255, 4),
    (255, 51, 7),
    (204, 70, 3),
]

# First 20 colors of Mapillary Vistas palette (full palette imported only when needed)
MAPILLARY_VISTAS_PALETTE_SUBSET: list[tuple[int, int, int]] = [
    (165, 42, 42),  # Bird
    (0, 192, 0),  # Ground Animal
    (196, 196, 196),  # Curb
    (190, 153, 153),
    (180, 165, 180),
    (102, 102, 156),
    (102, 102, 156),
    (128, 64, 255),
    (140, 140, 200),
    (170, 170, 170),
    (250, 170, 160),
    (96, 96, 96),
    (230, 150, 140),
    (128, 64, 128),
    (110, 110, 110),
    (244, 35, 232),
    (150, 100, 100),
    (70, 70, 70),
    (150, 120, 90),
    (220, 20, 60),
]


def get_palette(name_or_path: str | None = None) -> list[tuple[int, int, int]]:
    """
    Get a color palette by name or model path.

    Args:
        name_or_path (str | None): Name of the palette or model path. If None, returns ADE20K palette.
            Options: "cityscapes", "ade20k", "mapillary", "default", or a model path.

    Returns:
        list[tuple[int, int, int]]: List of RGB tuples representing the color palette.
    """
    # TODO: Consider how this interacts with Palette class in visualization.py
    # Handle direct palette names
    if name_or_path == "cityscapes" or (name_or_path and "cityscapes" in name_or_path):
        return CITYSCAPES_PALETTE

    elif name_or_path == "mapillary" or (name_or_path and "mapillary" in name_or_path):
        from .palettes import MAPILLARY_VISTAS_PALETTE

        return MAPILLARY_VISTAS_PALETTE

    elif (
        name_or_path == "ade20k"
        or name_or_path == "default"
        or (name_or_path and "ade" in name_or_path)
    ):
        from .palettes import ADE20K_PALETTE

        return ADE20K_PALETTE

    # Default to ADE20K if no match
    else:
        from .palettes import ADE20K_PALETTE

        return ADE20K_PALETTE


# ---- Progress Tracking and Logging ----


@contextmanager
def tqdm_context(*args: object, **kwargs: object) -> Iterator[tqdm]:
    """
    A context manager for tqdm progress bars.

    This context manager ensures that the tqdm progress bar is properly
    initialized and closed, even if an exception occurs.

    Args:
        *args (object): Positional arguments to pass to tqdm.
        **kwargs (object): Keyword arguments to pass to tqdm.

    Yields:
        tqdm: The tqdm progress bar object.
    """
    progress_bar = None
    try:
        progress_bar = tqdm(*args, **kwargs)
        yield progress_bar
    finally:
        if progress_bar is not None:
            progress_bar.close()


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
