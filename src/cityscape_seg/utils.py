import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from tqdm.asyncio import tqdm


def analyze_segmentation_map(
    seg_map: np.ndarray, num_categories: int
) -> Dict[int, Tuple[int, float]]:
    unique, counts = np.unique(seg_map, return_counts=True)
    total_pixels = seg_map.size
    category_analysis = {i: (0, 0.0) for i in range(num_categories)}

    for category_id, pixel_count in zip(unique, counts):
        percentage = (pixel_count / total_pixels) * 100
        category_analysis[int(category_id)] = (int(pixel_count), float(percentage))

    return category_analysis


def generate_category_stats(csv_file_path: Path) -> pd.DataFrame:
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


def initialize_csv_files(output_prefix: Path, category_names: Dict[int, str]):
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
):
    counts_row = [frame_count] + [analysis[i][0] for i in sorted(analysis.keys())]
    percentages_row = [frame_count] + [analysis[i][1] for i in sorted(analysis.keys())]

    pd.DataFrame([counts_row]).to_csv(counts_file, mode="a", header=False, index=False)
    pd.DataFrame([percentages_row]).to_csv(
        percentages_file, mode="a", header=False, index=False
    )


def get_video_files(directory: Path) -> List[Path]:
    video_extensions = [".mp4", ".avi", ".mov"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(directory.glob(f"*{ext}"))
    return video_files


def save_segmentation_map(
    seg_map: np.ndarray, output_prefix: Path, frame_count: int = None
):
    filename = f"{output_prefix.stem}_segmap{'_' + str(frame_count) if frame_count is not None else ''}.npy"
    np.save(output_prefix.with_name(filename), seg_map)


def save_colored_segmentation(
    colored_seg: np.ndarray, output_prefix: Path, frame_count: int = None
):
    filename = f"{output_prefix.stem}_colored{'_' + str(frame_count) if frame_count is not None else ''}.png"
    cv2.imwrite(
        str(output_prefix.with_name(filename)),
        cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR),
    )


def save_overlay(overlay: np.ndarray, output_prefix: Path, frame_count: int = None):
    filename = f"{output_prefix.stem}_overlay{'_' + str(frame_count) if frame_count is not None else ''}.png"
    cv2.imwrite(str(output_prefix.with_name(filename)), overlay)


def setup_logging(log_level, verbose=False):
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
def tqdm_context(*args, **kwargs):
    try:
        progress_bar = tqdm(*args, **kwargs)
        yield progress_bar
    finally:
        progress_bar.close()
