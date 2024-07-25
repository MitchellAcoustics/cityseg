import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .palettes import ADE20K_PALETTE, CITYSCAPES_PALETTE

logger = logging.getLogger(__name__)


def get_device(device: str) -> torch.device:
    """
    Determine the appropriate device (CUDA, MPS, or CPU) for running computations.

    Returns:
        torch.device: The selected device.
    """
    if device is not None:
        if device.lower() == "cuda":
            return torch.device("cuda")
        elif device.lower() == "mps":
            return torch.device("mps")
        elif device.lower() == "cpu":
            return torch.device("cpu")
        else:
            raise ValueError(f"Unknown device: {device}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_colormap(colormap_name: str, num_categories: int) -> np.ndarray:
    if colormap_name == "default":
        return create_default_colormap(num_categories)
    elif colormap_name == "cityscapes":
        return CITYSCAPES_PALETTE
    elif colormap_name == "ade20k":
        return ADE20K_PALETTE
    else:
        raise ValueError(f"Unknown colormap: {colormap_name}")


def create_default_colormap(num_categories: int) -> np.ndarray:
    colormap = np.random.randint(0, 256, size=(num_categories, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]  # Set background to black
    return colormap


def save_segmentation_map(
    seg_map: np.ndarray, output_prefix: Path, frame_count: int = None
):
    """
    Save a segmentation map as a NumPy array file.

    Args:
        seg_map (np.ndarray): The segmentation map to save.
        output_prefix (Path): Prefix for the output file path.
        frame_count (int, optional): Frame number for video processing. Defaults to None.
    """

    filename = f"{output_prefix.stem}_segmap{'_' + str(frame_count) if frame_count is not None else ''}.npy"
    np.save(output_prefix.with_name(filename), seg_map)


def save_colored_segmentation(
    colored_seg: np.ndarray, output_prefix: Path, frame_count: int = None
):
    """
    Save a colored segmentation map as an image file.

    Args:
        colored_seg (np.ndarray): The colored segmentation map to save.
        output_prefix (Path): Prefix for the output file path.
        frame_count (int, optional): Frame number for video processing. Defaults to None.
    """

    filename = f"{output_prefix.stem}_colored{'_' + str(frame_count) if frame_count is not None else ''}.png"
    cv2.imwrite(
        str(output_prefix.with_name(filename)),
        cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR),
    )


def save_overlay(
    frame: np.ndarray,
    colored_seg: np.ndarray,
    output_prefix: Path,
    frame_count: int = None,
):
    """
    Save an overlay of the original frame and the colored segmentation map.

    Args:
        frame (np.ndarray): The original image frame.
        colored_seg (np.ndarray): The colored segmentation map.
        output_prefix (Path): Prefix for the output file path.
        frame_count (int, optional): Frame number for video processing. Defaults to None.
    """

    overlay = cv2.addWeighted(frame, 0.5, colored_seg, 0.5, 0)
    filename = f"{output_prefix.stem}_overlay{'_' + str(frame_count) if frame_count is not None else ''}.png"
    cv2.imwrite(str(output_prefix.with_name(filename)), overlay)


def analyze_segmentation_map(
    seg_map: np.ndarray, num_categories: int
) -> Dict[int, Tuple[int, float]]:
    """
    Analyze a segmentation map to count pixels per category and calculate percentages.

    Args:
        seg_map (np.ndarray): The segmentation map to analyze.
        num_categories (int): The total number of categories.

    Returns:
        Dict[int, Tuple[int, float]]: A dictionary with category IDs as keys and tuples (pixel_count, percentage) as values.
    """

    if isinstance(seg_map, torch.Tensor):
        seg_map = seg_map.cpu().numpy()

    unique, counts = np.unique(seg_map, return_counts=True)
    total_pixels = seg_map.size
    category_analysis = {i: (0, 0.0) for i in range(num_categories)}

    for category_id, pixel_count in zip(unique, counts):
        percentage = (pixel_count / total_pixels) * 100
        category_analysis[int(category_id)] = (int(pixel_count), float(percentage))

    return category_analysis


def analyze_hdf5_segmaps(hdf5_file_path: Path, output_prefix: Path) -> None:
    """
    Analyze segmentation maps stored in an HDF5 file and export results to CSV files.

    Args:
        hdf5_file_path (Path): Path to the HDF5 file containing segmentation maps.
        output_prefix (Path): Prefix for output CSV files.

    Returns:
        None
    """
    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        # Retrieve metadata
        id2label = eval(hdf5_file.attrs["id2label"])
        num_categories = len(id2label)

        # Prepare CSV files
        counts_file = output_prefix.with_name(
            f"{output_prefix.stem}_analysis_counts.csv"
        )
        percentages_file = output_prefix.with_name(
            f"{output_prefix.stem}_analysis_percentages.csv"
        )

        headers = ["Frame"] + [id2label[i] for i in sorted(id2label.keys())]

        pd.DataFrame(columns=headers).to_csv(counts_file, index=False)
        pd.DataFrame(columns=headers).to_csv(percentages_file, index=False)

        seg_maps_dataset = hdf5_file["segmentation_maps"]
        total_frames = seg_maps_dataset.shape[0]

        for frame_idx in tqdm(range(total_frames), desc="Analyzing frames"):
            seg_map = seg_maps_dataset[frame_idx]

            # Perform category analysis
            analysis = analyze_segmentation_map(seg_map, num_categories)

            # Prepare rows for CSV files
            counts_row = [frame_idx] + [analysis[i][0] for i in sorted(analysis.keys())]
            percentages_row = [frame_idx] + [
                analysis[i][1] for i in sorted(analysis.keys())
            ]

            # Append to CSV files
            pd.DataFrame([counts_row]).to_csv(
                counts_file, mode="a", header=False, index=False
            )
            pd.DataFrame([percentages_row]).to_csv(
                percentages_file, mode="a", header=False, index=False
            )

    logger.info(
        f"Analysis complete. Output files: {counts_file} and {percentages_file}"
    )


def generate_category_stats(csv_file_path: Path) -> pd.DataFrame:
    """
    Generate statistics for each category from a CSV file of frame-by-frame data.

    Args:
        csv_file_path (Path): Path to the CSV file containing frame-by-frame category data.

    Returns:
        pd.DataFrame: DataFrame containing statistics (mean, median, std dev, min, max) for each category.
    """
    df = pd.read_csv(csv_file_path)

    # Exclude the 'Frame' column
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


def analyze_hdf5_segmaps_with_stats(hdf5_file_path: Path, output_prefix: Path) -> None:
    """
    Analyze segmentation maps stored in an HDF5 file, export results to CSV files,
    and generate category statistics.

    This function performs a comprehensive analysis of the segmentation results,
    including frame-by-frame analysis and overall statistics for both pixel counts
    and percentages.

    Args:
        hdf5_file_path (Path): Path to the HDF5 file containing segmentation maps.
        output_prefix (Path): Prefix for output CSV files.

    Returns:
        None

    Output:
        - Frame-by-frame analysis CSV files (counts and percentages)
        - Overall statistics CSV files (counts and percentages)
    """
    # Perform the basic analysis
    analyze_hdf5_segmaps(hdf5_file_path, output_prefix)

    # Generate and save statistics for counts
    counts_file = output_prefix.with_name(f"{output_prefix.stem}_analysis_counts.csv")
    counts_stats = generate_category_stats(counts_file)
    counts_stats.to_csv(
        output_prefix.with_name(f"{output_prefix.stem}_counts_stats.csv"), index=False
    )

    # Generate and save statistics for percentages
    percentages_file = output_prefix.with_name(
        f"{output_prefix.stem}_analysis_percentages.csv"
    )
    percentages_stats = generate_category_stats(percentages_file)
    percentages_stats.to_csv(
        output_prefix.with_name(f"{output_prefix.stem}_percentages_stats.csv"),
        index=False,
    )

    logger.info(
        f"Statistics generated. Output files: {output_prefix.stem}_counts_stats.csv and {output_prefix.stem}_percentages_stats.csv"
    )


def initialize_csv_files(output_prefix: Path, category_names: Dict[int, str]):
    """
    Initialize CSV files for storing category counts and percentages.

    Args:
        output_prefix (Path): Prefix for the output file paths.
        category_names (Dict[int, str]): Mapping of category IDs to names.

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
):
    """
    Append analysis results to the counts and percentages CSV files.

    Args:
        counts_file (Path): Path to the counts CSV file.
        percentages_file (Path): Path to the percentages CSV file.
        frame_count (int): The current frame number.
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
    Discover video files in the given directory.

    Args:
        directory (Path): Path to the directory to search for video files.

    Returns:
        List[Path]: List of paths to discovered video files.
    """
    video_extensions = [".mp4", ".avi", ".mov"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(directory.glob(f"*{ext}"))
    return video_files
