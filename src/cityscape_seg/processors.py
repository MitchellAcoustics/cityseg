import logging
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Union

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .models import SegmentationModelBase, create_segmentation_model
from .utils import (
    analyze_hdf5_segmaps_with_stats,
    analyze_segmentation_map,
    append_to_csv_files,
    generate_category_stats,
    get_colormap,
    get_video_files,
    initialize_csv_files,
    save_colored_segmentation,
    save_overlay,
    save_segmentation_map,
)

logger = logging.getLogger(__name__)


class ProcessorBase(ABC):
    """
    Abstract base class for image and video processors.

    This class defines the interface and common functionality for processors
    used in the segmentation pipeline.

    Attributes:
        model (SegmentationModelBase): The segmentation model to use for processing.
        config (Dict[str, Any]): Configuration dictionary for the processor.
        hdf5_file: HDF5 file handle for saving segmentation maps.
        counts_file: File path for saving category count data.
        percentages_file: File path for saving category percentage data.
        colormap: Color palette for visualization.
    """

    def __init__(self, model: SegmentationModelBase, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.hdf5_file = None
        self.counts_file = None
        self.percentages_file = None
        self.colormap = get_colormap(
            self.config["visualization"]["colormap"], self.model.num_categories
        )

    @abstractmethod
    def process(self):
        """
        Main processing method to be implemented by subclasses.

        This method should handle the entire processing pipeline for either images or videos.
        """
        pass

    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize an input image based on the maximum size specified in the model configuration.

        Args:
            image (Image.Image): Input PIL Image.

        Returns:
            Image.Image: Resized PIL Image.
        """
        if self.model.max_size is None:
            return image
        width, height = image.size
        scale = min(self.model.max_size / max(width, height), 1)
        new_width, new_height = int(width * scale), int(height * scale)
        return image.resize((new_width, new_height), Image.BILINEAR)

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process a single image through the segmentation pipeline.

        This method handles resizing, tiling (if applicable), and segmentation of the input image.

        Args:
            image (np.ndarray): Input image as a NumPy array.

        Returns:
            np.ndarray: Processed segmentation map.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        original_size = pil_image.size

        if self.model.tile_size:
            seg_map = self.process_tiled(pil_image)
        else:
            resized_image = self.resize_image(pil_image)
            seg_map = self.model.process_tile(resized_image)

        if isinstance(seg_map, torch.Tensor):
            seg_map = seg_map.cpu().numpy()

        if seg_map.shape[:2] != original_size[::-1]:
            seg_map = cv2.resize(
                seg_map, original_size, interpolation=cv2.INTER_NEAREST
            )

        return seg_map

    def process_tiled(self, pil_image: Image.Image) -> np.ndarray:
        """
        Process a large image by tiling it into smaller segments.

        This method is used when the input image is too large to process at once.

        Args:
            pil_image (Image.Image): Large input image as a PIL Image.

        Returns:
            np.ndarray: Processed segmentation map for the entire image.
        """
        width, height = pil_image.size
        seg_map = np.zeros((height, width), dtype=np.int32)
        for y in tqdm(
            range(0, height, self.model.tile_size), desc="Processing tiles", leave=False
        ):
            for x in range(0, width, self.model.tile_size):
                tile = pil_image.crop(
                    (
                        x,
                        y,
                        min(x + self.model.tile_size, width),
                        min(y + self.model.tile_size, height),
                    )
                )
                tile_seg = self.model.process_tile(tile)
                if isinstance(tile_seg, torch.Tensor):
                    tile_seg = tile_seg.cpu().numpy()
                seg_map[y : y + tile_seg.shape[0], x : x + tile_seg.shape[1]] = tile_seg
        return seg_map

    def colorize_segmentation(self, seg_map: np.ndarray) -> np.ndarray:
        """
        Colorize a segmentation map using the model's color palette.

        Args:
            seg_map (np.ndarray): Input segmentation map.

        Returns:
            np.ndarray: Colorized segmentation map.
        """
        if isinstance(seg_map, torch.Tensor):
            seg_map = seg_map.cpu().numpy()
        return np.array(self.model.palette, dtype=np.uint8)[
            seg_map.astype(np.int64) % len(self.model.palette)
        ]

    def save_results(
        self, frame: np.ndarray, seg_map: np.ndarray, frame_count: int = None
    ):
        """
        Save the processing results based on the configuration settings.

        Args:
            frame (np.ndarray): Original input frame.
            seg_map (np.ndarray): Segmentation map.
            frame_count (int, optional): Frame number for video processing.
        """
        output_prefix = Path(self.config["output_prefix"])

        if self.config["save_raw_segmentation"]:
            save_segmentation_map(seg_map, output_prefix, frame_count)

        colored_seg = self.colorize_segmentation(seg_map)

        if self.config["save_colored_segmentation"]:
            save_colored_segmentation(colored_seg, output_prefix, frame_count)

        if self.config["generate_overlay"]:
            alpha = self.config["visualization"]["alpha"]
            # Ensure frame and colored_seg have the same data type
            frame = frame.astype(np.float32)
            colored_seg = colored_seg.astype(np.float32)
            overlay = cv2.addWeighted(frame, 1 - alpha, colored_seg, alpha, 0)
            save_overlay(overlay.astype(np.uint8), output_prefix, frame_count)

    def initialize_hdf5(self, shape):
        """
        Initialize an HDF5 file for saving segmentation maps.

        Args:
            shape: Shape of the dataset to be created in the HDF5 file.
        """
        output_prefix = Path(self.config["output_prefix"])
        hdf5_path = output_prefix.with_name(f"{output_prefix.stem}_segmaps.h5")
        self.hdf5_file = h5py.File(str(hdf5_path), "w")
        self.hdf5_file.create_dataset(
            "segmentation_maps", shape, dtype=np.uint8, chunks=True, compression="gzip"
        )
        self.hdf5_file.attrs["id2label"] = str(self.model.category_names)

    def save_to_hdf5(self, seg_map: np.ndarray, index: int):
        """
        Save a segmentation map to the HDF5 file.

        Args:
            seg_map (np.ndarray): Segmentation map to be saved.
            index (int): Index at which to save the segmentation map.
        """
        self.hdf5_file["segmentation_maps"][index] = seg_map

    def close_hdf5(self):
        """
        Close the HDF5 file if it's open.
        """
        if self.hdf5_file:
            self.hdf5_file.close()


class ImageProcessor(ProcessorBase):
    """
    Processor class for handling single image inputs.

    This class implements the processing pipeline for single image segmentation.
    """

    def process(self):
        """
        Process a single input image through the segmentation pipeline.

        This method handles loading the image, performing segmentation, saving results,
        and generating analysis data.
        """
        input_path = Path(self.config["input"])
        output_prefix = Path(self.config["output_prefix"])

        frame = cv2.imread(str(input_path))
        if frame is None:
            raise IOError(f"Error reading input image: {input_path}")

        seg_map = self.process_image(frame)

        self.initialize_hdf5((1,) + seg_map.shape)
        self.save_to_hdf5(seg_map, 0)
        self.save_results(frame, seg_map)

        analysis = analyze_segmentation_map(seg_map, self.model.num_categories)
        self.counts_file, self.percentages_file = initialize_csv_files(
            output_prefix, self.model.category_names
        )
        append_to_csv_files(self.counts_file, self.percentages_file, 0, analysis)

        self.close_hdf5()
        logger.info(
            f"Image processing complete. Output files saved with prefix: {output_prefix}"
        )


class VideoProcessor(ProcessorBase):
    """
    Processor class for handling video inputs.

    This class implements the processing pipeline for video segmentation.
    """

    def process(self):
        """
        Process an input video through the segmentation pipeline.

        This method handles loading the video, performing frame-by-frame segmentation,
        saving results, and generating analysis data. If an existing HDF5 file is found,
        it will analyze the existing segmentation maps instead of reprocessing the video.
        """
        input_path = Path(self.config["input"])
        output_prefix = Path(self.config["output_prefix"])
        frame_step = self.config["frame_step"]

        hdf5_file_path = output_prefix.with_name(f"{output_prefix.stem}_segmaps.h5")

        if hdf5_file_path.exists():
            logger.info(
                f"Found existing HDF5 file: {hdf5_file_path}. Analyzing existing segmentation maps."
            )
            analyze_hdf5_segmaps_with_stats(hdf5_file_path, output_prefix)
            return

        video = cv2.VideoCapture(str(input_path))
        if not video.isOpened():
            raise IOError(f"Error opening video file: {input_path}")

        try:
            fps = int(video.get(cv2.CAP_PROP_FPS))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            processed_frames = total_frames // frame_step
            output_fps = max(1, fps // frame_step)

            self.initialize_hdf5((processed_frames, height, width))
            self.counts_file, self.percentages_file = initialize_csv_files(
                output_prefix, self.model.category_names
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            overlay_output = cv2.VideoWriter(
                str(output_prefix.with_name(f"{output_prefix.stem}_overlay.mp4")),
                fourcc,
                output_fps,
                (width, height),
            )
            solid_output = cv2.VideoWriter(
                str(output_prefix.with_name(f"{output_prefix.stem}_solid.mp4")),
                fourcc,
                output_fps,
                (width, height),
            )

            with tqdm(total=processed_frames, unit="frames") as pbar:
                for frame_idx, frame_count in enumerate(
                    range(0, total_frames, frame_step)
                ):
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    ret, frame = video.read()
                    if not ret:
                        logger.warning(
                            f"Failed to read frame at position {frame_count}. Stopping processing."
                        )
                        break

                    seg_map = self.process_image(frame)
                    colored_seg = self.colorize_segmentation(seg_map)
                    overlay = cv2.addWeighted(frame, 0.5, colored_seg, 0.5, 0)

                    self.save_to_hdf5(seg_map, frame_idx)
                    analysis = analyze_segmentation_map(
                        seg_map, self.model.num_categories
                    )
                    append_to_csv_files(
                        self.counts_file, self.percentages_file, frame_count, analysis
                    )

                    overlay_output.write(overlay.astype(np.uint8))
                    solid_output.write(colored_seg)

                    pbar.update(1)

        except Exception as e:
            logger.error(f"An error occurred during video processing: {str(e)}")
            raise
        finally:
            video.release()
            overlay_output.release()
            solid_output.release()
            cv2.destroyAllWindows()
            self.close_hdf5()

        self.generate_and_save_stats(output_prefix)

        logger.info(
            f"Video processing complete. Output files saved with prefix: {output_prefix}"
        )
        logger.info(
            f"Processed {processed_frames} frames out of {total_frames} total frames."
        )

    def generate_and_save_stats(self, output_prefix: Path):
        """
        Generate and save category statistics for the processed video.

        This method calculates overall statistics for category counts and percentages
        and saves them to CSV files.

        Args:
            output_prefix (Path): Prefix for output statistic files.
        """
        counts_stats = generate_category_stats(self.counts_file)
        counts_stats.to_csv(
            output_prefix.with_name(f"{output_prefix.stem}_counts_stats.csv"),
            index=False,
        )

        percentages_stats = generate_category_stats(self.percentages_file)
        percentages_stats.to_csv(
            output_prefix.with_name(f"{output_prefix.stem}_percentages_stats.csv"),
            index=False,
        )

        logger.info(
            f"Category statistics generated and saved with prefix: {output_prefix.stem}"
        )


class DirectoryProcessor:
    """
    Processor class for handling directory inputs containing multiple video files.

    This class manages the processing of multiple videos in a directory.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DirectoryProcessor.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the processor.
        """
        self.config = config
        self.model = create_segmentation_model(config["model"])

    def process(self):
        """
        Process all video files in the input directory.

        This method discovers video files in the input directory and processes
        each video sequentially using the VideoProcessor.
        """
        input_dir = Path(self.config["input"])
        video_files = get_video_files(input_dir)

        if not video_files:
            logger.warning(f"No video files found in directory: {input_dir}")
            return

        for video_file in tqdm(video_files, desc="Processing videos"):
            video_config = self.config.copy()
            video_config["input"] = str(video_file)
            video_config["output_prefix"] = str(
                Path(self.config["output_prefix"]) / video_file.stem.split("_")[0]
            )
            self._process_single_video(video_config)

        logger.info(f"Finished processing all videos in {input_dir}")

    def _process_single_video(self, video_config: Dict[str, Any]):
        """
        Process a single video file.

        Args:
            video_config (Dict[str, Any]): Configuration for processing a single video.
        """
        try:
            processor = VideoProcessor(self.model, video_config)
            processor.process()
        except Exception as e:
            logger.error(f"Error processing video {video_config['input']}: {str(e)}")


def create_processor(
    config: Dict[str, Any],
) -> Union[ProcessorBase, DirectoryProcessor]:
    """
    Factory function to create the appropriate processor based on the input file type.

    Args:
        config (Dict[str, Any]): Configuration dictionary for the processor.

    Returns:
        Union[ProcessorBase, DirectoryProcessor]: An instance of either ImageProcessor,
        VideoProcessor, or DirectoryProcessor.

    Raises:
        ValueError: If an unsupported input type is provided.
    """
    input_path = Path(config["input"])
    if input_path.is_dir():
        return DirectoryProcessor(config)
    elif input_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
        model = create_segmentation_model(config["model"])
        return VideoProcessor(model, config)
    elif input_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        model = create_segmentation_model(config["model"])
        return ImageProcessor(model, config)
    else:
        raise ValueError(f"Unsupported input type: {input_path}")
