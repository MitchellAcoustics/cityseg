import logging
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .config import Config, InputType
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

    def __init__(self, model: SegmentationModelBase, config: Config):
        self.model = model
        self.config = config
        self.hdf5_file = None
        self.counts_file = None
        self.percentages_file = None
        self.colormap = get_colormap(
            self.config.visualization.colormap, self.model.num_categories
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
        Colorize a segmentation map using the model's color palette or the specified colormap.

        Args:
            seg_map (np.ndarray): Input segmentation map.

        Returns:
            np.ndarray: Colorized segmentation map.
        """
        if isinstance(seg_map, torch.Tensor):
            seg_map = seg_map.cpu().numpy()

        if self.config.visualization.colormap == "default":
            return np.array(self.model.palette, dtype=np.uint8)[
                seg_map.astype(np.int64) % len(self.model.palette)
                ]
        else:
            return self.colormap[seg_map.astype(np.int64) % len(self.colormap)]

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
        output_path = self.config.get_output_path()

        if self.config.save_raw_segmentation:
            save_segmentation_map(seg_map, output_path, frame_count)

        if self.config.save_colored_segmentation or self.config.save_overlay:
            colored_seg = self.colorize_segmentation(seg_map)

            if self.config.save_colored_segmentation:
                save_colored_segmentation(colored_seg, output_path, frame_count)

            if self.config.save_overlay:
                alpha = self.config.visualization.alpha
                overlay = cv2.addWeighted(frame, 1 - alpha, colored_seg.astype(np.float32), alpha, 0)
                save_overlay(overlay.astype(np.uint8), output_path, frame_count)

    def initialize_hdf5(self, shape):
        """
        Initialize an HDF5 file for saving segmentation maps.

        Args:
            shape: Shape of the dataset to be created in the HDF5 file.
        """
        output_path = self.config.get_output_path()
        hdf5_path = output_path.with_name(f"{output_path.stem}_segmaps.h5")
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
        input_path = self.config.input
        output_path = self.config.get_output_path()

        frame = cv2.imread(str(input_path))
        if frame is None:
            raise IOError(f"Error reading input image: {input_path}")

        seg_map = self.process_image(frame)

        self.initialize_hdf5((1,) + seg_map.shape)
        self.save_to_hdf5(seg_map, 0)
        self.save_results(frame, seg_map)

        analysis = analyze_segmentation_map(seg_map, self.model.num_categories)
        self.counts_file, self.percentages_file = initialize_csv_files(
            output_path, self.model.category_names
        )
        append_to_csv_files(self.counts_file, self.percentages_file, 0, analysis)

        self.close_hdf5()
        logger.info(
            f"Image processing complete. Output files saved with prefix: {output_path}"
        )


class VideoProcessor(ProcessorBase):
    """
    Processor class for handling video inputs.

    This class implements the processing pipeline for video segmentation.
    """

    def process(self):
        logger = logging.getLogger(__name__)

        input_path = self.config.input
        output_path = self.config.get_output_path()
        frame_step = self.config.frame_step

        logger.info(f"Starting video processing for {input_path}")
        logger.debug(
            f"Configuration: save_colored_segmentation={self.config.save_colored_segmentation}, save_overlay={self.config.save_overlay}"
            )

        perform_colorization = self.config.save_colored_segmentation or self.config.save_overlay
        logger.info(f"Colorization will be {'performed' if perform_colorization else 'skipped'}")

        try:
            video = cv2.VideoCapture(str(input_path))
            if not video.isOpened():
                raise IOError(f"Error opening video file: {input_path}")

            logger.debug("Video file opened successfully")

            fps = int(video.get(cv2.CAP_PROP_FPS))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(f"Video properties: FPS={fps}, Width={width}, Height={height}, Total Frames={total_frames}")

            processed_frames = total_frames // frame_step
            output_fps = max(1, fps // frame_step)

            logger.info(f"Will process {processed_frames} frames at {output_fps} FPS")

            if self.config.save_raw_segmentation:
                logger.debug("Initializing HDF5 file for raw segmentation maps")
                self.initialize_hdf5((processed_frames, height, width))

            logger.debug("Initializing CSV files for analysis")
            self.counts_file, self.percentages_file = initialize_csv_files(
                    output_path, self.model.category_names
                    )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            colored_output = None
            overlay_output = None

            if perform_colorization:
                if self.config.save_colored_segmentation:
                    logger.debug("Creating colored segmentation video writer")
                    colored_output = cv2.VideoWriter(
                            str(output_path.with_name(f"{output_path.stem}_colored.mp4")),
                            fourcc,
                            output_fps,
                            (width, height),
                            )

                if self.config.save_overlay:
                    logger.debug("Creating overlay video writer")
                    overlay_output = cv2.VideoWriter(
                            str(output_path.with_name(f"{output_path.stem}_overlay.mp4")),
                            fourcc,
                            output_fps,
                            (width, height),
                            )

            with tqdm(total=processed_frames, unit="frames", leave=False) as pbar:
                for frame_idx, frame_count in enumerate(range(0, total_frames, frame_step)):
                    logger.debug(f"Processing frame {frame_count}")
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    ret, frame = video.read()
                    if not ret:
                        logger.warning(f"Failed to read frame at position {frame_count}. Stopping processing.")
                        break

                    logger.debug("Performing segmentation")
                    seg_map = self.process_image(frame)

                    if self.config.save_raw_segmentation:
                        logger.debug("Saving raw segmentation map")
                        self.save_to_hdf5(seg_map, frame_idx)

                    logger.debug("Analyzing segmentation map")
                    analysis = analyze_segmentation_map(seg_map, self.model.num_categories)
                    append_to_csv_files(self.counts_file, self.percentages_file, frame_count, analysis)

                    if perform_colorization:
                        logger.debug("Colorizing segmentation map")
                        colored_seg = self.colorize_segmentation(seg_map)

                        if self.config.save_colored_segmentation:
                            logger.debug("Writing colored segmentation frame")
                            colored_output.write(colored_seg.astype(np.uint8))

                        if self.config.save_overlay:
                            logger.debug("Creating and writing overlay frame")
                            alpha = self.config.visualization.alpha
                            overlay = cv2.addWeighted(
                                    frame.astype(np.float64),
                                    1 - alpha,
                                    colored_seg.astype(np.float64),
                                    alpha,
                                    0
                                    ).astype(np.uint8)
                            overlay_output.write(overlay)

                    pbar.update(1)

        except Exception as e:
            logger.exception(f"An error occurred during video processing: {str(e)}")
            raise

        finally:
            logger.debug("Cleaning up resources")
            video.release()
            if colored_output:
                colored_output.release()
            if overlay_output:
                overlay_output.release()
            cv2.destroyAllWindows()
            self.close_hdf5()

        self.generate_and_save_stats(output_path)

        logger.info(f"Video processing complete. Output files saved with prefix: {output_path}")
        logger.info(f"Processed {processed_frames} frames out of {total_frames} total frames.")

    def generate_and_save_stats(self, output_path: Path):
        """
        Generate and save category statistics for the processed video.

        This method calculates overall statistics for category counts and percentages
        and saves them to CSV files.

        Args:
            output_prefix (Path): Prefix for output statistic files.
        """
        counts_stats = generate_category_stats(self.counts_file)
        counts_stats.to_csv(
            output_path.with_name(f"{output_path.stem}_counts_stats.csv"),
            index=False,
        )

        percentages_stats = generate_category_stats(self.percentages_file)
        percentages_stats.to_csv(
            output_path.with_name(f"{output_path.stem}_percentages_stats.csv"),
            index=False,
        )

        logger.info(
            f"Category statistics generated and saved with prefix: {output_path.stem}"
        )


class DirectoryProcessor:
    """
    Processor class for handling directory inputs containing multiple video files.

    This class manages the processing of multiple videos in a directory.
    """

    def __init__(self, config: Config):
        """
        Initialize the DirectoryProcessor.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the processor.
        """
        self.config = config
        self.model = create_segmentation_model(config.model)

    def process(self):
        """
        Process all video files in the input directory.

        This method discovers video files in the input directory and processes
        each video sequentially using the VideoProcessor.
        """
        input_dir = self.config.input
        video_files = get_video_files(input_dir)

        if not video_files:
            logger.warning(f"No video files found in directory: {input_dir}")
            return

        # Create a single output directory for all videos
        output_dir = self.config.get_output_path()

        for video_file in tqdm(video_files, desc="Processing videos"):
            video_config = Config(
                input=video_file,
                output_dir=output_dir,  # Use the same output directory for all videos
                output_prefix=None,
                model=self.config.model,
                frame_step=self.config.frame_step,
                save_raw_segmentation=self.config.save_raw_segmentation,
                save_colored_segmentation=self.config.save_colored_segmentation,
                save_overlay=self.config.save_overlay,
                visualization=self.config.visualization,
            )
            self._process_single_video(video_config)

        logger.info(f"Finished processing all videos in {input_dir}")

    def _process_single_video(self, video_config: Config):
        """
        Process a single video file.

        Args:
            video_config (Dict[str, Any]): Configuration for processing a single video.
        """
        try:
            processor = VideoProcessor(self.model, video_config)
            processor.process()
        except Exception as e:
            logger.error(f"Error processing video {video_config.input}: {str(e)}")


def create_processor(config: Config) -> Union[ProcessorBase, DirectoryProcessor]:
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
    if config.input_type == InputType.DIRECTORY:
        return DirectoryProcessor(config)
    else:
        model = create_segmentation_model(config.model)
        if config.input_type == InputType.SINGLE_VIDEO:
            return VideoProcessor(model, config)
        else:  # SINGLE_IMAGE
            return ImageProcessor(model, config)
