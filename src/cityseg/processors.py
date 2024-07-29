"""
This module provides classes and functions for processing images and videos
using semantic segmentation models.

It includes processors for handling individual files (images or videos) and
directories containing multiple video files. The module also manages caching
of segmentation results, generation of output visualizations, and analysis
of segmentation statistics.
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import cv2
import h5py
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image

from .config import Config, ConfigHasher, InputType
from .exceptions import InputError, ProcessingError
from .file_handler import FileHandler
from .pipeline import create_segmentation_pipeline
from .processing_plan import ProcessingPlan
from .segmentation_analyzer import SegmentationAnalyzer
from .utils import get_segmentation_data_batch, get_video_files, tqdm_context
from .video_file_iterator import VideoFileIterator
from .visualization_handler import VisualizationHandler


class ImageProcessor:
    """
    Processes individual images using semantic segmentation models.

    This class handles the segmentation of single images, including saving results
    and analyzing the segmentation output.

    Attributes:
        config (Config): Configuration object containing processing parameters.
        pipeline: Segmentation pipeline for processing images.
        file_handler (FileHandler): Handles file operations.
        visualizer (VisualizationHandler): Handles visualization of segmentation results.
        analyzer (SegmentationAnalyzer): Analyzes segmentation results.
    """

    def __init__(self, config: Config):
        """
        Initializes the ImageProcessor with the given configuration.

        Args:
            config (Config): Configuration object for the processor.
        """
        self.config = config
        self.pipeline = create_segmentation_pipeline(config.model)
        self.file_handler = FileHandler()
        self.visualizer = VisualizationHandler()
        self.analyzer = SegmentationAnalyzer()

    def process(self) -> None:
        """
        Processes the input image according to the configuration.

        This method handles the entire image processing pipeline, including
        segmentation, result saving, and analysis.

        Raises:
            ProcessingError: If an error occurs during image processing.
        """
        logger.info(f"Processing image: {self.config.input}")
        try:
            image = Image.open(self.config.input).convert("RGB")
            if self.config.model.max_size:
                image.thumbnail(
                    (self.config.model.max_size, self.config.model.max_size)
                )

            result = self.pipeline([image])[0]

            self._save_results(image, result)
            self._analyze_results(result["seg_map"])

            logger.info("Image processing complete")
        except Exception as e:
            logger.exception(f"Error during image processing: {str(e)}")
            raise ProcessingError(f"Error during image processing: {str(e)}")

    def _save_results(self, image: Image.Image, result: Dict[str, Any]) -> None:
        """
        Saves the segmentation results based on the configuration.

        Args:
            image (Image.Image): The original input image.
            result (Dict[str, Any]): The segmentation result dictionary.
        """
        output_path = self.config.get_output_path()

        # Save raw segmentation
        if self.config.save_raw_segmentation:
            raw_seg_path = output_path.with_name(
                f"{output_path.stem}_raw_segmentation.png"
            )
            Image.fromarray(result["seg_map"].astype(np.uint8)).save(raw_seg_path)
            logger.info(f"Raw segmentation saved to {raw_seg_path}")

        # Save colored segmentation
        if self.config.save_colored_segmentation:
            colored_seg_path = output_path.with_name(
                f"{output_path.stem}_colored_segmentation.png"
            )
            colored_seg = self.visualizer.visualize_segmentation(
                np.array(image), result["seg_map"], result["palette"], colored_only=True
            )
            Image.fromarray(colored_seg).save(colored_seg_path)
            logger.info(f"Colored segmentation saved to {colored_seg_path}")

        # Save overlay
        if self.config.save_overlay:
            overlay_path = output_path.with_name(f"{output_path.stem}_overlay.png")
            overlay = self.visualizer.visualize_segmentation(
                np.array(image),
                result["seg_map"],
                result["palette"],
                colored_only=False,
            )
            Image.fromarray(overlay).save(overlay_path)
            logger.info(f"Overlay saved to {overlay_path}")

    def _analyze_results(self, seg_map: np.ndarray) -> None:
        """
        Analyzes the segmentation results and saves the analysis.

        Args:
            seg_map (np.ndarray): The segmentation map to analyze.
        """
        output_path = self.config.get_output_path()
        num_categories = self.config.model.num_classes

        analysis = self.analyzer.analyze_segmentation_map(seg_map, num_categories)

        counts_file = output_path.with_name(f"{output_path.stem}_category_counts.csv")
        percentages_file = output_path.with_name(
            f"{output_path.stem}_category_percentages.csv"
        )

        with open(counts_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["category_id", "pixel_count"])
            for category_id, (pixel_count, _) in analysis.items():
                writer.writerow([category_id, pixel_count])

        with open(percentages_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["category_id", "percentage"])
            for category_id, (_, percentage) in analysis.items():
                writer.writerow([category_id, percentage])

        logger.info(f"Category counts saved to {counts_file}")
        logger.info(f"Category percentages saved to {percentages_file}")


class VideoProcessor:
    """
    Processes video files using semantic segmentation models.

    This class handles the segmentation of video frames, including saving results,
    generating output videos, and analyzing the segmentation output.

    Attributes:
        config (Config): Configuration object containing processing parameters.
        pipeline: Segmentation pipeline for processing video frames.
        processing_plan (ProcessingPlan): Plan for video processing steps.
        file_handler (FileHandler): Handles file operations.
        visualizer (VisualizationHandler): Handles visualization of segmentation results.
        analyzer (SegmentationAnalyzer): Analyzes segmentation results.
    """

    def __init__(self, config: Config):
        """
        Initializes the VideoProcessor with the given configuration.

        Args:
            config (Config): Configuration object for the processor.
        """
        self.config = config
        self.pipeline = create_segmentation_pipeline(config.model)
        self.processing_plan = ProcessingPlan(config)
        self.file_handler = FileHandler()
        self.visualizer = VisualizationHandler()
        self.analyzer = SegmentationAnalyzer()
        logger.debug(f"VideoProcessor initialized with config: {config}")

    def process(self) -> None:
        """
        Processes the input video according to the configuration and processing plan.

        This method handles the entire video processing pipeline, including
        frame segmentation, result saving, video generation, and analysis.

        Raises:
            ProcessingError: If an error occurs during video processing.
        """
        logger.info(f"Processing video: {self.config.input.name}")
        try:
            output_path = self.config.get_output_path()
            hdf_path = output_path.with_name(f"{output_path.stem}_segmentation.h5")

            if self.processing_plan.plan["process_video"]:
                # logger.debug("Executing video frame processing")
                segmentation_data, metadata = self._process_video_frames()
                if self.processing_plan.plan["generate_hdf"]:
                    logger.debug(f"Saving segmentation data to HDF file: {hdf_path}")
                    self.file_handler.save_hdf_file(
                        hdf_path, segmentation_data, metadata
                    )
            else:
                logger.info(
                    f"Loading existing segmentation data from HDF file: {hdf_path.name}"
                )
                hdf_file, metadata = self.file_handler.load_hdf_file(hdf_path)
                segmentation_data = hdf_file[
                    "segmentation"
                ]  # This is now a h5py.Dataset

            # Generate videos based on the processing plan
            if (
                self.processing_plan.plan["generate_colored_video"]
                or self.processing_plan.plan["generate_overlay_video"]
            ):
                self.generate_videos(segmentation_data, metadata)

            if self.processing_plan.plan["analyze_results"]:
                logger.debug("Analyzing segmentation results")
                SegmentationAnalyzer.analyze_results(
                    segmentation_data, metadata, output_path
                )

            self._update_processing_history()

            logger.info("Video processing complete")
        except Exception as e:
            logger.exception(f"Error during video processing: {str(e)}")
            raise ProcessingError(f"Error during video processing: {str(e)}")
        finally:
            if hasattr(self, "hdf_file"):
                self.hdf_file.close()

    def _process_video_frames(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes video frames in batches and returns segmentation data and metadata.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Segmentation data and metadata.
        """
        cap = cv2.VideoCapture(str(self.config.input))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        segmentation_data = []

        logger.debug(f"Processing video frames in batches of {self.config.batch_size}")
        with tqdm_context(
            total=total_frames // self.config.frame_step,
            desc="Processing frames",
            disable=self.config.disable_tqdm,
        ) as pbar:
            for batch in self._frame_generator(
                cv2.VideoCapture(str(self.config.input))
            ):
                logger.debug("Loading batch into pipeline...")
                batch_results = self.pipeline(batch)
                logger.debug("Adding batch results to segmentation data...")
                segmentation_data.extend(
                    [result["seg_map"] for result in batch_results]
                )
                pbar.update(len(batch))
        cap.release()

        metadata = {
            "model_name": self.config.model.name,
            "original_video": str(self.config.input.name),
            "palette": np.array(self.pipeline.palette.tolist(), np.uint8)
            if self.pipeline.palette is not None
            else None,
            "label_ids": self.pipeline.model.config.id2label,
            "frame_count": len(segmentation_data),
            "frame_step": self.config.frame_step,
            "total_video_frames": total_frames,
            "fps": fps,
        }

        return np.array(segmentation_data), metadata

    def _frame_generator(self, cap: cv2.VideoCapture) -> Iterator[List[Image.Image]]:
        """
        Generates batches of frames from a video capture object.

        Args:
            cap (cv2.VideoCapture): The video capture object.

        Yields:
            Iterator[List[Image.Image]]: Batches of frames as PIL Image objects.
        """
        while True:
            frames = []
            for _ in range(self.config.batch_size):
                for _ in range(self.config.frame_step):
                    ret, frame = cap.read()
                    if not ret:
                        break
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                frames.append(pil_image)
            if not frames:
                break
            yield frames

    def generate_videos(
        self, segmentation_data: h5py.Dataset, metadata: Dict[str, Any]
    ) -> None:
        """
        Generates output videos based on the processing plan, using batched processing.

        Args:
            segmentation_data (h5py.Dataset): The segmentation data for all frames.
            metadata (Dict[str, Any]): Metadata about the video and segmentation.
        """
        if not (
            self.processing_plan.plan.get("generate_colored_video", False)
            or self.processing_plan.plan.get("generate_overlay_video", False)
        ):
            logger.info("No video generation required according to the processing plan")
            return

        start_time = time.time()
        cap = cv2.VideoCapture(str(self.config.input))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = metadata["fps"] / metadata["frame_step"]

        output_base = self.config.get_output_path()
        video_writers = self._initialize_video_writers(width, height, fps)

        chunk_size = 100  # Adjust this value based on available memory
        for chunk_start in range(0, len(segmentation_data), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(segmentation_data))
            seg_chunk = get_segmentation_data_batch(
                segmentation_data, chunk_start, chunk_end
            )

            frames = self._get_video_frames_batch(
                cap, chunk_start, chunk_end, metadata["frame_step"]
            )

        if self.processing_plan.plan.get("generate_colored_video", False):
            colored_frames = self.visualizer.visualize_segmentation(
                frames, seg_chunk, metadata["palette"], colored_only=True
            )
            for colored_frame in colored_frames:
                video_writers["colored"].write(
                    cv2.cvtColor(colored_frame, cv2.COLOR_RGB2BGR)
                )

        if self.processing_plan.plan.get("generate_overlay_video", False):
            overlay_frames = self.visualizer.visualize_segmentation(
                frames, seg_chunk, metadata["palette"], colored_only=False
            )
            for overlay_frame in overlay_frames:
                video_writers["overlay"].write(
                    cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR)
                )

        for writer in video_writers.values():
            writer.release()

        cap.release()
        logger.debug(
            f"Video generation completed in {time.time() - start_time:.2f} seconds"
        )
        logger.debug(f"Videos saved to: {output_base}")

    def _initialize_video_writers(
        self, width: int, height: int, fps: float
    ) -> Dict[str, cv2.VideoWriter]:
        """
        Initializes video writers for output videos.

        Args:
            width (int): Width of the video frame.
            height (int): Height of the video frame.
            fps (float): Frames per second of the output video.

        Returns:
            Dict[str, cv2.VideoWriter]: A dictionary of initialized video writers.
        """
        writers = {}
        output_base = self.config.get_output_path()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if self.processing_plan.plan.get("generate_colored_video", False):
            colored_path = output_base.with_name(f"{output_base.stem}_colored.mp4")
            writers["colored"] = cv2.VideoWriter(
                str(colored_path), fourcc, fps, (width, height)
            )

        if self.processing_plan.plan.get("generate_overlay_video", False):
            overlay_path = output_base.with_name(f"{output_base.stem}_overlay.mp4")
            writers["overlay"] = cv2.VideoWriter(
                str(overlay_path), fourcc, fps, (width, height)
            )

        return writers

    @staticmethod
    def _get_video_frames_batch(
        cap: cv2.VideoCapture, start: int, end: int, frame_step: int
    ) -> List[np.ndarray]:
        """
        Gets a batch of video frames.

        Args:
            cap (cv2.VideoCapture): Video capture object.
            start (int): Start index of the batch.
            end (int): End index of the batch.
            frame_step (int): Step between frames.

        Returns:
            List[np.ndarray]: A list of video frames.
        """
        frames = []
        for frame_index in range(start * frame_step, end * frame_step, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def _create_video(
        self,
        cap: cv2.VideoCapture,
        segmentation_data: h5py.Dataset,
        metadata: Dict[str, Any],
        output_path: Path,
        colored_only: bool,
    ) -> None:
        """
        Creates a video from segmentation data.

        Args:
            cap (cv2.VideoCapture): Video capture object of the original video.
            segmentation_data (h5py.Dataset): Segmentation data for all frames.
            metadata (Dict[str, Any]): Metadata about the video and segmentation.
            output_path (Path): Path to save the output video.
            colored_only (bool): If True, create colored segmentation; if False, create overlay.
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            metadata["fps"],
            (metadata["width"], metadata["height"]),
        )

        frame_index = 0
        seg_index = 0
        palette = np.array(metadata["palette"], dtype=np.uint8)

        with tqdm_context(
            total=metadata["frame_count"],
            desc=f"Generating {'colored' if colored_only else 'overlay'} video",
            disable=self.config.disable_tqdm,
        ) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % metadata["frame_step"] == 0:
                    seg_map = segmentation_data[seg_index]
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    visualized = self.visualizer.visualize_segmentation(
                        frame_rgb, seg_map, palette, colored_only=colored_only
                    )
                    out.write(cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR))
                    seg_index += 1
                else:
                    out.write(frame)

                frame_index += 1
                pbar.update(1)

        out.release()
        logger.info(
            f"{'Colored' if colored_only else 'Overlay'} video saved to {output_path}"
        )

    def _update_processing_history(self) -> None:
        """
        Updates the processing history JSON file with the current processing information.
        """
        output_path = self.config.get_output_path()
        history_file = output_path.with_name(
            f"{output_path.stem}_processing_history.json"
        )

        try:
            if history_file.exists():
                with open(history_file, "r") as f:
                    history = json.load(f)
            else:
                history = []

            current_entry = {
                "timestamp": datetime.now().isoformat(),
                "config_hash": ConfigHasher.calculate_hash(self.config),
                "input_file": str(self.config.input),
                "output_file": str(output_path),
            }

            history.append(current_entry)

            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)

            logger.info(f"Processing history updated in {history_file}")
        except Exception as e:
            logger.error(f"Error updating processing history: {str(e)}")


class SegmentationProcessor:
    """
    Handles segmentation processing for both images and videos.

    This class serves as a facade for ImageProcessor and VideoProcessor,
    delegating the processing based on the input type.

    Attributes:
        config (Config): Configuration object containing processing parameters.
        image_processor (ImageProcessor): Processor for handling image inputs.
        video_processor (VideoProcessor): Processor for handling video inputs.
    """

    def __init__(self, config: Config):
        """
        Initializes the SegmentationProcessor with the given configuration.

        Args:
            config (Config): Configuration object for the processor.
        """
        self.config = config
        self.image_processor = ImageProcessor(config)
        self.video_processor = VideoProcessor(config)
        logger.debug(f"SegmentationProcessor initialized with config: {config}")

    def process(self):
        """
        Processes the input based on its type (image or video).

        Raises:
            ValueError: If the input type is not supported.
        """
        if self.config.input_type == InputType.SINGLE_IMAGE:
            self.image_processor.process()
        elif self.config.input_type == InputType.SINGLE_VIDEO:
            self.video_processor.process()
        else:
            raise ValueError(f"Unsupported input type: {self.config.input_type}")


class DirectoryProcessor:
    """
    Processes multiple video files in a directory.

    This class handles the batch processing of video files found in a specified directory.

    Attributes:
        config (Config): Configuration object containing processing parameters.
        video_iterator (VideoFileIterator): Iterator for video files in the directory.
        logger: Logger instance for this processor.
    """

    def __init__(self, config: Config):
        """
        Initializes the DirectoryProcessor with the given configuration.

        Args:
            config (Config): Configuration object for the processor.
        """
        self.config = config
        self.video_iterator = VideoFileIterator(config.input)
        self.logger = logger.bind(
            processor_type=self.__class__.__name__,
            input_type=self.config.input_type.value,
            input_path=str(self.config.input),
            output_path=str(self.config.get_output_path()),
            frame_step=self.config.frame_step,
        )

    def process(self) -> None:
        """
        Processes all video files in the specified directory.

        This method iterates through all video files, processing each one
        according to the configuration.

        Raises:
            InputError: If no video files are found in the directory.
        """
        self.logger.debug(
            "Starting directory processing", input_path=str(self.config.input)
        )

        if not self.video_iterator.video_files:
            self.logger.error("No video files found")
            raise InputError(f"No video files found in directory: {self.config.input}")

        output_dir = self.config.get_output_path()
        self.logger.info(
            f"Output directory set: {str(output_dir)}", output_dir=str(output_dir)
        )

        with tqdm_context(
            total=len(self.video_iterator.video_files),
            desc="Processing videos",
            disable=self.config.disable_tqdm,
        ) as pbar:
            for video_file in self.video_iterator:
                if video_file.name in self.config.ignore_files:
                    self.logger.info(
                        f"Ignoring video file: {str(video_file.name)}",
                        video_file=str(video_file),
                    )
                    pbar.update(1)
                    continue
                try:
                    self._process_single_video(video_file, output_dir)
                except Exception as e:
                    self.logger.error(
                        "Error processing video",
                        video_file=str(video_file),
                        error=str(e),
                    )
                    self.logger.debug("Error details", exc_info=True)
                finally:
                    pbar.update(1)

        self.logger.info(
            "Finished processing all videos", input_directory=str(self.config.input)
        )

    def _process_single_video(self, video_file: Path, output_dir: Path) -> None:
        """
        Processes a single video file.

        Args:
            video_file (Path): Path to the video file to process.
            output_dir (Path): Directory to save the processing results.

        Raises:
            ProcessingError: If an error occurs during video processing.
        """
        logger.debug("Creating video config...", video_file=str(video_file))
        video_config = self._create_video_config(video_file, output_dir)
        logger.debug("Video config created", video_config=video_config)

        try:
            processor = SegmentationProcessor(video_config)
            processor.process()
        except Exception as e:
            self.logger.error(
                "Error in video processing", video_file=str(video_file), error=str(e)
            )
            raise ProcessingError(f"Error processing video {video_file}: {str(e)}")

    def _create_video_config(self, video_file: Path, output_dir: Path) -> Config:
        """
        Creates a configuration object for processing a single video.

        Args:
            video_file (Path): Path to the video file.
            output_dir (Path): Directory to save the processing results.

        Returns:
            Config: Configuration object for the video processor.
        """
        return Config(
            input=video_file,
            output_dir=output_dir,
            output_prefix=None,
            model=self.config.model,
            frame_step=self.config.frame_step,
            batch_size=self.config.batch_size,
            save_raw_segmentation=self.config.save_raw_segmentation,
            save_colored_segmentation=self.config.save_colored_segmentation,
            save_overlay=self.config.save_overlay,
            visualization=self.config.visualization,
            force_reprocess=self.config.force_reprocess,
        )


def create_processor(
    config: Config,
) -> Union[SegmentationProcessor, DirectoryProcessor]:
    """
    Creates and returns the appropriate processor based on the input type.

    Args:
        config (Config): Configuration object containing processing parameters.

    Returns:
        Union[SegmentationProcessor, DirectoryProcessor]: The appropriate processor instance.
    """
    if config.input_type == InputType.DIRECTORY:
        return DirectoryProcessor(config)
    else:
        return SegmentationProcessor(config)
