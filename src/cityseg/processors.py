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
from .pipeline import create_segmentation_pipeline
from .utils import (
    get_video_files,
)


class ProcessingHistory:
    """
    A class to manage and persist the processing history of video segmentation tasks.

    This class keeps track of individual processing runs, including timestamps,
    configuration hashes, and which outputs were generated in each run.

    Attributes:
        runs (List[Dict]): A list of dictionaries, each representing a processing run.
    """

    def __init__(self):
        """Initialize an empty ProcessingHistory."""
        self.runs = []

    def add_run(
        self, timestamp: str, config_hash: str, outputs_generated: Dict[str, bool]
    ) -> None:
        """
        Add a new processing run to the history.

        Args:
            timestamp (str): The timestamp of the processing run.
            config_hash (str): A hash of the relevant configuration used for the run.
            outputs_generated (Dict[str, bool]): A dictionary indicating which outputs were generated.
        """
        self.runs.append(
            {
                "timestamp": timestamp,
                "config_hash": config_hash,
                "outputs_generated": outputs_generated,
            }
        )
        logger.debug(f"Added new processing run to history. Timestamp: {timestamp}")

    def save(self, file_path: Path) -> None:
        """
        Save the processing history to a JSON file.

        Args:
            file_path (Path): The path where the history file will be saved.
        """
        with file_path.open("w") as f:
            json.dump({"runs": self.runs}, f)
        logger.debug(f"Saved processing history to {file_path}")

    @classmethod
    def load(cls, file_path: Path) -> "ProcessingHistory":
        """
        Load a processing history from a JSON file.

        Args:
            file_path (Path): The path to the history file.

        Returns:
            ProcessingHistory: A ProcessingHistory object populated with the loaded data.
        """
        history = cls()
        if file_path.exists():
            with file_path.open("r") as f:
                data = json.load(f)
                history.runs = data["runs"]
            logger.debug(f"Loaded processing history from {file_path}")
        else:
            logger.info(f"No processing history file found at {file_path}")
        return history


class SegmentationProcessor:
    """
    A processor for semantic segmentation of images and videos.

    This class handles the segmentation process for individual image and video files,
    including caching of results, visualization generation, and statistical analysis.

    Attributes:
        config (Config): Configuration object containing processing parameters.
        pipeline (SegmentationPipeline): The segmentation pipeline used for processing.
        logger (Logger): Logger instance for tracking processing events.
        processing_history (ProcessingHistory): Object to track processing history.
        processing_plan (Dict[str, bool]): Plan determining which processing steps to execute.
    """

    def __init__(self, config: Config):
        """
        Initialize the SegmentationProcessor.

        Args:
            config (Config): Configuration object containing processing parameters.
        """
        self.config = config
        self.pipeline = create_segmentation_pipeline(
            model_name=config.model.name,
            device=config.model.device,
        )
        self.palette = (
            self.pipeline.palette
            if self.pipeline.palette is not None
            else self._generate_palette(255)
        )  # Pre-compute palette for up to 256 classes
        self.logger = logger.bind(
            processor_type=self.__class__.__name__,
            input_type=self.config.input_type.value,
        )
        self.processing_history = self._load_processing_history()
        self.processing_plan = self._create_processing_plan()
        self.logger.debug(
            "SegmentationProcessor initialized for input video.",
            video_input=str(self.config.input),
        )

    def _load_processing_history(self) -> ProcessingHistory:
        """
        Load the processing history from a file or create a new one if not found.

        Returns:
            ProcessingHistory: The loaded or newly created processing history.
        """
        history_file = self._get_history_file_path()
        try:
            history = ProcessingHistory.load(history_file)
            self.logger.debug("Processing history loaded successfully")
            return history
        except Exception as e:
            self.logger.info(
                f"Failed to load processing history: {str(e)}. Starting with a new history."
            )
            return ProcessingHistory()

    def _get_history_file_path(self) -> Path:
        """
        Get the file path for the processing history JSON file.

        Returns:
            Path: The path to the processing history file.
        """
        output_path = self.config.get_output_path()
        return output_path.with_name(f"{output_path.stem}_processing_history.json")

    def _create_processing_plan(self) -> Dict[str, bool]:
        """
        Create a processing plan based on the current configuration and existing outputs.

        Returns:
            Dict[str, bool]: A dictionary representing the processing plan.
        """
        if self.config.force_reprocess:
            self.logger.info("Force reprocessing enabled. All steps will be executed.")
            return {
                "process_video": True,
                "generate_hdf": True,
                "generate_colored_video": self.config.save_colored_segmentation,
                "generate_overlay_video": self.config.save_overlay,
                "analyze_results": self.config.analyze_results,
            }

        existing_outputs = self._check_existing_outputs()

        plan = {
            "process_video": not existing_outputs["hdf_file_valid"],
            "generate_hdf": not existing_outputs["hdf_file_valid"],
            "generate_colored_video": self.config.save_colored_segmentation
            and not existing_outputs["colored_video_valid"],
            "generate_overlay_video": self.config.save_overlay
            and not existing_outputs["overlay_video_valid"],
            "analyze_results": self.config.analyze_results
            and not existing_outputs["analysis_files_valid"],
        }

        self.logger.debug(f"Created processing plan: {plan}")
        return plan

    def _check_existing_outputs(self) -> Dict[str, bool]:
        """
        Check the validity of existing output files.

        Returns:
            Dict[str, bool]: A dictionary indicating the validity of each output type.
        """
        output_path = self.config.get_output_path()
        hdf_path = output_path.with_name(f"{output_path.stem}_segmentation.h5")
        colored_video_path = output_path.with_name(f"{output_path.stem}_colored.mp4")
        overlay_video_path = output_path.with_name(f"{output_path.stem}_overlay.mp4")
        counts_file = output_path.with_name(f"{output_path.stem}_category_counts.csv")
        percentages_file = output_path.with_name(
            f"{output_path.stem}_category_percentages.csv"
        )

        results = {
            "hdf_file_valid": False,
            "colored_video_valid": False,
            "overlay_video_valid": False,
            "analysis_files_valid": False,
        }

        if hdf_path.exists():
            results["hdf_file_valid"] = self._verify_hdf_file(hdf_path)
            self.logger.debug(f"HDF file validity: {results['hdf_file_valid']}")

        if colored_video_path.exists():
            results["colored_video_valid"] = self._verify_video_file(colored_video_path)
            self.logger.debug(
                f"Colored video validity: {results['colored_video_valid']}"
            )

        if overlay_video_path.exists():
            results["overlay_video_valid"] = self._verify_video_file(overlay_video_path)
            self.logger.debug(
                f"Overlay video validity: {results['overlay_video_valid']}"
            )

        if counts_file.exists() and percentages_file.exists():
            results["analysis_files_valid"] = self._verify_analysis_files(
                counts_file, percentages_file
            )
            self.logger.debug(
                f"Analysis files validity: {results['analysis_files_valid']}"
            )

        return results

    def _verify_analysis_files(self, counts_file: Path, percentages_file: Path) -> bool:
        """
        Verify the integrity of analysis files.

        Args:
            counts_file (Path): Path to the category counts CSV file.
            percentages_file (Path): Path to the category percentages CSV file.

        Returns:
            bool: True if both files are valid, False otherwise.
        """
        try:
            # Perform basic checks on the files
            if counts_file.stat().st_size == 0 or percentages_file.stat().st_size == 0:
                self.logger.warning("One or both analysis files are empty")
                return False

            # You could add more sophisticated checks here, such as:
            # - Verifying the number of rows matches the expected frame count
            # - Checking that the headers are correct
            # - Validating that the data is within expected ranges

            return True
        except Exception as e:
            self.logger.error(f"Error verifying analysis files: {str(e)}")
            return False

    def process(self) -> None:
        """
        Process the input based on its type (image or video).

        Raises:
            ValueError: If the input type is not supported.
        """
        if self.config.input_type == InputType.SINGLE_IMAGE:
            self.process_image()
        elif self.config.input_type == InputType.SINGLE_VIDEO:
            self.process_video()
        else:
            raise ValueError(f"Unsupported input type: {self.config.input_type}")

    def process_image(self) -> None:
        """
        Process a single image file.

        This method handles loading the image, running it through the segmentation pipeline,
        saving the results, and analyzing the segmentation map.

        Raises:
            ProcessingError: If an error occurs during image processing.
        """
        self.logger.info(f"Processing image: {self.config.input}")
        try:
            # Load and preprocess the image
            image = Image.open(self.config.input).convert("RGB")
            if self.config.model.max_size:
                image.thumbnail(
                    (self.config.model.max_size, self.config.model.max_size)
                )

            # Run segmentation
            result = self.pipeline([image])[0]

            # Save and analyze results
            self.save_results(image, result)
            self.analyze_results(result["seg_map"])

            self.logger.info("Image processing complete")
        except Exception as e:
            self.logger.exception(f"Error during image processing: {str(e)}")
            raise ProcessingError(f"Error during image processing: {str(e)}")

    def process_video(self) -> None:
        """
        Process a single video file according to the current processing plan.
        """
        self.logger.info(f"Processing video: {self.config.input.name}")
        try:
            output_path = self.config.get_output_path()
            hdf_path = output_path.with_name(f"{output_path.stem}_segmentation.h5")

            if self.processing_plan["process_video"]:
                self.logger.debug("Executing video frame processing")
                segmentation_data, metadata = self.process_video_frames()
                if self.processing_plan["generate_hdf"]:
                    self.logger.debug(
                        f"Saving segmentation data to HDF file: {hdf_path}"
                    )
                    self.save_hdf_file(hdf_path, segmentation_data, metadata)
            else:
                self.logger.info(
                    f"Loading existing segmentation data from HDF file: {hdf_path.name}"
                )
                hdf_file, metadata = self.load_hdf_file(hdf_path)
                segmentation_data = hdf_file[
                    "segmentation"
                ]  # This is now a h5py.Dataset

            # Generate videos based on the processing plan
            if (
                self.processing_plan["generate_colored_video"]
                or self.processing_plan["generate_overlay_video"]
            ):
                self.generate_videos(segmentation_data, metadata)

            # Analyze results if needed
            if self.processing_plan["analyze_results"]:
                self.logger.debug("Analyzing segmentation results")
                self.analyze_results(segmentation_data, metadata)

            # Update processing history
            self._update_processing_history()

            self.logger.info("Video processing complete")
        except Exception as e:
            self.logger.exception(f"Error during video processing: {str(e)}")
            raise ProcessingError(f"Error during video processing: {str(e)}")
        finally:
            if hasattr(self, "hdf_file"):
                self.hdf_file.close()

    def _verify_hdf_file(self, file_path: Path) -> bool:
        """
        Verify the integrity and relevance of an existing HDF file.

        Args:
            file_path (Path): Path to the HDF file.

        Returns:
            bool: True if the file is valid and up-to-date, False otherwise.
        """
        try:
            with h5py.File(file_path, "r") as f:
                if "segmentation" not in f or "metadata" not in f:
                    self.logger.warning(
                        f"HDF file at {file_path} is missing required datasets"
                    )
                    return False

                json_metadata = f["metadata"][()]
                metadata = json.loads(json_metadata)

                if metadata.get("frame_step") != self.config.frame_step:
                    self.logger.warning(
                        f"HDF file frame step ({metadata.get('frame_step')}) does not match current config ({self.config.frame_step})"
                    )
                    return False

                segmentation_data = f["segmentation"]
                if len(segmentation_data) == 0:
                    self.logger.warning(
                        f"HDF file at {file_path} contains no segmentation data"
                    )
                    return False

                first_frame = segmentation_data[0]
                last_frame = segmentation_data[-1]
                if first_frame.shape != last_frame.shape:
                    self.logger.warning(
                        f"Inconsistent frame shapes in HDF file at {file_path}"
                    )
                    return False

            self.logger.debug(f"HDF file at {file_path} is valid and up-to-date")
            return True
        except Exception as e:
            self.logger.error(f"Error verifying HDF file at {file_path}: {str(e)}")
            return False

    def _verify_video_file(self, file_path: Path) -> bool:
        """
        Verify the integrity and relevance of an existing video file.

        Args:
            file_path (Path): Path to the video file.

        Returns:
            bool: True if the file is valid and up-to-date, False otherwise.
        """
        try:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                self.logger.warning(f"Unable to open video file at {file_path}")
                return False

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, first_frame = cap.read()
            if not ret:
                self.logger.warning(
                    f"Unable to read first frame from video file at {file_path}"
                )
                return False

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            ret, last_frame = cap.read()
            if not ret:
                self.logger.warning(
                    f"Unable to read last frame from video file at {file_path}"
                )
                return False

            cap.release()
            self.logger.debug(f"Video file at {file_path} is valid and up-to-date")
            return True
        except Exception as e:
            self.logger.error(f"Error verifying video file at {file_path}: {str(e)}")
            return False

    def process_video_frames(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process video frames to generate segmentation maps.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: A tuple containing the segmentation data
            and metadata.
        """
        # Lazy import of tqdm_context to avoid circular imports
        from .utils import tqdm_context

        cap = cv2.VideoCapture(str(self.config.input))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        segmentation_data = []

        with tqdm_context(
            total=total_frames // self.config.frame_step, desc="Processing frames"
        ) as pbar:
            for batch in self._frame_generator(
                cv2.VideoCapture(str(self.config.input))
            ):
                batch_results = self._process_batch(batch)
                segmentation_data.extend(
                    [result["seg_map"] for result in batch_results]
                )
                pbar.update(len(batch))

        metadata = {
            "model_name": self.config.model.name,
            "original_video": str(self.config.input.name),
            "palette": self.pipeline.palette.tolist()
            if self.pipeline.palette is not None
            else None,
            "label_ids": self.pipeline.model.config.id2label,
            "frame_count": len(segmentation_data),
            "frame_step": self.config.frame_step,
            "total_video_frames": total_frames,
            "fps": fps,
        }

        return np.array(segmentation_data), metadata

    def _initialize_video_capture(
        self,
    ) -> Tuple[cv2.VideoCapture, int, float, int, int]:
        """
        Initialize video capture and retrieve video properties.

        Returns:
            Tuple[cv2.VideoCapture, int, float, int, int]: A tuple containing the video capture object,
            frame count, FPS, width, and height of the video.
        """
        cap = cv2.VideoCapture(str(self.config.input))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap, frame_count, original_fps, width, height

    def _initialize_video_writers(
        self, width: int, height: int, fps: float
    ) -> Dict[str, cv2.VideoWriter]:
        """
        Initialize video writers for output videos.

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

        if self.processing_plan.get("generate_colored_video", False):
            colored_path = output_base.with_name(f"{output_base.stem}_colored.mp4")
            writers["colored"] = cv2.VideoWriter(
                str(colored_path), fourcc, fps, (width, height)
            )

        if self.processing_plan.get("generate_overlay_video", False):
            overlay_path = output_base.with_name(f"{output_base.stem}_overlay.mp4")
            writers["overlay"] = cv2.VideoWriter(
                str(overlay_path), fourcc, fps, (width, height)
            )

        return writers

    def _process_batch(self, batch: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Process a batch of images through the segmentation pipeline.

        Args:
            batch (List[Image.Image]): A list of PIL Image objects to process.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing segmentation results.
        """
        return self.pipeline(batch)

    def _write_output_frames(
        self,
        batch: List[Image.Image],
        batch_results: List[Dict[str, Any]],
        video_writers: Dict[str, cv2.VideoWriter],
    ) -> None:
        """
        Write processed frames to output video files.

        Args:
            batch (List[Image.Image]): A list of original PIL Image objects.
            batch_results (List[Dict[str, Any]]): A list of segmentation results.
            video_writers (Dict[str, cv2.VideoWriter]): A dictionary of video writers.
        """
        for pil_image, result in zip(batch, batch_results):
            if "colored" in video_writers:
                colored_seg = self.visualize_segmentation(
                    pil_image, result["seg_map"], result["palette"], colored_only=True
                )
                video_writers["colored"].write(
                    cv2.cvtColor(np.array(colored_seg), cv2.COLOR_RGB2BGR)
                )
            if "overlay" in video_writers:
                overlay = self.visualize_segmentation(
                    pil_image, result["seg_map"], result["palette"], colored_only=False
                )
                video_writers["overlay"].write(
                    cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
                )

    def save_hdf_file(
        self, file_path: Path, segmentation_data: np.ndarray, metadata: Dict[str, Any]
    ) -> None:
        """
        Save segmentation data and metadata to an HDF5 file.

        Args:
            file_path (Path): Path to save the HDF5 file.
            segmentation_data (np.ndarray): Array of segmentation maps.
            metadata (Dict[str, Any]): Metadata dictionary.
        """
        with h5py.File(file_path, "w") as f:
            f.create_dataset("segmentation", data=segmentation_data, compression="gzip")

            # Convert all metadata to JSON-compatible format
            json_metadata = json.dumps(metadata)
            f.create_dataset("metadata", data=json_metadata)

    def load_hdf_file(self, file_path: Path) -> Tuple[h5py.File, Dict[str, Any]]:
        """
        Load and return the HDF5 file handle and metadata.

        Args:
            file_path (Path): Path to the HDF5 file.

        Returns:
            Tuple[h5py.File, Dict[str, Any]]: A tuple containing the HDF5 file handle
            and metadata.
        """
        self.hdf_file = h5py.File(file_path, "r")
        json_metadata = self.hdf_file["metadata"][()]
        metadata = json.loads(json_metadata)
        if "palette" in metadata:
            metadata["palette"] = np.array(metadata["palette"], np.uint8)
        return self.hdf_file, metadata

    def get_segmentation_data_batch(self, start: int, end: int) -> np.ndarray:
        """
        Get a batch of segmentation data from the HDF5 file.

        Args:
            start (int): Start index of the batch.
            end (int): End index of the batch.

        Returns:
            np.ndarray: A batch of segmentation data.
        """
        return self.hdf_file["segmentation"][start:end]

    def generate_videos(
        self, segmentation_data: h5py.Dataset, metadata: Dict[str, Any]
    ) -> None:
        """
        Generate output videos based on the processing plan, using batched processing.
        """
        if not (
            self.processing_plan.get("generate_colored_video", False)
            or self.processing_plan.get("generate_overlay_video", False)
        ):
            self.logger.info(
                "No video generation required according to the processing plan."
            )
            return

        start_time = time.time()
        cap = cv2.VideoCapture(str(self.config.input))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = metadata["fps"] / metadata["frame_step"]

        output_base = self.config.get_output_path()
        video_writers = self._initialize_video_writers(width, height, fps)

        chunk_size = 100  # Adjust this value based on your memory constraints and performance needs
        for chunk_start in range(0, len(segmentation_data), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(segmentation_data))
            seg_chunk = self.get_segmentation_data_batch(chunk_start, chunk_end)

            frames = self._get_video_frames_batch(
                cap, chunk_start, chunk_end, metadata["frame_step"]
            )

            if self.processing_plan.get("generate_colored_video", False):
                colored_frames = self.visualize_segmentation(
                    frames, seg_chunk, metadata["palette"], colored_only=True
                )
                for colored_frame in colored_frames:
                    video_writers["colored"].write(
                        cv2.cvtColor(colored_frame, cv2.COLOR_RGB2BGR)
                    )

            if self.processing_plan.get("generate_overlay_video", False):
                overlay_frames = self.visualize_segmentation(
                    frames, seg_chunk, metadata["palette"], colored_only=False
                )
                for overlay_frame in overlay_frames:
                    video_writers["overlay"].write(
                        cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR)
                    )

        for writer in video_writers.values():
            writer.release()
        cap.release()

        self.logger.debug(
            f"Video generation took {time.time() - start_time:.4f} seconds"
        )
        self.logger.debug(f"Videos saved to: {output_base}")

    def _get_video_frames_batch(
        self, cap: cv2.VideoCapture, start: int, end: int, frame_step: int
    ) -> List[np.ndarray]:
        """
        Get a batch of video frames.

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

    def _update_processing_history(self) -> None:
        """
        Update the processing history with the current run's information.
        """
        config_hash = ConfigHasher.calculate_hash(self.config)
        self.processing_history.add_run(
            timestamp=datetime.now().isoformat(),
            config_hash=config_hash,
            outputs_generated=self.processing_plan,
        )
        self.processing_history.save(self._get_history_file_path())

    def analyze_results(
        self, segmentation_data: h5py.Dataset, metadata: Dict[str, Any]
    ) -> None:
        """
        Analyze segmentation results and generate statistics using chunked processing.

        Args:
            segmentation_data (h5py.Dataset): Memory-mapped segmentation data.
            metadata (Dict[str, Any]): Metadata dictionary.
        """
        output_path = self.config.get_output_path()
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
                seg_chunk = self.get_segmentation_data_batch(chunk_start, chunk_end)

                for frame_idx, seg_map in enumerate(seg_chunk, start=chunk_start):
                    analysis = self.analyze_segmentation_map(seg_map, len(id2label))
                    frame_number = frame_idx * metadata["frame_step"]

                    counts_row = [frame_number] + [
                        analysis[i][0] for i in sorted(analysis.keys())
                    ]
                    percentages_row = [frame_number] + [
                        analysis[i][1] for i in sorted(analysis.keys())
                    ]

                    counts_writer.writerow(counts_row)
                    percentages_writer.writerow(percentages_row)

        self._generate_category_stats(
            counts_file, output_path.with_name(f"{output_path.stem}_counts_stats.csv")
        )
        self._generate_category_stats(
            percentages_file,
            output_path.with_name(f"{output_path.stem}_percentages_stats.csv"),
        )

    @staticmethod
    def analyze_segmentation_map(
        seg_map: np.ndarray, num_categories: int
    ) -> Dict[int, tuple[int, float]]:
        """
        Analyze a segmentation map to compute pixel counts and percentages for each category.

        Args:
            seg_map (np.ndarray): The segmentation map to analyze.
            num_categories (int): The total number of categories in the segmentation.

        Returns:
            Dict[int, tuple[int, float]]: A dictionary where keys are category IDs and values
            are tuples of (pixel count, percentage) for each category.
        """
        unique, counts = np.unique(seg_map, return_counts=True)
        total_pixels = seg_map.size
        category_analysis = {i: (0, 0.0) for i in range(num_categories)}

        for category_id, pixel_count in zip(unique, counts):
            percentage = (pixel_count / total_pixels) * 100
            category_analysis[int(category_id)] = (int(pixel_count), float(percentage))

        return category_analysis

    def _generate_category_stats(self, input_file: Path, output_file: Path) -> None:
        """
        Generate category statistics from input CSV file.

        Args:
            input_file (Path): Path to the input CSV file (counts or percentages).
            output_file (Path): Path to save the output statistics CSV file.
        """
        try:
            # Read the entire CSV file
            df = pd.read_csv(input_file)

            # Exclude the 'Frame' column and calculate statistics
            category_columns = df.columns[1:]
            stats = df[category_columns].agg(["mean", "median", "std", "min", "max"])

            # Transpose the results for a more readable output
            stats = stats.transpose()

            # Save the statistics to the output file
            stats.to_csv(output_file)

            self.logger.info(f"Category statistics saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error generating category stats: {str(e)}")
            raise

    def visualize_segmentation(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        seg_maps: Union[np.ndarray, List[np.ndarray]],
        palette: Optional[np.ndarray],
        colored_only: bool = False,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Visualize the segmentation maps for multiple images.

        Args:
            images (Union[np.ndarray, List[np.ndarray]]): The original images or a single image.
            seg_maps (Union[np.ndarray, List[np.ndarray]]): The segmentation maps or a single segmentation map.
            palette (Optional[np.ndarray]): The color palette for visualization.
            colored_only (bool): If True, return only the colored segmentation maps.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The visualized segmentation maps or overlays.
        """
        if palette is None:
            palette = self._generate_palette(256)  # Assuming max 256 classes

        # Convert single image/seg_map to list for uniform processing
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
            seg_maps = [seg_maps]

        results = []
        for image, seg_map in zip(images, seg_maps):
            # Vectorized color application
            color_seg = palette[seg_map]

            if colored_only:
                results.append(color_seg)
            else:
                img = image * 0.5 + color_seg * 0.5
                results.append(img.astype(np.uint8))

        return results[0] if len(results) == 1 else results

    def _frame_generator(self, cap: cv2.VideoCapture) -> Iterator[List[Image.Image]]:
        """
        Generate batches of frames from a video capture object.

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

    def _generate_palette(self, num_colors: int) -> np.ndarray:
        """
        Generate a color palette for visualization.

        Args:
            num_colors (int): The number of colors to generate.

        Returns:
            np.ndarray: The generated color palette.
        """

        return np.array(
            [
                [(i * 100) % 255, (i * 150) % 255, (i * 200) % 255]
                for i in range(num_colors)
            ],
            dtype=np.uint8,
        )


class DirectoryProcessor:
    """
    A processor for handling multiple video files in a directory.

    This class manages the processing of multiple video files within a specified
    directory, utilizing the SegmentationProcessor for individual video processing.

    Attributes:
        config (Config): Configuration object containing processing parameters.
        logger (Logger): Logger instance for tracking processing events.
    """

    def __init__(self, config: Config):
        """
        Initialize the DirectoryProcessor.

        Args:
            config (Config): Configuration object containing processing parameters.
        """
        self.config = config
        self.logger = logger.bind(
            processor_type=self.__class__.__name__,
            input_type=self.config.input_type.value,
            input_path=str(self.config.input),
            output_path=str(self.config.get_output_path()),
            frame_step=self.config.frame_step,
        )

    def process(self) -> None:
        """
        Process all video files in the specified directory.

        This method identifies all video files in the input directory,
        processes each video using SegmentationProcessor, and handles any errors
        that occur during processing.

        Raises:
            InputError: If no video files are found in the specified directory.
        """
        # Lazy import of tqdm_context
        from .utils import tqdm_context

        self.logger.debug(
            "Starting directory processing", input_path=str(self.config.input)
        )

        video_files = self.get_video_files()
        if not video_files:
            self.logger.error("No video files found")
            raise InputError(f"No video files found in directory: {self.config.input}")

        output_dir = self.config.get_output_path()
        self.logger.info(
            f"Output directory set: {str(output_dir)}", output_dir=str(output_dir)
        )

        with tqdm_context(total=len(video_files), desc="Processing videos") as pbar:
            for video_file in video_files:
                if video_file.name in self.config.ignore_files:
                    self.logger.info(
                        f"Ignoring video file: {str(video_file.name)}",
                        video_file=str(video_file),
                    )
                    pbar.update(1)
                    continue
                try:
                    self.process_single_video(video_file, output_dir)
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

    def get_video_files(self) -> List[Path]:
        """
        Get a list of video files in the input directory.

        Returns:
            List[Path]: A list of paths to video files.
        """
        video_files = get_video_files(self.config.input)
        self.logger.info(f"Found {len(video_files)} video files in {self.config.input}")
        return video_files

    def process_single_video(self, video_file: Path, output_dir: Path) -> None:
        """
        Process a single video file.

        Args:
            video_file (Path): Path to the video file to process.
            output_dir (Path): Directory to save the processing results.

        Raises:
            ProcessingError: If an error occurs during video processing.
        """
        video_config = self.create_video_config(video_file, output_dir)

        try:
            processor = SegmentationProcessor(video_config)
            processor.process()
        except Exception as e:
            self.logger.error(
                "Error in video processing", video_file=str(video_file), error=str(e)
            )
            raise ProcessingError(f"Error processing video {video_file}: {str(e)}")

    def create_video_config(self, video_file: Path, output_dir: Path) -> Config:
        """
        Create a configuration object for processing a single video.

        Args:
            video_file (Path): Path to the video file.
            output_dir (Path): Directory to save the processing results.

        Returns:
            Config: A configuration object for the video processing.
        """
        return Config(
            input=video_file,
            output_dir=output_dir,
            output_prefix=None,
            model=self.config.model,
            frame_step=self.config.frame_step,
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
    Create and return the appropriate processor based on the input type.

    Args:
        config (Config): Configuration object containing processing parameters.

    Returns:
        Union[SegmentationProcessor, DirectoryProcessor]: An instance of either
        SegmentationProcessor or DirectoryProcessor, depending on the input type.
    """
    if config.input_type == InputType.DIRECTORY:
        return DirectoryProcessor(config)
    else:
        return SegmentationProcessor(config)
