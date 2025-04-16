"""
This module provides classes and functions for processing images and videos
using semantic segmentation models.

It includes processors for handling individual files (images or videos) and
directories containing multiple video files. The module also manages caching
of segmentation results, generation of output visualizations, and analysis
of segmentation statistics.

Each processor class provides focused functionality following the single 
responsibility principle, and can be used independently or combined through 
the Hamilton-based workflow defined in the hamilton_driver and 
hamilton_functions modules.
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union

import cv2
import numpy as np
import xarray as xr
from loguru import logger
from PIL import Image

from .config import Config, ConfigHasher, InputType, ModelConfig
from .exceptions import InputError, ProcessingError
from .file_handler import FileHandler
from .image_processor import ImageProcessor
from .pipeline import create_segmentation_pipeline
from .processing_plan import ProcessingPlan
from .segmentation_analyzer import SegmentationAnalyzer
from .segmentation_processor import SegmentationProcessor
from .storage_adapter import ZarrSegmentationStorage, ParquetAnalysisStorage, StorageFactory
from .utils import get_segmentation_batch, tqdm_context
from .video_file_iterator import VideoFileIterator
from .video_processor import VideoProcessor
from .video_resource import VideoResource
from .visualization_handler import VisualizationHandler
from .dataset_builder import DatasetBuilder
from .hamilton_driver import process as hamilton_process


class ImageProcessorLegacy:
    """
    Processes individual images using semantic segmentation models.

    This class handles the segmentation of single images, including saving results
    and analyzing the segmentation output. It now delegates to the specialized component
    classes and ultimately to Hamilton for orchestration.

    Attributes:
        config (Config): Configuration object containing processing parameters.
    """

    def __init__(self, config: Config):
        """
        Initializes the ImageProcessor with the given configuration.

        Args:
            config (Config): Configuration object for the processor.
        """
        self.config = config

    def process(self) -> None:
        """
        Processes the input image according to the configuration.

        This method delegates to the Hamilton-based implementation which handles
        the entire image processing pipeline including segmentation, result 
        saving, and analysis.

        Raises:
            ProcessingError: If an error occurs during image processing.
        """
        logger.info(f"Processing image: {self.config.input}")
        try:
            # Use the Hamilton driver to process the image
            result = hamilton_process(self.config)
            
            # Check for errors
            if 'error' in result:
                raise ProcessingError(f"Error in Hamilton workflow: {result['error']}")
                
            logger.info("Image processing complete")
        except Exception as e:
            logger.exception(f"Error during image processing: {str(e)}")
            raise ProcessingError(f"Error during image processing: {str(e)}")
            
    def process_direct(self) -> Dict[str, Any]:
        """
        Processes the input image using direct component calls without Hamilton.
        
        This method provides an alternative API for advanced users who want to
        manually control the processing pipeline without Hamilton orchestration.
        
        Returns:
            Dict[str, Any]: Dictionary containing processing results.
        """
        try:
            logger.info(f"Direct processing of image: {self.config.input}")
            
            # Load and preprocess the image
            image = ImageProcessor.load_image(self.config.input)
            if self.config.model.max_size:
                image = ImageProcessor.resize_image(image, self.config.model.max_size)
                
            # Create segmentation pipeline
            pipeline = SegmentationProcessor.create_pipeline(self.config.model)
            
            # Process the image
            result = SegmentationProcessor.process_image(image, pipeline)
            
            # Get metadata
            metadata = {
                "label2id": result.get("label2id", {}),
                "id2label": result.get("id2label", {}),
                "palette": result.get("palette", None)
            }
            
            # Save results
            output_path = self.config.get_output_path()
            visualizer = VisualizationHandler()
            
            output_files = {}
            
            # Save raw segmentation
            if self.config.save_raw_segmentation:
                raw_seg_path = output_path.with_name(f"{output_path.stem}_raw_segmentation.png")
                ImageProcessor.save_image(result["seg_map"], raw_seg_path)
                output_files["raw_segmentation"] = str(raw_seg_path)
                logger.info(f"Raw segmentation saved to {raw_seg_path}")
            
            # Save colored segmentation
            if self.config.save_colored_segmentation:
                colored_seg_path = output_path.with_name(f"{output_path.stem}_colored_segmentation.png")
                colored_seg = visualizer.visualize_segmentation(
                    np.array(image), result["seg_map"], result["palette"], colored_only=True
                )
                ImageProcessor.save_image(colored_seg, colored_seg_path)
                output_files["colored_segmentation"] = str(colored_seg_path)
                logger.info(f"Colored segmentation saved to {colored_seg_path}")
            
            # Save overlay
            if self.config.save_overlay:
                overlay_path = output_path.with_name(f"{output_path.stem}_overlay.png")
                overlay = visualizer.visualize_segmentation(
                    np.array(image), result["seg_map"], result["palette"], colored_only=False
                )
                ImageProcessor.save_image(overlay, overlay_path)
                output_files["overlay"] = str(overlay_path)
                logger.info(f"Overlay saved to {overlay_path}")
            
            # Analyze results
            if self.config.analyze_results:
                analyzer = SegmentationAnalyzer()
                num_categories = len(result.get("id2label", {}))
                analysis = analyzer.analyze_segmentation_map(result["seg_map"], num_categories)
                
                # Extract counts and percentages
                counts = {category_id: count for category_id, (count, _) in analysis.items()}
                percentages = {category_id: percentage for category_id, (_, percentage) in analysis.items()}
                
                # Save analysis
                analysis_storage = ParquetAnalysisStorage()
                parquet_path = analysis_storage.save_category_analysis(
                    counts,
                    percentages,
                    output_path.with_name(f"{output_path.stem}_category_analysis")
                )
                output_files["analysis"] = str(parquet_path)
                logger.info(f"Category analysis saved to {parquet_path}")
            
            logger.info("Image direct processing complete")
            return {
                "image": image,
                "result": result,
                "metadata": metadata,
                "output_files": output_files
            }
            
        except Exception as e:
            logger.exception(f"Error during direct image processing: {str(e)}")
            return {"error": str(e)}


class VideoProcessorLegacy:
    """
    Processes video files using semantic segmentation models.

    This class handles the segmentation of video frames, including saving results,
    generating output videos, and analyzing the segmentation output. It now
    delegates to the specialized component classes and ultimately to Hamilton
    for orchestration.

    Attributes:
        config (Config): Configuration object containing processing parameters.
        processing_plan (ProcessingPlan): Plan for video processing steps.
    """

    def __init__(self, config: Config):
        """
        Initializes the VideoProcessor with the given configuration.

        Args:
            config (Config): Configuration object for the processor.
        """
        self.config = config
        self.processing_plan = ProcessingPlan(config)
        logger.debug(f"VideoProcessorLegacy initialized with config: {config}")

    def get_output_video_path(self) -> Path:
        """
        Returns the output path for the processed video.

        Returns:
            Path: The output path for the processed video.
        """
        return self.config.get_output_path()

    def get_output_segmentation_path(self) -> Path:
        """
        Returns the output path for the processed segmentation data file.

        Returns:
            Path: The output path for the processed segmentation data file.
        """
        output_path = self.config.get_output_path()
        return output_path.with_name(f"{output_path.stem}_segmentation.zarr")

    def load_segmentation_data(self) -> Tuple[xr.Dataset, Dict[str, Any]]:
        """
        Loads segmentation data and metadata from a Zarr file.

        Returns:
            Tuple[xr.Dataset, Dict[str, Any]]: Loaded segmentation dataset and metadata.
        """
        storage = ZarrSegmentationStorage()
        return storage.load_segmentation_data(self.get_output_segmentation_path())

    def process(self) -> None:
        """
        Processes the input video according to the configuration and processing plan.

        This method delegates to the Hamilton-based implementation which handles
        the entire video processing pipeline including frame segmentation, result
        saving, video generation, and analysis.

        Raises:
            ProcessingError: If an error occurs during video processing.
        """
        logger.info(f"Processing video: {self.config.input.name}")
        try:
            # Use the Hamilton driver to process the video
            if self.processing_plan.plan["process_video"]:
                logger.debug("Using Hamilton workflow to process video")
                result = hamilton_process(self.config)
                
                if 'error' in result:
                    raise ProcessingError(f"Error in Hamilton workflow: {result['error']}")
                
                # For backward compatibility with existing visualization code
                segmentation_dataset = result['dataset']
                metadata = result['metadata']
            else:
                # Load existing segmentation data
                zarr_path = self.get_output_segmentation_path()
                logger.info(
                    f"Loading existing segmentation data from Zarr file: {zarr_path.name}"
                )
                segmentation_dataset, metadata = self.load_segmentation_data()

            # Generate videos based on the processing plan
            if (
                self.processing_plan.plan["generate_colored_video"]
                or self.processing_plan.plan["generate_overlay_video"]
            ):
                self._generate_videos_direct(segmentation_dataset, metadata)

            # Update processing history
            self._update_processing_history()

            logger.info("Video processing complete")
        except Exception as e:
            logger.exception(f"Error during video processing: {str(e)}")
            raise ProcessingError(f"Error during video processing: {str(e)}")

    def process_direct(self) -> Dict[str, Any]:
        """
        Processes a video using direct component calls without Hamilton.
        
        This method provides an alternative API for advanced users who want to
        manually control the processing pipeline without Hamilton orchestration.
        
        Returns:
            Dict[str, Any]: Dictionary containing processing results.
        """
        try:
            logger.info(f"Direct processing of video: {self.config.input}")
            output_path = self.get_output_video_path()
            
            # Get video metadata
            video_metadata = VideoProcessor.get_metadata(self.config.input)
            
            # Determine frame indices
            frame_indices = VideoProcessor.get_frame_indices(
                video_metadata["frame_count"], 
                self.config.frame_step
            )
            
            # Get frames
            frames = VideoProcessor.get_frames(self.config.input, frame_indices)
            
            # Create segmentation pipeline
            pipeline = SegmentationProcessor.create_pipeline(self.config.model)
            
            # Process frames
            batch_results = SegmentationProcessor.process_batch(frames, pipeline)
            seg_maps = SegmentationProcessor.extract_segmentation_maps(batch_results)
            seg_metadata = SegmentationProcessor.extract_metadata(batch_results)
            
            # Create dataset
            dataset = DatasetBuilder.create_video_dataset(
                seg_maps,
                video_metadata,
                frame_indices,
                self.config.model.to_dict(),
                seg_metadata
            )
            
            # Save segmentation data
            segmentation_path = DatasetBuilder.save_segmentation(
                dataset, 
                output_path
            )
            
            # Save analysis if requested
            analysis_path = None
            if self.config.analyze_results:
                analysis_path = DatasetBuilder.save_analysis(
                    dataset,
                    output_path
                )
            
            # Generate videos if requested
            visualization_paths = {}
            if self.processing_plan.plan["generate_colored_video"] or self.processing_plan.plan["generate_overlay_video"]:
                visualization_paths = self._generate_videos_direct(dataset, dict(dataset.attrs))
            
            # Update processing history
            self._update_processing_history()
            
            logger.info("Video direct processing complete")
            return {
                "metadata": video_metadata,
                "dataset": dataset,
                "segmentation_path": str(segmentation_path),
                "analysis_path": str(analysis_path) if analysis_path else None,
                "visualization_paths": visualization_paths
            }
            
        except Exception as e:
            logger.exception(f"Error during direct video processing: {str(e)}")
            return {"error": str(e)}

    def _generate_videos_direct(
        self, segmentation_dataset: xr.Dataset, metadata: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generates output videos using direct component calls.

        Args:
            segmentation_dataset (xr.Dataset): The segmentation dataset containing all frames.
            metadata (Dict[str, Any]): Metadata about the video and segmentation.
            
        Returns:
            Dict[str, str]: Dictionary mapping video types to their file paths.
        """
        if not (
            self.processing_plan.plan.get("generate_colored_video", False)
            or self.processing_plan.plan.get("generate_overlay_video", False)
        ):
            logger.info("No video generation required according to the processing plan")
            return {}

        start_time = time.time()
        visualization_paths = {}
        
        # Get video metadata
        video_metadata = VideoProcessor.get_metadata(self.config.input)
        
        width = video_metadata["width"]
        height = video_metadata["height"]
        fps = metadata.get("fps", video_metadata["fps"]) / metadata.get("frame_step", 1)
        
        # Get palette from metadata
        palette = np.array(metadata.get("palette", []), dtype=np.uint8)
        if len(palette) == 0 and "palette" in segmentation_dataset.attrs:
            palette_attr = segmentation_dataset.attrs.get("palette")
            if isinstance(palette_attr, list):
                palette = np.array(palette_attr, dtype=np.uint8)

        output_base = self.config.get_output_path()
        
        # Initialize video writers
        writers = {}
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if self.processing_plan.plan.get("generate_colored_video", False):
            colored_path = output_base.with_name(f"{output_base.stem}_colored.mp4")
            writers["colored"] = cv2.VideoWriter(
                str(colored_path), fourcc, fps, (width, height)
            )
            visualization_paths["colored"] = str(colored_path)

        if self.processing_plan.plan.get("generate_overlay_video", False):
            overlay_path = output_base.with_name(f"{output_base.stem}_overlay.mp4")
            writers["overlay"] = cv2.VideoWriter(
                str(overlay_path), fourcc, fps, (width, height)
            )
            visualization_paths["overlay"] = str(overlay_path)
        
        # Get access to the segmentation data
        segmentation_data = segmentation_dataset.segmentation
        total_frames = segmentation_data.shape[0]
        visualizer = VisualizationHandler()

        # Process in chunks for memory efficiency
        chunk_size = 100  # Adjust this value based on available memory
        for chunk_start in range(0, total_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_frames)
            
            # Get segmentation data batch using xarray
            seg_chunk = segmentation_data.isel(time=slice(chunk_start, chunk_end)).values
            
            # Get the corresponding video frames
            frame_indices = list(range(chunk_start, chunk_end))
            # Map these to actual frame indices in the video using frame_step
            frame_step = metadata.get("frame_step", 1)
            video_frame_indices = [idx * int(frame_step) for idx in frame_indices]
            
            # Load the frames
            frames = VideoProcessor.get_frames(self.config.input, video_frame_indices)
            frames_np = [np.array(frame) for frame in frames]
            
            # Generate and write video frames
            if self.processing_plan.plan.get("generate_colored_video", False):
                colored_frames = visualizer.visualize_segmentation(
                    frames_np, seg_chunk, palette, colored_only=True
                )
                for colored_frame in colored_frames:
                    writers["colored"].write(
                        cv2.cvtColor(colored_frame, cv2.COLOR_RGB2BGR)
                    )

            if self.processing_plan.plan.get("generate_overlay_video", False):
                overlay_frames = visualizer.visualize_segmentation(
                    frames_np, seg_chunk, palette, colored_only=False
                )
                for overlay_frame in overlay_frames:
                    writers["overlay"].write(
                        cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR)
                    )

        # Release all resources
        for writer in writers.values():
            writer.release()
            
        logger.debug(
            f"Video generation completed in {time.time() - start_time:.2f} seconds"
        )
        logger.debug(f"Videos saved to: {output_base}")
        
        return visualization_paths

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


class SegmentationProcessorWrapper:
    """
    Handles segmentation processing for both images and videos.

    This class serves as a facade for ImageProcessorLegacy and VideoProcessorLegacy,
    delegating the processing based on the input type. It acts as a compatibility
    layer for existing code while the system transitions to the Hamilton-based workflow.

    Attributes:
        config (Config): Configuration object containing processing parameters.
        image_processor (ImageProcessorLegacy): Processor for handling image inputs.
        video_processor (VideoProcessorLegacy): Processor for handling video inputs.
    """

    def __init__(self, config: Config):
        """
        Initializes the SegmentationProcessorWrapper with the given configuration.

        Args:
            config (Config): Configuration object for the processor.
        """
        self.config = config
        self.image_processor = ImageProcessorLegacy(config)
        self.video_processor = VideoProcessorLegacy(config)
        logger.debug(f"SegmentationProcessorWrapper initialized with config: {config}")

    def process(self):
        """
        Processes the input based on its type (image or video).
        
        This method delegates to the appropriate processor instance which
        ultimately uses the Hamilton-based implementation.

        Raises:
            ValueError: If the input type is not supported.
        """
        if self.config.input_type == InputType.SINGLE_IMAGE:
            self.image_processor.process()
        elif self.config.input_type == InputType.SINGLE_VIDEO:
            self.video_processor.process()
        else:
            raise ValueError(f"Unsupported input type: {self.config.input_type}")
            
    def process_direct(self) -> Dict[str, Any]:
        """
        Processes the input directly without using Hamilton.
        
        This method provides an alternative API for advanced users who want to
        manually control the processing pipeline without Hamilton orchestration.
        
        Returns:
            Dict[str, Any]: Dictionary containing processing results.
        """
        if self.config.input_type == InputType.SINGLE_IMAGE:
            return self.image_processor.process_direct()
        elif self.config.input_type == InputType.SINGLE_VIDEO:
            return self.video_processor.process_direct()
        else:
            return {
                "error": f"Unsupported input type for direct processing: {self.config.input_type}"
            }


class DirectoryProcessorLegacy:
    """
    Processes multiple video files in a directory.

    This class handles the batch processing of video files found in a specified directory.
    It now delegates to the Hamilton-based implementation for each video file.

    Attributes:
        config (Config): Configuration object containing processing parameters.
        video_iterator (VideoFileIterator): Iterator for video files in the directory.
        logger: Logger instance for this processor.
    """

    def __init__(self, config: Config):
        """
        Initializes the DirectoryProcessorLegacy with the given configuration.

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
        according to the configuration. It delegates to the Hamilton-based 
        implementation for each individual file.

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

    def process_direct(self) -> Dict[str, Dict[str, Any]]:
        """
        Processes all video files in the directory using direct component calls.
        
        This method provides an alternative API for advanced users who want to
        manually control the processing pipeline without Hamilton orchestration.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping file paths to their processing results.
        """
        self.logger.debug(
            "Starting direct directory processing", input_path=str(self.config.input)
        )
        
        results = {}
        
        if not self.video_iterator.video_files:
            self.logger.error("No video files found")
            return {"error": f"No video files found in directory: {self.config.input}"}
            
        output_dir = self.config.get_output_path()
        self.logger.info(
            f"Output directory set: {str(output_dir)}", output_dir=str(output_dir)
        )
        
        for video_file in self.video_iterator:
            if video_file.name in self.config.ignore_files:
                self.logger.info(
                    f"Ignoring video file: {str(video_file.name)}",
                    video_file=str(video_file),
                )
                results[str(video_file)] = {"status": "skipped"}
                continue
                
            try:
                video_config = self._create_video_config(video_file, output_dir)
                processor = SegmentationProcessorWrapper(video_config)
                results[str(video_file)] = processor.process_direct()
            except Exception as e:
                self.logger.error(
                    "Error in direct video processing", 
                    video_file=str(video_file), 
                    error=str(e)
                )
                results[str(video_file)] = {"error": str(e)}
                
        self.logger.info(
            "Finished direct processing of all videos", 
            input_directory=str(self.config.input)
        )
        return results

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
            # Use Hamilton to process the video
            result = hamilton_process(video_config)
            
            if 'error' in result:
                raise ProcessingError(f"Error in Hamilton workflow: {result['error']}")
                
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
            disable_tqdm=self.config.disable_tqdm,
        )


def create_processor(
    config: Config,
) -> Union[SegmentationProcessorWrapper, DirectoryProcessorLegacy]:
    """
    Creates and returns the appropriate processor based on the input type.

    This function serves as a factory for creating processor instances that
    delegate to the Hamilton-based implementation. It maintains API compatibility
    with existing code while transitioning to Hamilton for orchestration.

    Args:
        config (Config): Configuration object containing processing parameters.

    Returns:
        Union[SegmentationProcessorWrapper, DirectoryProcessorLegacy]: The appropriate processor instance.
    """
    if config.input_type == InputType.DIRECTORY:
        return DirectoryProcessorLegacy(config)
    else:
        return SegmentationProcessorWrapper(config)
