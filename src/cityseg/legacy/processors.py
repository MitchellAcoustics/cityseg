"""
This module provides legacy processor classes for backward compatibility.

These classes maintain the original interface while delegating to the new
component-based architecture.
"""

from typing import Any, Dict, Union

from loguru import logger

from ..core import Config, InputType, ProcessingError
from ..components.image import ImageProcessor
from ..components.video import VideoProcessor
from ..components.segmentation import SegmentationProcessor
from ..components.dataset import DatasetBuilder
from ..workflow.hamilton import process as hamilton_process


class ImageProcessorLegacy:
    """
    Legacy adapter for image processing.

    This class maintains the original interface while delegating to the
    new component-based architecture.
    """

    def __init__(self, config: Config):
        """
        Initialize with configuration.

        Args:
            config: Configuration for processing
        """
        self.config = config

    def process(self) -> None:
        """
        Process the input image according to the configuration.

        This method delegates to the Hamilton-based implementation.

        Raises:
            ProcessingError: If an error occurs during processing
        """
        logger.info(f"Processing image: {self.config.input}")
        try:
            # Use the Hamilton driver to process the image
            result = hamilton_process(self.config)

            # Check for errors
            if "error" in result:
                raise ProcessingError(f"Error in Hamilton workflow: {result['error']}")

            logger.info("Image processing complete")
        except Exception as e:
            logger.exception(f"Error during image processing: {str(e)}")
            raise ProcessingError(f"Error during image processing: {str(e)}")

    def process_direct(self) -> Dict[str, Any]:
        """
        Process the image directly using component classes.

        This method provides access to the direct component API for advanced users.

        Returns:
            Dictionary with processing results

        Raises:
            ProcessingError: If an error occurs during processing
        """
        logger.info(f"Processing image directly: {self.config.input}")
        try:
            # Load and resize image
            image = ImageProcessor.load_image(self.config.input)
            resized_image = ImageProcessor.resize_image(
                image, self.config.model.max_size
            )

            # Create segmentation pipeline
            pipeline = SegmentationProcessor.create_pipeline(self.config.model)

            # Process image
            result = SegmentationProcessor.process_image(resized_image, pipeline)
            seg_map = result["seg_map"]

            # Create dataset
            dataset = DatasetBuilder.create_image_dataset(
                seg_map,
                self.config.model.to_dict(),
                SegmentationProcessor.extract_metadata([result]),
            )

            # Save results
            output_path = self.config.get_output_path()
            segmentation_path = DatasetBuilder.save_segmentation(dataset, output_path)

            if self.config.analyze_results:
                analysis_path = DatasetBuilder.save_analysis(dataset, output_path)
            else:
                analysis_path = None

            return {
                "segmentation_path": segmentation_path,
                "analysis_path": analysis_path,
                "dataset": dataset,
            }
        except Exception as e:
            logger.exception(f"Error during direct image processing: {str(e)}")
            raise ProcessingError(f"Error during direct image processing: {str(e)}")


class VideoProcessorLegacy:
    """
    Legacy adapter for video processing.

    This class maintains the original interface while delegating to the
    new component-based architecture.
    """

    def __init__(self, config: Config):
        """
        Initialize with configuration.

        Args:
            config: Configuration for processing
        """
        self.config = config

    def process(self) -> None:
        """
        Process the input video according to the configuration.

        This method delegates to the Hamilton-based implementation.

        Raises:
            ProcessingError: If an error occurs during processing
        """
        logger.info(f"Processing video: {self.config.input}")
        try:
            # Use the Hamilton driver to process the video
            result = hamilton_process(self.config)

            # Check for errors
            if "error" in result:
                raise ProcessingError(f"Error in Hamilton workflow: {result['error']}")

            logger.info("Video processing complete")
        except Exception as e:
            logger.exception(f"Error during video processing: {str(e)}")
            raise ProcessingError(f"Error during video processing: {str(e)}")

    def process_direct(self) -> Dict[str, Any]:
        """
        Process the video directly using component classes.

        This method provides access to the direct component API for advanced users.

        Returns:
            Dictionary with processing results

        Raises:
            ProcessingError: If an error occurs during processing
        """
        logger.info(f"Processing video directly: {self.config.input}")
        try:
            # Get video metadata and frame indices
            video_metadata = VideoProcessor.get_metadata(self.config.input)
            frame_indices = VideoProcessor.get_frame_indices(
                video_metadata["frame_count"], self.config.frame_step
            )

            # Extract frames
            frames = VideoProcessor.get_frames(self.config.input, frame_indices)

            # Resize frames if needed
            resized_frames = [
                ImageProcessor.resize_image(frame, self.config.model.max_size)
                for frame in frames
            ]

            # Create segmentation pipeline
            pipeline = SegmentationProcessor.create_pipeline(self.config.model)

            # Process frames
            results = SegmentationProcessor.process_batch(resized_frames, pipeline)
            segmentation_maps = SegmentationProcessor.extract_segmentation_maps(results)
            segmentation_metadata = SegmentationProcessor.extract_metadata(results)

            # Create dataset
            dataset = DatasetBuilder.create_video_dataset(
                segmentation_maps,
                video_metadata,
                frame_indices,
                self.config.model.to_dict(),
                segmentation_metadata,
            )

            # Save results
            output_path = self.config.get_output_path()
            segmentation_path = DatasetBuilder.save_segmentation(dataset, output_path)

            if self.config.analyze_results:
                analysis_path = DatasetBuilder.save_analysis(dataset, output_path)
            else:
                analysis_path = None

            return {
                "segmentation_path": segmentation_path,
                "analysis_path": analysis_path,
                "dataset": dataset,
            }
        except Exception as e:
            logger.exception(f"Error during direct video processing: {str(e)}")
            raise ProcessingError(f"Error during direct video processing: {str(e)}")


class DirectoryProcessorLegacy:
    """
    Legacy adapter for directory processing.

    This class maintains the original interface while delegating to the
    new component-based architecture.
    """

    def __init__(self, config: Config):
        """
        Initialize with configuration.

        Args:
            config: Configuration for processing
        """
        self.config = config

    def process(self) -> None:
        """
        Process the input directory according to the configuration.

        This method delegates to the Hamilton-based implementation.

        Raises:
            ProcessingError: If an error occurs during processing
        """
        logger.info(f"Processing directory: {self.config.input}")
        try:
            # Use the Hamilton driver to process the directory
            result = hamilton_process(self.config)

            # Check for errors
            if "error" in result:
                raise ProcessingError(f"Error in Hamilton workflow: {result['error']}")

            logger.info("Directory processing complete")
        except Exception as e:
            logger.exception(f"Error during directory processing: {str(e)}")
            raise ProcessingError(f"Error during directory processing: {str(e)}")

    def process_direct(self) -> Dict[str, Any]:
        """
        Process the directory directly using component classes.

        This method provides access to the direct component API for advanced users.

        Returns:
            Dictionary with processing results

        Raises:
            ProcessingError: If an error occurs during processing
        """
        logger.info(f"Processing directory directly: {self.config.input}")
        try:
            from ..components.video import VideoFileIterator

            # Get all video files
            video_iterator = VideoFileIterator(
                self.config.input, self.config.ignore_files
            )
            video_files = list(video_iterator)

            results = []
            for video_file in video_files:
                # Create a new config for this video
                video_config = Config(
                    input=video_file,
                    output_dir=self.config.output_dir,
                    model=self.config.model,
                    frame_step=self.config.frame_step,
                    batch_size=self.config.batch_size,
                    save_raw_segmentation=self.config.save_raw_segmentation,
                    save_colored_segmentation=self.config.save_colored_segmentation,
                    save_overlay=self.config.save_overlay,
                    analyze_results=self.config.analyze_results,
                    visualization=self.config.visualization,
                    force_reprocess=self.config.force_reprocess,
                    disable_tqdm=self.config.disable_tqdm,
                )

                # Process the video
                processor = VideoProcessorLegacy(video_config)
                result = processor.process_direct()

                results.append({"video_path": str(video_file), "result": result})

            return {"processed_videos": results}
        except Exception as e:
            logger.exception(f"Error during direct directory processing: {str(e)}")
            raise ProcessingError(f"Error during direct directory processing: {str(e)}")


def create_processor(
    config: Config,
) -> Union[ImageProcessorLegacy, VideoProcessorLegacy, DirectoryProcessorLegacy]:
    """
    Factory function to create the appropriate processor based on input type.

    Args:
        config: Configuration for processing

    Returns:
        Appropriate processor instance

    Raises:
        ValueError: If the input type is not supported
    """
    if config.input_type == InputType.SINGLE_IMAGE:
        return ImageProcessorLegacy(config)
    elif config.input_type == InputType.SINGLE_VIDEO:
        return VideoProcessorLegacy(config)
    elif config.input_type == InputType.DIRECTORY:
        return DirectoryProcessorLegacy(config)
    else:
        raise ValueError(f"Unsupported input type: {config.input_type}")
