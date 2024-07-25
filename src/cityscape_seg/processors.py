import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Generator, Tuple, List
from contextlib import contextmanager

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .config import Config, InputType
from .models import SegmentationModelBase, create_segmentation_model
from .utils import (
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
from .exceptions import ConfigurationError, ProcessingError, ModelError, InputError

logger = logging.getLogger(__name__)


class ProcessorBase(ABC):
    """
    Abstract base class for image and video processors.

    This class defines the interface and common functionality for processors
    used in the segmentation pipeline.

    Attributes:
        model (SegmentationModelBase): The segmentation model to use for processing.
        config (Config): Configuration object containing processing parameters.
        hdf5_file: HDF5 file handle for saving segmentation maps.
        counts_file (Path): File path for saving category count data.
        percentages_file (Path): File path for saving category percentage data.
        colormap (np.ndarray): Color palette for visualization.
    """

    def __init__(self, model: SegmentationModelBase, config: Config):
        """
        Initialize the ProcessorBase.

        Args:
            model (SegmentationModelBase): The segmentation model to use.
            config (Config): Configuration object.
        """
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
        for y in range(0, height, self.model.tile_size):
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

    def initialize_hdf5(self, shape: Tuple[int, ...]):
        """
        Initialize an HDF5 file for saving segmentation maps.

        Args:
            shape (Tuple[int, ...]): Shape of the dataset to be created in the HDF5 file.
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

    @abstractmethod
    def save_results(
        self, frame: np.ndarray, seg_map: np.ndarray, frame_count: int = None
    ):
        """
        Save the processing results based on the configuration settings.

        This method should be implemented by subclasses to handle saving results
        specific to image or video processing.

        Args:
            frame (np.ndarray): Original input frame or image.
            seg_map (np.ndarray): Segmentation map.
            frame_count (int, optional): Frame number for video processing.
        """
        pass


class ImageProcessor(ProcessorBase):
    """
    Processor class for handling single image inputs.

    This class implements the processing pipeline for single image segmentation,
    including loading the image, performing segmentation, saving results,
    and generating analysis data.

    Attributes:
        Inherits all attributes from ProcessorBase.
    """

    def __init__(self, model: SegmentationModelBase, config: Config):
        """
        Initialize the ImageProcessor.

        Args:
            model (SegmentationModelBase): The segmentation model to use.
            config (Config): Configuration object.
        """
        super().__init__(model, config)
        self.logger = logging.getLogger(__name__)

    def process(self):
        """
        Process a single input image through the segmentation pipeline.

        This method handles loading the image, performing segmentation, saving results,
        and generating analysis data.
        """
        self.logger.info(f"Starting image processing for {self.config.input}")

        try:
            frame = self.load_image()
            seg_map = self.process_image(frame)
            self.save_results(frame, seg_map)
            self.analyze_results(seg_map)

            self.logger.info(
                f"Image processing complete. Output files saved with prefix: {self.config.get_output_path()}"
            )
        except IOError as e:
            raise InputError(f"Error loading input image: {str(e)}")
        except ProcessingError as e:
            raise ProcessingError(f"Error during image processing: {str(e)}")
        finally:
            self.close_hdf5()

    def load_image(self) -> np.ndarray:
        """
        Load the input image file.

        Returns:
            np.ndarray: Loaded image as a NumPy array.

        Raises:
            IOError: If there's an error reading the input image.
        """
        frame = cv2.imread(str(self.config.input))
        if frame is None:
            raise IOError(f"Error reading input image: {self.config.input}")
        return frame

    def save_results(
        self, frame: np.ndarray, seg_map: np.ndarray, frame_count: int = None
    ):
        """
        Save the processing results based on the configuration settings.

        Args:
            frame (np.ndarray): Original input image.
            seg_map (np.ndarray): Segmentation map.
            frame_count (int, optional): Not used for single image processing, included for compatibility.
        """
        output_path = self.config.get_output_path()

        if self.config.save_raw_segmentation:
            self.initialize_hdf5((1,) + seg_map.shape)
            self.save_to_hdf5(seg_map, 0)
            save_segmentation_map(seg_map, output_path)

        if self.config.save_colored_segmentation or self.config.save_overlay:
            colored_seg = self.colorize_segmentation(seg_map)

            if self.config.save_colored_segmentation:
                save_colored_segmentation(colored_seg, output_path)

            if self.config.save_overlay:
                alpha = self.config.visualization.alpha
                overlay = cv2.addWeighted(
                    frame, 1 - alpha, colored_seg.astype(np.float32), alpha, 0
                )
                save_overlay(overlay.astype(np.uint8), output_path)

    def analyze_results(self, seg_map: np.ndarray):
        """
        Analyze the segmentation results and save the analysis data.

        Args:
            seg_map (np.ndarray): Segmentation map to analyze.
        """
        analysis = analyze_segmentation_map(seg_map, self.model.num_categories)
        output_path = self.config.get_output_path()
        self.counts_file, self.percentages_file = initialize_csv_files(
            output_path, self.model.category_names
        )
        append_to_csv_files(self.counts_file, self.percentages_file, 0, analysis)
        self.logger.info(
            f"Segmentation analysis complete. Results saved to {self.counts_file} and {self.percentages_file}"
        )


class VideoProcessor(ProcessorBase):
    """
    Processor class for handling video inputs.

    This class implements the processing pipeline for video segmentation, including
    frame extraction, segmentation, analysis, and result saving.

    Attributes:
        model (SegmentationModelBase): The segmentation model to use for processing.
        config (Config): Configuration object containing processing parameters.
        video: OpenCV VideoCapture object for the input video.
        fps (int): Frames per second of the input video.
        width (int): Width of the video frames.
        height (int): Height of the video frames.
        total_frames (int): Total number of frames in the video.
        processed_frames (int): Number of frames to be processed.
        output_fps (int): Frames per second for the output video.
        fourcc: FourCC code for video codec.
        colored_output: VideoWriter for colored segmentation output.
        overlay_output: VideoWriter for overlay output.
    """

    def __init__(self, model: SegmentationModelBase, config: Config):
        """
        Initialize the VideoProcessor.

        Args:
            model (SegmentationModelBase): The segmentation model to use.
            config (Config): Configuration object.
        """
        super().__init__(model, config)
        self.video = None
        self.fps = 0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.processed_frames = 0
        self.output_fps = 0
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.colored_output = None
        self.overlay_output = None
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def managed_video_processing(self):
        """
        Context manager for video processing resources.

        This method handles the initialization and cleanup of resources used during
        video processing, ensuring proper resource management and error handling.

        Yields:
            None
        """
        try:
            self.initialize_resources()
            yield
        finally:
            self.cleanup_resources()

    def initialize_resources(self):
        """
        Initialize resources for video processing.

        This method opens the input video file, initializes video properties,
        and sets up output video writers and analysis files.
        """
        self.video = cv2.VideoCapture(str(self.config.input))
        if not self.video.isOpened():
            raise IOError(f"Error opening video file: {self.config.input}")

        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.processed_frames = self.total_frames // self.config.frame_step
        self.output_fps = max(1, self.fps // self.config.frame_step)

        output_path = self.config.get_output_path()

        if self.config.save_raw_segmentation:
            self.initialize_hdf5((self.processed_frames, self.height, self.width))

        self.counts_file, self.percentages_file = initialize_csv_files(
            output_path, self.model.category_names
        )

        if self.config.save_colored_segmentation:
            self.colored_output = cv2.VideoWriter(
                str(output_path.with_name(f"{output_path.stem}_colored.mp4")),
                self.fourcc,
                self.output_fps,
                (self.width, self.height),
            )

        if self.config.save_overlay:
            self.overlay_output = cv2.VideoWriter(
                str(output_path.with_name(f"{output_path.stem}_overlay.mp4")),
                self.fourcc,
                self.output_fps,
                (self.width, self.height),
            )

    def cleanup_resources(self):
        """
        Clean up resources used during video processing.

        This method closes video files, writers, and other opened resources.
        """
        if self.video:
            self.video.release()
        if self.colored_output:
            self.colored_output.release()
        if self.overlay_output:
            self.overlay_output.release()
        cv2.destroyAllWindows()
        self.close_hdf5()

    def frame_generator(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generate frames from the video at specified intervals.

        Yields:
            Tuple[int, np.ndarray]: A tuple containing the frame count and the frame image.
        """
        frame_count = 0
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            if frame_count % self.config.frame_step == 0:
                yield frame_count, frame
            frame_count += 1

    def process_frame(self, frame_count: int, frame: np.ndarray):
        """
        Process a single video frame.

        This method handles the segmentation, analysis, and saving of results for a single frame.

        Args:
            frame_count (int): The current frame number.
            frame (np.ndarray): The frame image to process.
        """
        seg_map = self.process_image(frame)

        if self.config.save_raw_segmentation:
            self.save_to_hdf5(seg_map, frame_count // self.config.frame_step)

        analysis = analyze_segmentation_map(seg_map, self.model.num_categories)
        append_to_csv_files(
            self.counts_file, self.percentages_file, frame_count, analysis
        )

        self.save_results(frame, seg_map, frame_count)

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
        if self.config.save_colored_segmentation or self.config.save_overlay:
            colored_seg = self.colorize_segmentation(seg_map)

            if self.config.save_colored_segmentation:
                self.colored_output.write(colored_seg.astype(np.uint8))

            if self.config.save_overlay:
                alpha = self.config.visualization.alpha
                overlay = cv2.addWeighted(
                    frame.astype(np.float64),
                    1 - alpha,
                    colored_seg.astype(np.float64),
                    alpha,
                    0,
                ).astype(np.uint8)
                self.overlay_output.write(overlay)

    def process(self):
        """
        Process the input video.

        This method orchestrates the entire video processing pipeline, including
        frame extraction, segmentation, analysis, and result saving.
        """
        self.logger.info(f"Starting video processing for {self.config.input}")

        try:
            with self.managed_video_processing():
                for frame_count, frame in tqdm(
                    self.frame_generator(),
                    total=self.processed_frames,
                    desc="Processing frames",
                ):
                    try:
                        self.process_frame(frame_count, frame)
                    except Exception as e:
                        self.logger.error(f"Error processing frame {frame_count}: {str(e)}")
                        self.logger.debug("Error details:", exc_info=True)

            self.generate_and_save_stats(self.config.get_output_path())
            self.logger.info(
                f"Video processing complete. Output files saved with prefix: {self.config.get_output_path()}"
            )
            self.logger.info(
                f"Processed {self.processed_frames} frames out of {self.total_frames} total frames."
            )
        except IOError as e:
            raise InputError(f"Error reading input video: {str(e)}")
        except ProcessingError as e:
            raise ProcessingError(f"Error during video processing: {str(e)}")

    def generate_and_save_stats(self, output_path: Path):
        """
        Generate and save category statistics for the processed video.

        This method calculates overall statistics for category counts and percentages
        and saves them to CSV files.

        Args:
            output_path (Path): Path for output statistic files.
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

        self.logger.info(
            f"Category statistics generated and saved with prefix: {output_path.stem}"
        )


class DirectoryProcessor:
    """
    Processor class for handling directory inputs containing multiple video files.

    This class manages the processing of multiple videos in a directory, creating
    a single output directory for all processed videos.

    Attributes:
        config (Config): Configuration object containing processing parameters.
        model (SegmentationModelBase): The segmentation model to use for processing.
        logger (logging.Logger): Logger for this class.
    """

    def __init__(self, config: Config):
        """
        Initialize the DirectoryProcessor.

        Args:
            config (Config): Configuration object for the processor.
        """
        self.config = config
        self.model = create_segmentation_model(config.model)
        self.logger = logging.getLogger(__name__)

    def process(self):
        """
        Process all video files in the input directory.

        This method discovers video files in the input directory and processes
        each video sequentially using the VideoProcessor.
        """
        self.logger.info(f"Starting directory processing for {self.config.input}")

        video_files = self.get_video_files()
        if not video_files:
            raise InputError(f"No video files found in directory: {self.config.input}")

        output_dir = self.config.get_output_path()
        self.logger.info(f"Output directory for all videos: {output_dir}")

        for video_file in tqdm(video_files, desc="Processing videos"):
            try:
                self.process_single_video(video_file, output_dir)
            except Exception as e:
                self.logger.error(f"Error processing video {video_file}: {str(e)}")
                self.logger.debug("Error details:", exc_info=True)

        self.logger.info(f"Finished processing all videos in {self.config.input}")

    def get_video_files(self) -> List[Path]:
        """
        Discover video files in the input directory.

        Returns:
            List[Path]: List of paths to discovered video files.
        """
        video_files = get_video_files(self.config.input)
        self.logger.info(f"Found {len(video_files)} video files in {self.config.input}")
        return video_files

    def process_single_video(self, video_file: Path, output_dir: Path):
        """
        Process a single video file.

        Args:
            video_file (Path): Path to the video file to process.
            output_dir (Path): Directory to save the processing results.
        """
        self.logger.info(f"Processing video: {video_file}")

        video_config = self.create_video_config(video_file, output_dir)

        try:
            processor = VideoProcessor(self.model, video_config)
            processor.process()
        except Exception as e:
            self.logger.error(f"Error processing video {video_file}: {str(e)}")
            self.logger.debug("Error details:", exc_info=True)

    def create_video_config(self, video_file: Path, output_dir: Path) -> Config:
        """
        Create a configuration object for processing a single video.

        Args:
            video_file (Path): Path to the video file to process.
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
            save_raw_segmentation=self.config.save_raw_segmentation,
            save_colored_segmentation=self.config.save_colored_segmentation,
            save_overlay=self.config.save_overlay,
            visualization=self.config.visualization,
        )


def create_processor(
    config: Config,
) -> Union[ImageProcessor, VideoProcessor, DirectoryProcessor]:
    """
    Factory function to create the appropriate processor based on the input file type.

    This function examines the input type specified in the configuration and creates
    the corresponding processor object. It handles single image inputs, single video
    inputs, and directory inputs containing multiple videos.

    Args:
        config (Config): Configuration object containing processing parameters.

    Returns:
        Union[ImageProcessor, VideoProcessor, DirectoryProcessor]: An instance of either
        ImageProcessor, VideoProcessor, or DirectoryProcessor, depending on the input type.

    Raises:
        ValueError: If an unsupported input type is provided in the configuration.

    Examples:
        >>> config = Config(input="path/to/image.jpg", ...)
        >>> processor = create_processor(config)
        >>> isinstance(processor, ImageProcessor)
        True

        >>> config = Config(input="path/to/video.mp4", ...)
        >>> processor = create_processor(config)
        >>> isinstance(processor, VideoProcessor)
        True

        >>> config = Config(input="path/to/video/directory", ...)
        >>> processor = create_processor(config)
        >>> isinstance(processor, DirectoryProcessor)
        True
    """
    if config.input_type == InputType.DIRECTORY:
        return DirectoryProcessor(config)
    else:
        try:
            model = create_segmentation_model(config.model)
        except ValueError as e:
            raise ModelError(f"Error creating segmentation model: {str(e)}")

        if config.input_type == InputType.SINGLE_VIDEO:
            return VideoProcessor(model, config)
        elif config.input_type == InputType.SINGLE_IMAGE:
            return ImageProcessor(model, config)
        else:
            raise ConfigurationError(f"Unsupported input type: {config.input_type}")
