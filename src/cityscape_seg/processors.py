import logging
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from loguru import logger
from tqdm.auto import tqdm
from PIL import Image

from .config import Config, InputType
from .exceptions import InputError, ProcessingError
from .pipeline import create_segmentation_pipeline
from .utils import (
    analyze_segmentation_map,
    append_to_csv_files,
    generate_category_stats,
    get_video_files,
    initialize_csv_files,
    save_colored_segmentation,
    save_overlay,
    save_segmentation_map,
)

class TqdmCompatibleSink:
    def __init__(self, level=logging.INFO):
        self.level = level

    def write(self, message):
        tqdm.write(message, end="")

logger.remove()
logger.add(TqdmCompatibleSink(), format="{time} | {level} | {message}", level="INFO")
logger.add("file_{time}.log", rotation="1 day")

class SegmentationProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = create_segmentation_pipeline(
            model_name=config.model.name,
            device=config.model.device,
        )
        self.logger = logger.bind(
            processor_type=self.__class__.__name__,
            input_type=self.config.input_type.value,
        )

    def process(self):
        if self.config.input_type == InputType.SINGLE_IMAGE:
            self.process_image()
        elif self.config.input_type == InputType.SINGLE_VIDEO:
            self.process_video()
        else:
            raise ValueError(f"Unsupported input type: {self.config.input_type}")

    def process_image(self):
        self.logger.info(f"Processing image: {self.config.input}")
        try:
            image = Image.open(self.config.input).convert("RGB")
            if self.config.model.max_size:
                image.thumbnail((self.config.model.max_size, self.config.model.max_size))

            result = self.pipeline(image)
            self.save_results(image, result)
            self.analyze_results(result['seg_map'])

            self.logger.info("Image processing complete")
        except Exception as e:
            self.logger.exception(f"Error during image processing: {str(e)}")
            raise ProcessingError(f"Error during image processing: {str(e)}")

    def process_video(self):
        self.logger.info(f"Processing video: {self.config.input}")
        try:
            output_path = self.config.get_output_path()
            results = self.pipeline.process_video(
                str(self.config.input),
                str(output_path) if self.config.save_colored_segmentation else None,
                frame_interval=self.config.frame_step,
                show_progress=True
            )

            self.save_video_results(results)
            self.analyze_video_results(results)

            self.logger.info("Video processing complete")
        except Exception as e:
            self.logger.exception(f"Error during video processing: {str(e)}")
            raise ProcessingError(f"Error during video processing: {str(e)}")

    def save_results(self, image: Image.Image, result: dict):
        output_path = self.config.get_output_path()

        if self.config.save_raw_segmentation:
            save_segmentation_map(result['seg_map'], output_path)

        if self.config.save_colored_segmentation:
            colored_seg = self.pipeline.visualize_segmentation(image, result['seg_map'])
            save_colored_segmentation(colored_seg, output_path)

        if self.config.save_overlay:
            overlay = self.pipeline.visualize_segmentation(image, result['seg_map'])
            save_overlay(overlay, output_path)

    def save_video_results(self, results: List[dict]):
        output_path = self.config.get_output_path()

        if self.config.save_raw_segmentation:
            for i, result in enumerate(results):
                save_segmentation_map(result['seg_map'], output_path, frame_count=i)

        # Note: Colored segmentation is already saved during video processing if enabled

        if self.config.save_overlay and not self.config.save_colored_segmentation:
            cap = cv2.VideoCapture(str(self.config.input))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path.with_name(f"{output_path.stem}_overlay.mp4")),
                fourcc,
                cap.get(cv2.CAP_PROP_FPS) // self.config.frame_step,
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )

            for i, result in enumerate(results):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * self.config.frame_step)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                overlay = self.pipeline.visualize_segmentation(image, result['seg_map'])
                out.write(cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR))

            cap.release()
            out.release()

    def analyze_results(self, seg_map: np.ndarray):
        analysis = analyze_segmentation_map(seg_map, len(self.pipeline.model.config.id2label))
        output_path = self.config.get_output_path()
        counts_file, percentages_file = initialize_csv_files(
            output_path, self.pipeline.model.config.id2label
        )
        append_to_csv_files(counts_file, percentages_file, 0, analysis)

    def analyze_video_results(self, results: List[dict]):
        output_path = self.config.get_output_path()
        counts_file, percentages_file = initialize_csv_files(
            output_path, self.pipeline.model.config.id2label
        )

        for i, result in enumerate(results):
            analysis = analyze_segmentation_map(result['seg_map'], len(self.pipeline.model.config.id2label))
            append_to_csv_files(counts_file, percentages_file, i * self.config.frame_step, analysis)

        counts_stats = generate_category_stats(counts_file)
        counts_stats.to_csv(output_path.with_name(f"{output_path.stem}_counts_stats.csv"), index=False)

        percentages_stats = generate_category_stats(percentages_file)
        percentages_stats.to_csv(output_path.with_name(f"{output_path.stem}_percentages_stats.csv"), index=False)

class DirectoryProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logger.bind(
            processor_type=self.__class__.__name__,
            input_type=self.config.input_type.value,
            input_path=str(self.config.input),
            output_path=str(self.config.get_output_path()),
            frame_step=self.config.frame_step,
        )

    def process(self):
        self.logger.info("Starting directory processing", input_path=str(self.config.input))

        video_files = self.get_video_files()
        if not video_files:
            self.logger.error("No video files found")
            raise InputError(f"No video files found in directory: {self.config.input}")

        output_dir = self.config.get_output_path()
        self.logger.info("Output directory set", output_dir=str(output_dir))
        self.logger.info("Video files found", count=len(video_files))

        for video_file in tqdm(video_files, desc="Processing videos"):
            try:
                self.process_single_video(video_file, output_dir)
                self.logger.info("Video processed", video_file=str(video_file))
            except Exception as e:
                self.logger.error("Error processing video", video_file=str(video_file), error=str(e))
                self.logger.debug("Error details", exc_info=True)

        self.logger.info("Finished processing all videos", input_directory=str(self.config.input))

    def get_video_files(self) -> List[Path]:
        video_files = get_video_files(self.config.input)
        self.logger.info(f"Found {len(video_files)} video files in {self.config.input}")
        return video_files

    def process_single_video(self, video_file: Path, output_dir: Path):
        self.logger.info("Processing video", video_file=str(video_file))

        video_config = self.create_video_config(video_file, output_dir)

        try:
            processor = SegmentationProcessor(video_config)
            processor.process()
        except Exception as e:
            self.logger.error("Error in video processing", video_file=str(video_file), error=str(e))
            raise ProcessingError(f"Error processing video {video_file}: {str(e)}")

    def create_video_config(self, video_file: Path, output_dir: Path) -> Config:
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

def create_processor(config: Config) -> Union[SegmentationProcessor, DirectoryProcessor]:
    if config.input_type == InputType.DIRECTORY:
        return DirectoryProcessor(config)
    else:
        return SegmentationProcessor(config)