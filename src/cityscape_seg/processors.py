import logging
from pathlib import Path
from typing import List, Union, Optional

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

            result = self.pipeline([image])[0]
            self.save_results(image, result)
            self.analyze_results(result['seg_map'])

            self.logger.info("Image processing complete")
        except Exception as e:
            self.logger.exception(f"Error during image processing: {str(e)}")
            raise ProcessingError(f"Error during image processing: {str(e)}")

    def process_video(self):
        self.logger.info(f"Processing video: {self.config.input}")
        try:
            cap, frame_count, original_fps, width, height = self._initialize_video_capture()
            output_fps = self.config.output_fps or int(original_fps / self.config.frame_step)
            video_writers = self._initialize_video_writers(width, height, output_fps)

            results = []
            for batch in tqdm(self._frame_generator(cap), total=frame_count // self.config.frame_step):
                batch_results = self._process_batch(batch)
                results.extend(batch_results)
                self._write_output_frames(batch, batch_results, video_writers)

            cap.release()
            for writer in video_writers.values():
                writer.release()

            self.save_video_results(results)
            self.analyze_video_results(results)

            self.logger.info("Video processing complete")
        except Exception as e:
            self.logger.exception(f"Error during video processing: {str(e)}")
            raise ProcessingError(f"Error during video processing: {str(e)}")

    def _initialize_video_capture(self):
        cap = cv2.VideoCapture(str(self.config.input))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap, frame_count, original_fps, width, height

    def _initialize_video_writers(self, width, height, fps):
        writers = {}
        output_base = self.config.get_output_path()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if self.config.save_colored_segmentation:
            colored_path = output_base.with_name(f"{output_base.stem}_colored.mp4")
            writers['colored'] = cv2.VideoWriter(str(colored_path), fourcc, fps, (width, height))

        if self.config.save_overlay:
            overlay_path = output_base.with_name(f"{output_base.stem}_overlay.mp4")
            writers['overlay'] = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))

        return writers

    def _process_batch(self, batch):
        return self.pipeline(batch)

    def _write_output_frames(self, batch, batch_results, video_writers):
        for pil_image, result in zip(batch, batch_results):
            if 'colored' in video_writers:
                colored_seg = self.visualize_segmentation(pil_image, result['seg_map'], result['palette'], colored_only=True)
                video_writers['colored'].write(cv2.cvtColor(np.array(colored_seg), cv2.COLOR_RGB2BGR))
            if 'overlay' in video_writers:
                overlay = self.visualize_segmentation(pil_image, result['seg_map'], result['palette'], colored_only=False)
                video_writers['overlay'].write(cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR))

    def visualize_segmentation(self, image: Image.Image, seg_map: np.ndarray, palette: Optional[np.ndarray], colored_only: bool = False) -> np.ndarray:
        if palette is None:
            palette = self._generate_palette(len(np.unique(seg_map)))

        image_array = np.array(image)
        color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for label_id, color in enumerate(palette):
            color_seg[seg_map == label_id] = color

        if colored_only:
            return color_seg
        else:
            img = image_array * 0.5 + color_seg * 0.5
            return img.astype(np.uint8)

    def _frame_generator(self, cap):
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


    def _generate_palette(self, num_colors):
        def _generate_color(i):
            r = int((i * 100) % 255)
            g = int((i * 150) % 255)
            b = int((i * 200) % 255)
            return [r, g, b]

        return np.array([_generate_color(i) for i in range(num_colors)])

    def save_results(self, image: Image.Image, result: dict):
        output_path = self.config.get_output_path()

        if self.config.save_raw_segmentation:
            save_segmentation_map(result['seg_map'], output_path)

        if self.config.save_colored_segmentation:
            colored_seg = self.visualize_segmentation(image, result['seg_map'], result['palette'])
            save_colored_segmentation(colored_seg, output_path)

        if self.config.save_overlay:
            overlay = self.visualize_segmentation(image, result['seg_map'], result['palette'])
            save_overlay(overlay, output_path)

    def save_video_results(self, results: List[dict]):
        output_path = self.config.get_output_path()

        if self.config.save_raw_segmentation:
            for i, result in enumerate(results):
                save_segmentation_map(result['seg_map'], output_path, frame_count=i)

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