import csv
import json
from pathlib import Path
from typing import List, Tuple, Union, Optional

import cv2
import h5py
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image

from .config import Config, InputType
from .exceptions import InputError, ProcessingError
from .pipeline import create_segmentation_pipeline
from .utils import (
    analyze_segmentation_map,
    get_video_files,
)


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
                image.thumbnail(
                    (self.config.model.max_size, self.config.model.max_size)
                )

            result = self.pipeline([image])[0]
            self.save_results(image, result)
            self.analyze_results(result["seg_map"])

            self.logger.info("Image processing complete")
        except Exception as e:
            self.logger.exception(f"Error during image processing: {str(e)}")
            raise ProcessingError(f"Error during image processing: {str(e)}")

    def process_video(self):
        self.logger.info(f"Processing video: {self.config.input}")
        try:
            output_path = self.config.get_output_path()
            hdf_path = output_path.with_name(f"{output_path.stem}_segmentation.h5")

            if (
                not self.config.force_reprocess
                and hdf_path.exists()
                and self.verify_hdf_file(hdf_path)
            ):
                self.logger.info(f"Found valid existing segmentation file: {hdf_path}")
                segmentation_data, metadata = self.load_hdf_file(hdf_path)
            else:
                if hdf_path.exists():
                    if self.config.force_reprocess:
                        self.logger.info(
                            "Force reprocessing enabled. Reprocessing video."
                        )
                    else:
                        self.logger.warning(
                            f"Existing segmentation file is invalid or incomplete: {hdf_path}"
                        )
                segmentation_data, metadata = self.process_video_frames()
                self.save_hdf_file(hdf_path, segmentation_data, metadata)

            self.generate_output_videos(segmentation_data, metadata)
            self.analyze_results(segmentation_data, metadata)

            self.logger.info("Video processing complete")
        except Exception as e:
            self.logger.exception(f"Error during video processing: {str(e)}")
            raise ProcessingError(f"Error during video processing: {str(e)}")

    def verify_hdf_file(self, file_path: Path) -> bool:
        try:
            with h5py.File(file_path, "r") as f:
                if "segmentation" not in f or "metadata" not in f:
                    self.logger.warning("HDF file missing required datasets")
                    return False

                json_metadata = f["metadata"][()]
                metadata = json.loads(json_metadata)
                if "frame_count" not in metadata or "frame_step" not in metadata:
                    self.logger.warning("HDF metadata missing required fields")
                    return False

                original_video = metadata["original_video"]
                saved_frame_count = metadata["frame_count"]
                saved_frame_step = metadata["frame_step"]

                if original_video != str(self.config.input.name):
                    self.logger.warning(
                        f"Original video name mismatch: expected {self.config.input}, found {original_video}"
                    )
                    return False

                if saved_frame_step != self.config.frame_step:
                    self.logger.warning(
                        f"Frame step mismatch: expected {self.config.frame_step}, found {saved_frame_step}"
                    )
                    return False

                cap = cv2.VideoCapture(str(self.config.input))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                expected_frame_count = (
                    total_frames + self.config.frame_step - 1
                ) // self.config.frame_step

                if saved_frame_count != expected_frame_count:
                    self.logger.warning(
                        f"Frame count mismatch: expected {expected_frame_count}, found {saved_frame_count}"
                    )
                    return False

                if len(f["segmentation"]) != saved_frame_count:
                    self.logger.warning("Segmentation data length mismatch")
                    return False

                return True
        except Exception as e:
            self.logger.error(f"Error verifying HDF file: {str(e)}")
            return False

    def process_video_frames(self):
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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if self.config.save_colored_segmentation:
            colored_path = output_base.with_name(f"{output_base.stem}_colored.mp4")
            writers["colored"] = cv2.VideoWriter(
                str(colored_path), fourcc, fps, (width, height)
            )

        if self.config.save_overlay:
            overlay_path = output_base.with_name(f"{output_base.stem}_overlay.mp4")
            writers["overlay"] = cv2.VideoWriter(
                str(overlay_path), fourcc, fps, (width, height)
            )

        return writers

    def _process_batch(self, batch):
        return self.pipeline(batch)

    def _write_output_frames(self, batch, batch_results, video_writers):
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
        self, file_path: Path, segmentation_data: np.ndarray, metadata: dict
    ):
        with h5py.File(file_path, "w") as f:
            f.create_dataset("segmentation", data=segmentation_data, compression="gzip")

            # Convert all metadata to JSON-compatible format
            json_metadata = json.dumps(metadata)
            f.create_dataset("metadata", data=json_metadata)

    def load_hdf_file(self, file_path: Path) -> Tuple[np.ndarray, dict]:
        with h5py.File(file_path, "r") as f:
            segmentation_data = f["segmentation"][:]
            json_metadata = f["metadata"][()]
            metadata = json.loads(json_metadata)
        return segmentation_data, metadata

    def generate_output_videos(self, segmentation_data: np.ndarray, metadata: dict):
        output_path = self.config.get_output_path()
        fps = metadata["fps"]
        frame_step = metadata["frame_step"]

        if self.config.save_colored_segmentation:
            self.logger.info("Generating colored segmentation video")
            self._generate_video(
                segmentation_data,
                metadata,
                output_path.with_name(f"{output_path.stem}_colored.mp4"),
                colored_only=True,
                fps=fps / frame_step,
            )

        if self.config.save_overlay:
            self.logger.info("Generating overlay video")
            self._generate_video(
                segmentation_data,
                metadata,
                output_path.with_name(f"{output_path.stem}_overlay.mp4"),
                colored_only=False,
                fps=fps / frame_step,
            )

    def _generate_video(
        self,
        segmentation_data: np.ndarray,
        metadata: dict,
        output_path: Path,
        colored_only: bool,
        fps: float,
    ):
        cap = cv2.VideoCapture(str(self.config.input))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % metadata["frame_step"] == 0:
                seg_index = frame_index // metadata["frame_step"]
                if seg_index >= len(segmentation_data):
                    break

                seg_map = segmentation_data[seg_index]
                if colored_only:
                    output_frame = self.visualize_segmentation(
                        frame, seg_map, metadata["palette"], colored_only=True
                    )
                else:
                    output_frame = self.visualize_segmentation(
                        frame, seg_map, metadata["palette"], colored_only=False
                    )

                out.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

            frame_index += 1

        self.logger.debug(f"Video saved to: {output_path}")
        cap.release()
        out.release()

    def analyze_results(self, segmentation_data: np.ndarray, metadata: dict):
        output_path = self.config.get_output_path()
        counts_file = output_path.with_name(f"{output_path.stem}_category_counts.csv")
        percentages_file = output_path.with_name(
            f"{output_path.stem}_category_percentages.csv"
        )

        id2label = metadata["label_ids"]
        headers = ["Frame"] + [id2label[i] for i in sorted(id2label.keys())]

        with open(counts_file, "w", newline="") as cf, open(
            percentages_file, "w", newline=""
        ) as pf:
            counts_writer = csv.writer(cf)
            percentages_writer = csv.writer(pf)
            counts_writer.writerow(headers)
            percentages_writer.writerow(headers)

            for frame_idx, seg_map in enumerate(segmentation_data):
                analysis = analyze_segmentation_map(seg_map, len(id2label))
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

    def _generate_category_stats(self, input_file: Path, output_file: Path):
        df = pd.read_csv(input_file)
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

        pd.DataFrame(stats).to_csv(output_file, index=False)

    def visualize_segmentation(
        self,
        image: Image.Image,
        seg_map: np.ndarray,
        palette: Optional[np.ndarray],
        colored_only: bool = False,
    ) -> np.ndarray:
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
        # Lazy import of tqdm_context
        from .utils import tqdm_context

        self.logger.info(
            "Starting directory processing", input_path=str(self.config.input)
        )

        video_files = self.get_video_files()
        if not video_files:
            self.logger.error("No video files found")
            raise InputError(f"No video files found in directory: {self.config.input}")

        output_dir = self.config.get_output_path()
        self.logger.info("Output directory set", output_dir=str(output_dir))
        self.logger.info("Video files found", count=len(video_files))

        with tqdm_context(total=len(video_files), desc="Processing videos") as pbar:
            for video_file in video_files:
                try:
                    self.process_single_video(video_file, output_dir)
                    self.logger.info("Video processed", video_file=str(video_file))
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
            self.logger.error(
                "Error in video processing", video_file=str(video_file), error=str(e)
            )
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
            force_reprocess=self.config.force_reprocess,
        )


def create_processor(
    config: Config,
) -> Union[SegmentationProcessor, DirectoryProcessor]:
    if config.input_type == InputType.DIRECTORY:
        return DirectoryProcessor(config)
    else:
        return SegmentationProcessor(config)
