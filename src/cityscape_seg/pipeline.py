from typing import Dict, Any, Union, List
import numpy as np
from transformers import ImageSegmentationPipeline, AutoModelForSemanticSegmentation, AutoImageProcessor
from transformers import OneFormerProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import cv2
from tqdm.auto import tqdm
import torch


class SegmentationPipeline(ImageSegmentationPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.palette = self._get_palette()

    def _get_palette(self):
        if hasattr(self.model.config, 'palette'):
            return np.array(self.model.config.palette)
        elif 'ade' in self.model.config._name_or_path:
            from .palettes import ADE20K_PALETTE
            return np.array(ADE20K_PALETTE)
        else:
            return self._generate_palette(self.model.config.num_labels)

    def _generate_palette(self, num_colors):
        def _generate_color(i):
            r = int((i * 100) % 255)
            g = int((i * 150) % 255)
            b = int((i * 200) % 255)
            return [r, g, b]

        return np.array([_generate_color(i) for i in range(num_colors)])

    def create_single_segmentation_map(self, annotations, target_size):
        seg_map = np.zeros(target_size, dtype=np.int32)
        for annotation in annotations:
            mask = np.array(annotation['mask'])
            label_id = self.model.config.label2id[annotation['label']]
            seg_map[mask != 0] = label_id

        return {
            'seg_map' : seg_map,
            'label2id': self.model.config.label2id,
            'id2label': self.model.config.id2label,
            'palette' : self.palette
            }

    def visualize_segmentation(self, image: Image.Image, seg_map: np.ndarray) -> np.ndarray:
        image_array = np.array(image)
        color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for label_id, color in enumerate(self.palette):
            color_seg[seg_map == label_id] = color

        img = image_array * 0.5 + color_seg * 0.5
        return img.astype(np.uint8)

    def _is_single_image_result(self, result):
        if not result:
            return True
        if isinstance(result[0], dict) and 'mask' in result[0]:
            return True
        if isinstance(result[0], list) and result[0] and isinstance(result[0][0], dict) and 'mask' in result[0][0]:
            return False
        raise ValueError("Unexpected result structure")

    def __call__(self, images, **kwargs):
        result = super().__call__(images, **kwargs)

        if self._is_single_image_result(result):
            return self.create_single_segmentation_map(result, result[0]['mask'].size[::-1])
        else:
            return [self.create_single_segmentation_map(img_result, img_result[0]['mask'].size[::-1]) for img_result in
                    result]

    def process_video(self, video_path, output_path=None, frame_interval=1, batch_size=16, show_progress=True):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        segmentation_maps = []

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps,
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                                  )

        if batch_size == -1:
            batch_size = int(frame_count // frame_interval)
        total_batches = (frame_count + frame_interval * batch_size - 1) // (frame_interval * batch_size)

        if show_progress:
            pbar = tqdm(total=total_batches, desc="Processing video batches")

        for batch_start in range(0, frame_count, frame_interval * batch_size):
            batch_frames = []
            batch_images = []

            for i in range(batch_size):
                frame_num = batch_start + i * frame_interval
                if frame_num >= frame_count:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                batch_frames.append(frame)
                batch_images.append(image)

            batch_results = self(batch_images)
            segmentation_maps.extend(batch_results)

            if output_path:
                for frame, result in zip(batch_frames, batch_results):
                    vis_frame = self.visualize_segmentation(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                                            result['seg_map']
                                                            )
                    out.write(cv2.cvtColor(np.array(vis_frame), cv2.COLOR_RGB2BGR))

            if show_progress:
                pbar.update(1)

        cap.release()
        if output_path:
            out.release()

        if show_progress:
            pbar.close()

        return segmentation_maps


def create_segmentation_pipeline(model_name: str, device: str = None, **kwargs):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'oneformer' in model_name.lower():
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        image_processor = OneFormerProcessor.from_pretrained(model_name)
    elif 'mask2former' in model_name.lower():
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        image_processor = AutoImageProcessor.from_pretrained(model_name)
    else:
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        image_processor = AutoImageProcessor.from_pretrained(model_name)

    return SegmentationPipeline(model=model, image_processor=image_processor, device=device, subtask='semantic', **kwargs)