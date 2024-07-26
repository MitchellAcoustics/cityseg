# %%

from typing import Dict, Any, Union, List
import numpy as np
from transformers import ImageSegmentationPipeline
from PIL import Image
from cityscape_seg.palettes import ADE20K_PALETTE
import cv2
from tqdm.auto import tqdm

class SingleMapImageSegmentationPipeline(ImageSegmentationPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.palette = self._get_palette()

    def _get_palette(self):
        # Try to get the palette from the model's config
        if hasattr(self.model.config, 'palette'):
            return np.array(self.model.config.palette)
        elif 'ade' in self.model.config._name_or_path:
            return np.array(ADE20K_PALETTE)
        # If not found, try to generate a palette based on the number of labels
        elif hasattr(self.model.config, 'num_labels'):
            num_labels = self.model.config.num_labels
            return self._generate_palette(num_labels)

        # If still not found, fall back to a default palette or raise an error
        else:
            print("Warning: Unable to determine palette. Using a default palette.")
            return self._generate_palette(256)  # Generate a default palette with 256 colors

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
        """
        Check if the result is for a single image or multiple images.

        Returns:
        - True if it's a list of annotations for a single image
        - False if it's a list of lists of annotations for multiple images
        """
        if not result:
            return True  # Empty result, assume single image
        if isinstance(result[0], dict) and 'mask' in result[0]:
            return True  # List of annotations for a single image
        if isinstance(result[0], list) and result[0] and isinstance(result[0][0], dict) and 'mask' in result[0][0]:
            return False  # List of lists of annotations for multiple images
        raise ValueError("Unexpected result structure")

    def __call__(self, images, **kwargs):
        result = super().__call__(images, **kwargs)

        if self._is_single_image_result(result):
            # Single image case
            return self.create_single_segmentation_map(result, result[0]['mask'].size[::-1])
        else:
            # Multiple images case
            return [self.create_single_segmentation_map(img_result, img_result[0]['mask'].size[::-1]) for img_result in result]

# %%
class VideoSegmentationPipeline(SingleMapImageSegmentationPipeline):
    def process_video(self, video_path, output_path=None, frame_interval=1, batch_size=16, show_progress=True):
        """
        Process a video file in batches and return segmentation maps for each frame.

        Args:
        video_path (str): Path to the input video file.
        output_path (str, optional): Path to save the output video with segmentation overlay.
        frame_interval (int): Process every nth frame.
        batch_size (int): Number of frames to process in each batch.
        show_progress (bool): Whether to show a progress bar.

        Returns:
        List of segmentation maps, one for each processed frame.
        """
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

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                batch_frames.append(frame)
                batch_images.append(image)

            # Process the batch
            batch_results = self(batch_images)
            segmentation_maps.extend(batch_results)

            if output_path:
                for frame, result in zip(batch_frames, batch_results):
                    # Visualize segmentation and save to output video
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


if __name__ == "__main__":
    # %%
    from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
    import matplotlib.pyplot as plt

    # Load model and image processor
    # model = AutoModelForSemanticSegmentation.from_pretrained("microsoft/beit-large-finetuned-ade-640-640")
    # image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-large-finetuned-ade-640-640")

    # Create the pipeline
    # pipe = SingleMapImageSegmentationPipeline(model=model, image_processor=image_processor)
    #
    # # Load an image
    # image_path = "/Users/mitch/Documents/GitHub/cityscape-seg/example_inputs/EustonTap-Screenshot1.png"
    # image = Image.open(image_path)
    # image = image.convert("RGB")

    # # Run the pipeline
    # result = pipe(image)

    # %%
    # image2_path = "/Users/mitch/Documents/GitHub/cityscape-seg/example_inputs/EustonTap-Screenshot2.png"
    # image2 = Image.open(image2_path)
    # image2 = image2.convert("RGB")
    # multi_res = pipe([image, image2])

    # %%

    # Video processing example
    # image_vis = pipe.visualize_segmentation(image, result['seg_map'])
    # plt.imshow(image_vis)
    # plt.axis('off')
    # plt.show()
    #
    # image2_vis = pipe.visualize_segmentation(image2, multi_res[1]['seg_map'])
    # plt.imshow(image2_vis)
    # plt.axis('off')
    # plt.show()

    from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor, Mask2FormerForUniversalSegmentation
    import torch

    # model = AutoModelForSemanticSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    # image_processor = AutoImageProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
            # "facebook/mask2former-swin-large-ade-semantic",
            "facebook/mask2former-swin-large-mapillary-vistas-semantic",
            )

    image_processor = AutoImageProcessor.from_pretrained(
            # "facebook/mask2former-swin-large-ade-semantic",
            "facebook/mask2former-swin-large-mapillary-vistas-semantic",
            )

    pipe = VideoSegmentationPipeline(model=model, image_processor=image_processor, device=torch.device('mps'),
                                     subtask='semantic')

    video_path = "/Users/mitch/Documents/GitHub/cityscape-seg/example_inputs/Carlov2_15s_3840x2160.mov"
    output_path = "/Users/mitch/Documents/GitHub/cityscape-seg/example_inputs/output_Carlov2_mask2former_mapillary-vistas.mp4"

    segmentation_maps = pipe.process_video(video_path, output_path, frame_interval=1, batch_size=10, show_progress=True)
    print(f"Processed {len(segmentation_maps)} frames.")