"""Core segmentation functionality using xarray datasets.

This module provides the main interface for running semantic segmentation
and creating CitySeg xarray datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
import warnings

import numpy as np
import torch
import xarray as xr
from loguru import logger
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    AutoProcessor,
    BeitForSemanticSegmentation,
    Mask2FormerForUniversalSegmentation,
    MaskFormerForInstanceSegmentation,
    OneFormerForUniversalSegmentation,
    SegformerForSemanticSegmentation,
)
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel

from .components.media import MediaDataset
from .config import Config, ModelConfig


def _prepare_seg_model_args(config: Config | ModelConfig | None = None, **kwargs):
    """Prepare model arguments from config and/or kwargs."""
    if isinstance(config, ModelConfig):
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                setattr(config, key, value)
        return config
    elif isinstance(config, Config):
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                setattr(config.model, key, value)
        return config.model
    elif config is None:
        return ModelConfig(**kwargs)


def load_segmentation_model(
    config: Config | ModelConfig | None = None, device: str | None = "auto", **kwargs
) -> tuple[PreTrainedModel, BaseImageProcessor]:
    """Load a segmentation model and processor from HuggingFace.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('cpu', 'cuda', 'mps', or None for auto)

    Returns:
        Tuple of (model, image_processor)
    """
    model_config = _prepare_seg_model_args(config, device=device, **kwargs)

    model_name = model_config.name
    model_type = model_config.model_type
    device_map = "auto" if device == "auto" else model_config.device
    dataset = model_config.dataset

    # Load model and processor
    logger.info(f"Loading model {model_name} on device {device}")

    # Initialize the appropriate model and image processor based on the model name
    if "oneformer" == model_type:
        warnings.warn(
            "OneFormer models are experimental and may not be fully supported"
        )
        try:
            model = OneFormerForUniversalSegmentation.from_pretrained(
                model_name, device_map=device_map
            )
            processor = AutoProcessor.from_pretrained(model_name)
        except ValueError as e:
            raise ValueError(
                f"Failed to load OneFormer model '{model_name}': {e}"
            ) from e

    elif "mask2former" == model_type:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name, device_map=device_map
        )
        processor = AutoImageProcessor.from_pretrained(model_name)

    elif "maskformer" == model_type:
        model = MaskFormerForInstanceSegmentation.from_pretrained(
            model_name, device_map=device_map
        )
        processor = AutoImageProcessor.from_pretrained(model_name)

    elif "beit" == model_type:
        if device != "cpu":
            logger.warning(
                "Beit models are not supported on GPU and will be loaded on CPU"
            )
        device = "cpu"
        model = BeitForSemanticSegmentation.from_pretrained(
            model_name, device_map=device_map
        )
        processor = AutoImageProcessor.from_pretrained(model_name)

    elif "segformer" == model_type:
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, device_map=device_map
        )
        processor = AutoImageProcessor.from_pretrained(model_name)

        if dataset == "sidewalk-semantic":
            logger.debug("Loading Sidewalk Semantic dataset label mappings...")
            with open("SemanticSidewalk_id2label.json") as f:
                id2label = json.load(f)
            model.config.id2label = id2label
    else:
        model = AutoModelForSemanticSegmentation.from_pretrained(
            model_name, device_map=device_map
        )
        processor = AutoImageProcessor.from_pretrained(model_name)

    model.eval()

    logger.info("Model loaded successfully")
    return model, processor


def segment_image(
    image: Image.Image | np.ndarray | str | Path,
    model: PreTrainedModel | None = None,
    processor: BaseImageProcessor | None = None,
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    return_confidence: bool = False,
) -> xr.Dataset:
    """Segment a single image and return as xarray Dataset.

    Args:
        image: Input image (PIL Image, numpy array, or file path)
        model: Pre-loaded segmentation model (loads if None)
        processor: Pre-loaded image processor (loads if None)
        model_name: Model to use if model/processor not provided
        return_confidence: Whether to include confidence scores

    Returns:
        SegmentationDataset with segmentation results
    """
    # Load model if not provided
    if model is None or processor is None:
        model, processor = load_segmentation_model(model_name)

    # Load and preprocess image
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
        source_file = str(image)
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
        source_file = None
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
        source_file = None
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Process image
    inputs = processor(images=pil_image, return_tensors="pt")

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get segmentation map
    logits = outputs.logits

    # Resize to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=pil_image.size[
            ::-1
        ],  # PIL size is (width, height), torch wants (height, width)
        mode="bilinear",
        align_corners=False,
    )

    # Get predictions
    predictions = upsampled_logits.argmax(dim=1)
    segmentation_map = predictions[0].cpu().numpy().astype(np.int32)

    # Get confidence scores if requested
    confidence_scores = None
    if return_confidence:
        probabilities = torch.softmax(upsampled_logits, dim=1)
        max_probs = probabilities.max(dim=1)[0]
        confidence_scores = max_probs[0].cpu().numpy().astype(np.float32)

    # Get class labels from model config
    class_labels = {}
    if hasattr(model.config, "id2label"):
        class_labels = {int(k): v for k, v in model.config.id2label.items()}
    else:
        # Generate default labels
        num_classes = logits.shape[1]
        class_labels = {i: f"class_{i}" for i in range(num_classes)}

    # Create dataset
    ds = create_segmentation_dataset(
        segmentation_data=segmentation_map,
        model_name=model_name,
        class_labels=class_labels,
        confidence_data=confidence_scores,
        source_files=source_file,
        model_version=getattr(model.config, "model_version", None),
    )

    logger.info(f"Segmented image: {segmentation_map.shape}")
    return ds


def segment_from_path(
    input_path: str | Path,
    output_path: str | Path | None = None,
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    output_format: Literal["zarr", "netcdf", "hdf5"] = "zarr",
    return_confidence: bool = False,
) -> xr.Dataset:
    """One-shot function to segment an image/video and optionally save results.

    Args:
        input_path: Path to input image or video
        output_path: Path to save results (optional)
        model_name: HuggingFace model to use
        output_format: Format for saved results
        return_confidence: Whether to include confidence scores

    Returns:
        SegmentationDataset with results
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load model once
    model, processor = load_segmentation_model(model_name)

    # Check if input is video or image
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    if input_path.suffix.lower() in image_extensions:
        # Process single image
        ds = segment_image(
            image=input_path,
            model=model,
            processor=processor,
            model_name=model_name,
            return_confidence=return_confidence,
        )

    elif input_path.suffix.lower() in video_extensions:
        # Process video
        ds = segment_video(
            video_path=input_path,
            model=model,
            processor=processor,
            model_name=model_name,
            return_confidence=return_confidence,
        )

    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    # Save results if output path provided
    if output_path is not None:
        from .components import save_segmentation_dataset

        save_segmentation_dataset(ds, output_path, format=output_format)
        logger.info(f"Results saved to {output_path}")

    return ds


def segment_video(
    video_path: str | Path,
    model: PreTrainedModel | None = None,
    processor: BaseImageProcessor | None = None,
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    return_confidence: bool = False,
    max_frames: int | None = None,
) -> xr.Dataset:
    """Segment a video file and return as xarray Dataset.

    Args:
        video_path: Path to video file
        model: Pre-loaded segmentation model
        processor: Pre-loaded image processor
        model_name: Model to use if model/processor not provided
        return_confidence: Whether to include confidence scores
        max_frames: Maximum number of frames to process (None for all)

    Returns:
        SegmentationDataset with video segmentation results
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for video processing")

    # Load model if not provided
    if model is None or processor is None:
        model, processor = load_segmentation_model(model_name)

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    logger.info(f"Processing video: {total_frames} frames at {fps:.2f} FPS")

    # Process frames
    segmentation_maps = []
    confidence_maps = [] if return_confidence else None

    frame_idx = 0
    with torch.no_grad():
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Process frame
            inputs = processor(images=pil_image, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Run inference
            outputs = model(**inputs)
            logits = outputs.logits

            # Resize to original frame size
            h, w = frame_rgb.shape[:2]
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

            # Get predictions
            predictions = upsampled_logits.argmax(dim=1)
            seg_map = predictions[0].cpu().numpy().astype(np.int32)
            segmentation_maps.append(seg_map)

            # Get confidence if requested
            if return_confidence:
                if confidence_maps is None:
                    confidence_maps = []
                # Calculate confidence map
                probabilities = torch.softmax(upsampled_logits, dim=1)
                max_probs = probabilities.max(dim=1)[0]
                conf_map = max_probs[0].cpu().numpy().astype(np.float32)
                confidence_maps.append(conf_map)

            frame_idx += 1

            if frame_idx % 10 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()

    # Stack into arrays
    segmentation_array = np.stack(segmentation_maps)
    confidence_array = np.stack(confidence_maps) if confidence_maps else None

    # Get class labels
    class_labels = {}
    if hasattr(model.config, "id2label"):
        class_labels = {int(k): v for k, v in model.config.id2label.items()}
    else:
        num_classes = segmentation_array.max() + 1
        class_labels = {i: f"class_{i}" for i in range(num_classes)}

    # Create dataset
    ds = create_segmentation_dataset(
        segmentation_data=segmentation_array,
        model_name=model_name,
        class_labels=class_labels,
        confidence_data=confidence_array,
        source_files=str(video_path),
        model_version=getattr(model.config, "model_version", None),
        processing_metadata={"fps": fps, "total_frames": total_frames},
    )

    logger.info(f"Segmented video: {segmentation_array.shape}")
    return ds


def apply_segmentation(
    media_ds: xr.Dataset,
    model: PreTrainedModel | None = None,
    processor: BaseImageProcessor | None = None,
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    return_confidence: bool = False,
) -> xr.Dataset:
    """Apply segmentation to an existing MediaDataset.

    Args:
        media_ds: MediaDataset containing image/video data
        model: Pre-loaded segmentation model (loads if None)
        processor: Pre-loaded image processor (loads if None)
        model_name: Model to use if model/processor not provided
        return_confidence: Whether to include confidence scores

    Returns:
        Enhanced dataset with segmentation results added

    Raises:
        ValueError: If media_ds is not a valid MediaDataset
    """
    # Validate input dataset
    MediaDataset.validate_dataset(media_ds)

    # Load model if not provided
    if model is None or processor is None:
        model, processor = load_segmentation_model(model_name)

    # Get image data
    image_data = media_ds.image.values
    is_video = MediaDataset.is_video(media_ds)

    device = next(model.parameters()).device

    if is_video:
        # Process video frames
        num_frames = image_data.shape[0]
        segmentation_maps = []
        confidence_maps = [] if return_confidence else None

        logger.info(f"Processing {num_frames} video frames...")

        with torch.no_grad():
            for frame_idx in range(num_frames):
                frame = image_data[frame_idx]  # Shape: (H, W, 3)

                # Convert to PIL Image for processor
                pil_image = Image.fromarray(frame.astype(np.uint8))

                # Process frame
                inputs = processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Run inference
                outputs = model(**inputs)
                logits = outputs.logits

                # Resize to original frame size
                h, w = frame.shape[:2]
                upsampled_logits = torch.nn.functional.interpolate(
                    logits,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )

                # Get predictions
                predictions = upsampled_logits.argmax(dim=1)
                seg_map = predictions[0].cpu().numpy().astype(np.int32)
                segmentation_maps.append(seg_map)

                # Get confidence if requested
                if return_confidence:
                    if confidence_maps is None:
                        confidence_maps = []
                    probabilities = torch.softmax(upsampled_logits, dim=1)
                    max_probs = probabilities.max(dim=1)[0]
                    conf_map = max_probs[0].cpu().numpy().astype(np.float32)
                    confidence_maps.append(conf_map)

                if (frame_idx + 1) % 10 == 0:
                    logger.info(f"Processed {frame_idx + 1}/{num_frames} frames")

        # Stack arrays
        segmentation_array = np.stack(segmentation_maps)
        confidence_array = np.stack(confidence_maps) if confidence_maps else None
        seg_dims = ["time", "y", "x"]

    else:
        # Process image
        inputs = processor(images=image_data, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits

        # Resize to original image size
        h, w = image_data.shape[:2]
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        # Get predictions
        predictions = upsampled_logits.argmax(dim=1)
        segmentation_array = predictions[0].cpu().numpy().astype(np.int32)

        # Get confidence if requested
        confidence_array = None
        if return_confidence:
            probabilities = torch.softmax(upsampled_logits, dim=1)
            max_probs = probabilities.max(dim=1)[0]
            confidence_array = max_probs[0].cpu().numpy().astype(np.float32)

        seg_dims = ["y", "x"]

    # Get class labels from model config
    class_labels = {}
    if hasattr(model.config, "id2label"):
        class_labels = {int(k): v for k, v in model.config.id2label.items()}
    else:
        # Generate default labels
        unique_classes = np.unique(segmentation_array)
        class_labels = {int(cls): f"class_{cls}" for cls in unique_classes}

    # Create a copy of the media dataset to avoid modifying original
    enhanced_ds = media_ds.copy(deep=True)

    # Add class_id coordinate for segmentation data
    class_ids = list(class_labels.keys())
    if "class_id" not in enhanced_ds.coords:
        enhanced_ds = enhanced_ds.assign_coords(class_id=("class_id", class_ids))

    # Add segmentation data
    enhanced_ds["seg_map"] = (seg_dims, segmentation_array.astype(np.int32))

    # Add confidence data if requested
    if confidence_array is not None:
        enhanced_ds["confidence"] = (seg_dims, confidence_array.astype(np.float32))

    # Add class labels as data variable
    class_label_array = np.array([class_labels[cid] for cid in class_ids])
    enhanced_ds["class_label"] = (["class_id"], class_label_array)

    # Add palette as data variable (generate default colors)
    palette_array = np.zeros((len(class_ids), 3), dtype=np.uint8)

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("tab20")

    for i, cid in enumerate(class_ids):
        # Generate default color
        color = cmap(i / len(class_ids))
        r, g, b = color[:3]
        palette_array[i] = [int(r * 255), int(g * 255), int(b * 255)]

    enhanced_ds["palette"] = (["class_id", "rgb"], palette_array)

    # Update attributes
    enhanced_ds.attrs.update(
        {
            "model_name": model_name,
            "class_labels": class_labels,
            "model_version": getattr(model.config, "model_version", None),
            "has_segmentation": True,
        }
    )

    # Add processing metadata
    from datetime import datetime

    processing_info = {
        "segmentation_applied_at": datetime.now().isoformat(),
        "segmentation_model": model_name,
        "return_confidence": return_confidence,
    }

    if "processing_metadata" in enhanced_ds.attrs:
        enhanced_ds.attrs["processing_metadata"].update(processing_info)
    else:
        enhanced_ds.attrs["processing_metadata"] = processing_info

    logger.info(f"Applied segmentation to {'video' if is_video else 'image'}")
    return enhanced_ds


# Pipeline functions for convenient workflows
def segment_image_file(
    file_path: str | Path,
    model: PreTrainedModel | None = None,
    processor: BaseImageProcessor | None = None,
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    return_confidence: bool = False,
) -> xr.Dataset:
    """Complete pipeline: load image file -> apply segmentation.

    Args:
        file_path: Path to image file
        model: Pre-loaded segmentation model (loads if None)
        processor: Pre-loaded image processor (loads if None)
        model_name: Model to use if model/processor not provided
        return_confidence: Whether to include confidence scores

    Returns:
        Dataset with both image data and segmentation results
    """
    from .components.media import load_image

    # Load image as MediaDataset
    media_ds = load_image(file_path)

    # Apply segmentation
    return apply_segmentation(
        media_ds=media_ds,
        model=model,
        processor=processor,
        model_name=model_name,
        return_confidence=return_confidence,
    )


def segment_video_file(
    file_path: str | Path,
    model: PreTrainedModel | None = None,
    processor: BaseImageProcessor | None = None,
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    return_confidence: bool = False,
    max_frames: int | None = None,
) -> xr.Dataset:
    """Complete pipeline: load video file -> apply segmentation.

    Args:
        file_path: Path to video file
        model: Pre-loaded segmentation model (loads if None)
        processor: Pre-loaded image processor (loads if None)
        model_name: Model to use if model/processor not provided
        return_confidence: Whether to include confidence scores
        max_frames: Maximum number of frames to process (None for all)

    Returns:
        Dataset with both video data and segmentation results
    """
    from .components.media import load_video

    # Load video as MediaDataset
    media_ds = load_video(file_path, max_frames=max_frames)

    # Apply segmentation
    return apply_segmentation(
        media_ds=media_ds,
        model=model,
        processor=processor,
        model_name=model_name,
        return_confidence=return_confidence,
    )
