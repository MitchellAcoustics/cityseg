"""Core segmentation functionality using xarray datasets.

This module provides the main interface for running semantic segmentation
and creating CitySeg xarray datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import xarray as xr
from loguru import logger
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel

from .components import create_segmentation_dataset


def load_segmentation_model(
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    device: str | None = None,
) -> tuple[PreTrainedModel, BaseImageProcessor]:
    """Load a segmentation model and processor from HuggingFace.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('cpu', 'cuda', 'mps', or None for auto)

    Returns:
        Tuple of (model, image_processor)
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info(f"Loading model {model_name} on device {device}")

    # Load model and processor
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)

    # Move model to device
    model.to(device)
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
