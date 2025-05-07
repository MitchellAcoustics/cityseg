"""Helper functions for generating test data."""

import numpy as np
import xarray as xr
from datetime import datetime
from PIL import Image
import os
import cv2
from pathlib import Path

# Import needed dependencies


def create_test_segmentation_data(
    frame_count=3,
    height=480,
    width=640,
    num_classes=19,  # typical for Cityscapes
    model_name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    pattern="grid",
):
    """
    Create a test segmentation dataset.

    Args:
        frame_count: Number of frames to generate
        height: Height of the segmentation maps
        width: Width of the segmentation maps
        num_classes: Number of segmentation classes (max value in segmentation maps)
        model_name: Name of the model to include in metadata
        pattern: Pattern to use for segmentation maps ("grid", "circle", or "random")

    Returns:
        xarray.Dataset with segmentation data and metadata
    """
    # Create deterministic segmentation maps with fixed seed
    np.random.seed(42)

    if pattern == "grid":
        # Create a grid pattern
        grid_size = max(width, height) // 10  # Size of each grid cell
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)

        # Create grid pattern
        grid_x = xx // grid_size % num_classes
        grid_y = yy // grid_size % num_classes
        seg_map = (grid_x + grid_y) % num_classes
        segmentation_data = np.array([seg_map.astype(np.int32)] * frame_count)

    elif pattern == "circle":
        # Create concentric circles
        center_x, center_y = width // 2, height // 2
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)

        # Compute distance from center
        distances = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

        # Create concentric circles
        circle_width = max(width, height) // (2 * num_classes)  # Width of each circle
        seg_map = (distances // circle_width) % num_classes
        segmentation_data = np.array([seg_map.astype(np.int32)] * frame_count)

    else:
        # Random but deterministic data
        segmentation_data = np.random.randint(
            0, num_classes, size=(frame_count, height, width), dtype=np.int32
        )

    # Create Cityscapes-like id2label and label2id dictionaries if using the standard 19 classes
    if num_classes == 19:  # Standard Cityscapes classes
        cityscapes_classes = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        id2label = {i: name for i, name in enumerate(cityscapes_classes)}
        label2id = {name: i for i, name in enumerate(cityscapes_classes)}
    else:
        # Simple id2label and label2id dictionaries
        id2label = {i: f"class_{i}" for i in range(num_classes)}
        label2id = {f"class_{i}": i for i in range(num_classes)}

    # Generate a deterministic color palette
    np.random.seed(42)
    palette = np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)

    # Create dataset with metadata
    ds = xr.Dataset(
        data_vars={
            "segmentation": (["frames", "height", "width"], segmentation_data),
        },
        coords={
            "frames": np.arange(frame_count),
            "height": np.arange(height),
            "width": np.arange(width),
        },
        attrs={
            "model_name": model_name,
            "frame_count": frame_count,
            "processing_date": datetime.now().isoformat(),
            "input_file": "test_input.mp4",
            "id2label": id2label,
            "label2id": label2id,
            "palette": palette.tolist(),
        },
    )

    return ds


def create_test_image(height=480, width=640, channels=3, pattern=None):
    """
    Create a test RGB image with data.

    Args:
        height: Height of the image
        width: Width of the image
        channels: Number of color channels (3 for RGB, 4 for RGBA)
        pattern: Optional pattern to use ("gradient", "checkerboard", or None for random)

    Returns:
        PIL.Image object
    """
    if pattern == "gradient":
        # Create a horizontal gradient
        x = np.linspace(0, 255, width)
        img_data = np.zeros((height, width, channels), dtype=np.uint8)
        for i in range(height):
            for c in range(min(3, channels)):
                img_data[i, :, c] = x * (c + 1) % 255

    elif pattern == "checkerboard":
        # Create a checkerboard pattern
        x = np.arange(width) // 50 % 2
        y = np.arange(height) // 50 % 2
        checker = np.logical_xor.outer(y, x).astype(np.uint8) * 255
        img_data = np.zeros((height, width, channels), dtype=np.uint8)
        for c in range(min(3, channels)):
            img_data[:, :, c] = checker

    else:
        # Create random RGB data with fixed seed
        np.random.seed(42)
        img_data = np.random.randint(
            0, 256, size=(height, width, channels), dtype=np.uint8
        )

    # Convert to PIL Image
    if channels == 3:
        return Image.fromarray(img_data, mode="RGB")
    elif channels == 4:
        return Image.fromarray(img_data, mode="RGBA")
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")


def save_test_images(
    output_dir, count=3, prefix="test_image", format="png", pattern=None
):
    """
    Save multiple test images to the specified directory.

    Args:
        output_dir: Directory to save images to
        count: Number of images to generate
        prefix: Filename prefix for the generated images
        format: Image format (png, jpg, etc.)
        pattern: Optional pattern to use for the images

    Returns:
        List of paths to the created images
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save test images
    image_paths = []
    for i in range(count):
        # Vary the pattern slightly for each image if using patterns
        if pattern == "gradient":
            # Shift the gradient phase for each image
            img = create_test_image(pattern="gradient")
        elif pattern == "checkerboard":
            # Invert the checkerboard for alternating images
            img = create_test_image(pattern="checkerboard")
            if i % 2 == 1:
                img = Image.fromarray(255 - np.array(img))
        else:
            img = create_test_image()

        path = os.path.join(output_dir, f"{prefix}_{i}.{format}")
        img.save(path)
        image_paths.append(path)

    return image_paths


def create_test_video(
    output_path, frame_count=30, fps=30, width=640, height=480, pattern=None
):
    """
    Create a test video file with the specified parameters.

    Args:
        output_path: Path where the video will be saved
        frame_count: Number of frames in the video
        fps: Frames per second
        width: Width of the video in pixels
        height: Height of the video in pixels
        pattern: Optional pattern to use for the frames

    Returns:
        Path to the created video file
    """
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Generate frames and add to video
    for i in range(frame_count):
        if pattern == "moving_circle":
            # Create a frame with a moving circle
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            center_x = int(
                width / 2 + (width / 4) * np.sin(i * 2 * np.pi / frame_count)
            )
            center_y = int(
                height / 2 + (height / 4) * np.cos(i * 2 * np.pi / frame_count)
            )
            cv2.circle(frame, (center_x, center_y), 50, (0, 0, 255), -1)

        elif pattern == "text":
            # Create a frame with changing text
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            text = f"Frame {i}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), font, 1, (0, 0, 0), 2)

        else:
            # Create a random frame with fixed seed for reproducibility
            np.random.seed(42 + i)
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

        # Write the frame
        out.write(frame)

    # Release the video writer
    out.release()

    return output_path
