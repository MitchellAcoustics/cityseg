"""Direct component tests for CitySeg without using Hamilton workflow."""

import pytest
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

from cityseg.core.config import ModelConfig


@pytest.fixture
def example_image_file():
    """Path to an existing example image."""
    path = Path(
        "/Users/mitch/Documents/GitHub/cityseg/example_inputs/EustonTap-Screenshot1.png"
    )
    if not path.exists():
        pytest.skip(f"Example image not found: {path}")
    return path


@pytest.fixture
def example_video_file():
    """Path to an existing example video."""
    path = Path(
        "/Users/mitch/Documents/GitHub/cityseg/example_inputs/CaledonianPark1_15s_3840x2160.mov"
    )
    if not path.exists():
        pytest.skip(f"Example video not found: {path}")
    return path


@pytest.fixture
def test_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_image_segmentation_direct(example_image_file, test_output_dir):
    """Test direct segmentation of an image without using the Hamilton workflow.

    This test uses the individual component classes directly to verify they work.
    """
    pytest.importorskip("torch")  # Skip if torch not installed

    from cityseg.components.image import ImageProcessor
    from cityseg.components.segmentation import SegmentationProcessor
    from cityseg.analysis.visualization import VisualizationHandler

    # Load the image
    image = ImageProcessor.load_image(example_image_file)
    assert isinstance(image, Image.Image), "Failed to load image"

    # Resize for processing
    model_max_size = 640
    resized_image = ImageProcessor.resize_image(image, model_max_size)

    # Configure and load the model
    model_config = ModelConfig(
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        model_type="segformer",
        device="cpu",
        num_workers=0,
    )

    # Create the segmentation pipeline
    segmentation_pipeline = SegmentationProcessor.create_pipeline(model_config)
    assert segmentation_pipeline is not None, "Failed to create segmentation pipeline"

    # Process the image
    segmentation_result = SegmentationProcessor.process_image(
        resized_image, segmentation_pipeline
    )
    assert segmentation_result is not None, "Failed to process image"
    assert "seg_map" in segmentation_result, "No segmentation map in result"

    # Extract segmentation map
    seg_map = segmentation_result["seg_map"]
    assert isinstance(seg_map, np.ndarray), "Segmentation map is not a numpy array"

    # Get the palette
    palette = segmentation_result.get("palette")

    # Create an overlay visualization
    image_array = np.array(resized_image)
    overlay = VisualizationHandler.visualize_segmentation(
        image_array, seg_map, palette, colored_only=False, alpha=0.5
    )
    assert isinstance(overlay, np.ndarray), "Failed to create overlay"

    # Save the overlay as an image
    overlay_image = Image.fromarray(overlay)
    output_path = test_output_dir / "test_overlay.png"
    overlay_image.save(output_path)

    assert output_path.exists(), f"Failed to save overlay to {output_path}"

    # Print some information about the segmentation
    unique_classes = np.unique(seg_map)
    class_names = [
        segmentation_result["id2label"].get(int(cls), f"Unknown-{cls}")
        for cls in unique_classes
    ]

    print(f"\nSuccessfully processed {example_image_file.name}")
    print(f"Image size: {image.size}, Resized: {resized_image.size}")
    print(f"Segmentation map shape: {seg_map.shape}")
    print(f"Classes found: {class_names}")
    print(f"Overlay saved to: {output_path}")


@pytest.mark.slow
def test_video_frame_extraction(example_video_file, test_output_dir):
    """Test extracting and processing frames from a video file."""
    pytest.importorskip("torch")  # Skip if torch not installed

    from cityseg.components.video import VideoProcessor
    from cityseg.components.segmentation import SegmentationProcessor
    from cityseg.analysis.visualization import VisualizationHandler

    # Get video metadata
    metadata = VideoProcessor.get_metadata(example_video_file)
    assert metadata is not None, "Failed to get video metadata"

    # Print out basic metadata
    duration = metadata.get("duration", metadata["frame_count"] / metadata["fps"])
    print(
        f"\nVideo metadata: {metadata['width']}x{metadata['height']}, "
        f"{metadata['frame_count']} frames, {metadata['fps']} fps, "
        f"{duration:.2f}s long"
    )

    # Extract a small number of frames for processing speed
    frame_indices = [0, 30, 60]  # Just 3 frames
    frames = VideoProcessor.get_frames(example_video_file, frame_indices)

    assert len(frames) == len(frame_indices), (
        "Failed to extract expected number of frames"
    )
    assert all(isinstance(f, Image.Image) for f in frames), (
        "Extracted frames are not PIL images"
    )

    # Resize frames
    resized_frames = [f.resize((640, 360)) for f in frames]

    # Configure and load the model
    model_config = ModelConfig(
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        model_type="segformer",
        device="cpu",
        num_workers=0,
    )

    # Create the segmentation pipeline
    segmentation_pipeline = SegmentationProcessor.create_pipeline(model_config)

    # Process the frames in batch
    results = SegmentationProcessor.process_batch(resized_frames, segmentation_pipeline)
    assert len(results) == len(frames), "Failed to process all frames"

    # Extract segmentation maps
    seg_maps = [result["seg_map"] for result in results]
    palette = results[0].get("palette")  # Use the palette from the first result

    # Create visualizations
    frame_arrays = [np.array(frame) for frame in resized_frames]
    overlays = VisualizationHandler.visualize_segmentation(
        frame_arrays, seg_maps, palette, colored_only=False, alpha=0.5
    )

    # Save the overlays as images
    for i, overlay in enumerate(overlays):
        output_path = test_output_dir / f"frame_{i}_overlay.png"
        Image.fromarray(overlay).save(output_path)
        assert output_path.exists(), f"Failed to save overlay to {output_path}"

    # Get class counts for first frame to display
    seg_map = seg_maps[0]
    unique_classes, counts = np.unique(seg_map, return_counts=True)
    class_names = [
        results[0]["id2label"].get(int(cls), f"Unknown-{cls}") for cls in unique_classes
    ]

    # Get the class counts as percentage
    total_pixels = seg_map.size
    percentages = [count / total_pixels * 100 for count in counts]

    # Print class information
    print("\nClass distribution in first frame:")
    for cls, name, percent in zip(unique_classes, class_names, percentages):
        print(f"  {name} (ID {cls}): {percent:.1f}%")

    print(f"\nFrames extracted and processed: {len(frames)}")
    print(f"Overlays saved to: {test_output_dir}")
