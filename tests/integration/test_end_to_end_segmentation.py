"""
End-to-end integration test for CitySeg's component-based architecture.

This test verifies the complete segmentation pipeline using the component-based
approach demonstrated in the component_demo.ipynb notebook. It tests both
image and video segmentation, including data storage and analysis.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pathlib import Path
from PIL import Image

from cityseg.components import ImageProcessor, SegmentationProcessor, VideoProcessor
from cityseg.analysis import VisualizationHandler, SegmentationAnalyzer
from cityseg.storage.storage import ZarrSegmentationStorage


# --- Test fixtures ---


@pytest.fixture
def expected_output_dir():
    """Path to the directory containing expected outputs."""
    path = Path(__file__).parent.parent / "fixtures" / "expected"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture
def generate_expected_outputs(
    example_image_path,
    example_video_path,
    expected_output_dir,
    test_model_name,
    test_model_type,
):
    """
    Generate expected outputs for testing.

    This fixture runs the segmentation pipeline on the example image and video
    and saves the outputs to the expected_output_dir for later comparison.
    """
    # Create model configuration
    from cityseg.core.config import ModelConfig

    model_config = ModelConfig(
        name=test_model_name,
        model_type=test_model_type,
        device="cpu",  # Use CPU for testing
        max_size=640,  # Resize to this maximum dimension
        num_workers=0,  # No multiprocessing for testing
    )

    # --- Process image ---

    # Load the image
    image = ImageProcessor.load_image(example_image_path)

    # Resize for processing
    resized_image = ImageProcessor.resize_image(image, model_config.max_size)

    # Create the segmentation pipeline
    segmentation_pipeline = SegmentationProcessor.create_pipeline(model_config)

    # Process the image
    result = SegmentationProcessor.process_image(resized_image, segmentation_pipeline)

    # Extract segmentation map and metadata
    seg_map = result["seg_map"]
    palette = result.get("palette")
    id2label = result.get("id2label", {})

    # Create colored segmentation map
    # Ensure seg_map is a numpy array
    seg_map_array = np.asarray(seg_map)
    colored_segmentation = VisualizationHandler.visualize_segmentation(
        np.array(resized_image),
        seg_map_array,
        palette,
        colored_only=True,  # TODO: Revisit the Palette definition and type to fix this.
    )

    # Create overlay visualization
    overlay = VisualizationHandler.visualize_segmentation(
        np.array(resized_image), seg_map_array, palette, colored_only=False, alpha=0.6
    )

    # Save visualizations
    Image.fromarray(np.array(resized_image)).save(
        expected_output_dir / "original_image.png"
    )
    Image.fromarray(colored_segmentation).save(
        expected_output_dir / "segmentation_image.png"
    )
    Image.fromarray(overlay).save(expected_output_dir / "overlay_image.png")

    # Create dataset for segmentation data
    # Ensure seg_map is a numpy array
    seg_map_array = np.asarray(seg_map)
    seg_data = xr.DataArray(
        np.expand_dims(seg_map_array, axis=0),
        dims=["time", "y", "x"],
        coords={
            "time": [0],
            "y": np.arange(seg_map_array.shape[0]),
            "x": np.arange(seg_map_array.shape[1]),
        },
    )

    # Create an xarray Dataset with the segmentation data
    dataset = xr.Dataset({"segmentation": seg_data})

    # Add metadata as attributes
    dataset.attrs = {
        "model": model_config.name,
        "model_type": model_config.model_type,
        "frame_step": 1,
        "input_file": str(example_image_path),
        "original_size": f"{image.width}x{image.height}",
        "id2label": id2label,
    }

    # Save segmentation data to Zarr
    zarr_storage = ZarrSegmentationStorage()
    zarr_path = zarr_storage.save_segmentation_data(
        dataset, dataset.attrs, expected_output_dir / "segmentation_data"
    )

    # Save analysis to Parquet
    analysis_path = SegmentationAnalyzer.analyze_segmentation_dataset(
        dataset, expected_output_dir / "segmentation_data"
    )

    # --- Process video frames ---

    # Get video metadata
    VideoProcessor.get_metadata(example_video_path)

    # Extract specific frames (just the first frame for testing)
    frame_indices = [0]
    frames = VideoProcessor.get_frames(example_video_path, frame_indices)

    # Process the frame
    if frames:
        # Resize frame
        resized_frame = ImageProcessor.resize_image(frames[0], model_config.max_size)

        # Process with segmentation model
        result = SegmentationProcessor.process_image(
            resized_frame, segmentation_pipeline
        )

        # Create overlay
        frame_overlay = VisualizationHandler.visualize_segmentation(
            np.array(resized_frame),
            result["seg_map"],
            result.get("palette"),
            colored_only=False,
            alpha=0.6,
        )

        # Create colored segmentation
        frame_segmentation = VisualizationHandler.visualize_segmentation(
            np.array(resized_frame),
            result["seg_map"],
            result.get("palette"),
            colored_only=True,
        )

        # Save frame outputs
        Image.fromarray(np.array(resized_frame)).save(
            expected_output_dir / "frame0_original.png"
        )
        Image.fromarray(frame_segmentation).save(
            expected_output_dir / "frame0_segmentation.png"
        )
        Image.fromarray(frame_overlay).save(expected_output_dir / "frame0_overlay.png")

    return {
        "image_seg_map": seg_map,
        "image_palette": palette,
        "image_id2label": id2label,
        "zarr_path": zarr_path,
        "analysis_path": analysis_path,
    }


# --- Tests ---


def test_image_segmentation_pipeline(
    example_image_path,
    expected_output_dir,
    test_model_name,
    test_model_type,
    generate_expected_outputs,
):
    """
    Test the complete image segmentation pipeline using the component-based approach.

    This test verifies:
    1. Image loading and preprocessing
    2. Segmentation pipeline creation and application
    3. Visualization generation
    4. Data storage in Zarr format
    5. Analysis generation and storage in Parquet format
    """
    # Create model configuration
    from cityseg.core.config import ModelConfig

    model_config = ModelConfig(
        name=test_model_name,
        model_type=test_model_type,
        device="cpu",  # Use CPU for testing
        max_size=640,  # Resize to this maximum dimension
        num_workers=0,  # No multiprocessing for testing
    )

    # --- Process image ---

    # Load the image
    image = ImageProcessor.load_image(example_image_path)
    assert image is not None, "Failed to load image"
    assert isinstance(image, Image.Image), "Loaded image is not a PIL Image"

    # Resize for processing
    resized_image = ImageProcessor.resize_image(image, model_config.max_size)
    assert resized_image is not None, "Failed to resize image"
    assert max(resized_image.width, resized_image.height) <= model_config.max_size, (
        "Image not resized correctly"
    )

    # Create the segmentation pipeline
    segmentation_pipeline = SegmentationProcessor.create_pipeline(model_config)
    assert segmentation_pipeline is not None, "Failed to create segmentation pipeline"

    # Process the image
    result = SegmentationProcessor.process_image(resized_image, segmentation_pipeline)
    assert result is not None, "Failed to process image"
    assert "seg_map" in result, "Segmentation result missing seg_map"

    # Extract segmentation map and metadata
    seg_map = result["seg_map"]
    palette = result.get("palette")
    id2label = result.get("id2label", {})

    # Convert seg_map to numpy array for consistent handling
    seg_map_array = np.asarray(seg_map)

    assert seg_map_array is not None, "Segmentation map is None"
    assert seg_map_array.shape[0] == resized_image.height, (
        "Segmentation map height doesn't match image"
    )
    assert seg_map_array.shape[1] == resized_image.width, (
        "Segmentation map width doesn't match image"
    )

    # Verify segmentation classes
    unique_classes = np.unique(seg_map_array)
    assert len(unique_classes) > 0, "No classes found in segmentation map"
    assert all(cls in id2label for cls in unique_classes if cls < len(id2label)), (
        "Unknown classes in segmentation map"
    )

    # Create colored segmentation map
    colored_segmentation = VisualizationHandler.visualize_segmentation(
        np.array(resized_image), seg_map_array, palette, colored_only=True
    )
    assert colored_segmentation is not None, "Failed to create colored segmentation"
    assert colored_segmentation.shape[:2] == (
        resized_image.height,
        resized_image.width,
    ), "Colored segmentation size mismatch"

    # Create overlay visualization
    overlay = VisualizationHandler.visualize_segmentation(
        np.array(resized_image), seg_map_array, palette, colored_only=False, alpha=0.6
    )
    assert overlay is not None, "Failed to create overlay"
    assert overlay.shape[:2] == (resized_image.height, resized_image.width), (
        "Overlay size mismatch"
    )

    # Compare with expected outputs
    expected_segmentation = np.array(
        Image.open(expected_output_dir / "segmentation_image.png")
    )
    expected_overlay = np.array(Image.open(expected_output_dir / "overlay_image.png"))

    # Check shapes match
    assert colored_segmentation.shape == expected_segmentation.shape, (
        "Segmentation shape mismatch"
    )
    assert overlay.shape == expected_overlay.shape, "Overlay shape mismatch"

    # Compare image content - using mean pixel difference (allows for small compression differences)
    segmentation_diff = np.abs(
        colored_segmentation.astype(np.float32)
        - expected_segmentation.astype(np.float32)
    ).mean()
    overlay_diff = np.abs(
        overlay.astype(np.float32) - expected_overlay.astype(np.float32)
    ).mean()

    # Allow for small differences due to compression/decompression
    assert segmentation_diff < 5.0, (
        f"Segmentation content differs significantly: mean diff = {segmentation_diff}"
    )
    assert overlay_diff < 5.0, (
        f"Overlay content differs significantly: mean diff = {overlay_diff}"
    )

    # Create dataset for segmentation data
    seg_data = xr.DataArray(
        np.expand_dims(seg_map, axis=0),
        dims=["time", "y", "x"],
        coords={
            "time": [0],
            "y": np.arange(seg_map.shape[0]),
            "x": np.arange(seg_map.shape[1]),
        },
    )

    # Create an xarray Dataset with the segmentation data
    dataset = xr.Dataset({"segmentation": seg_data})

    # Add metadata as attributes
    dataset.attrs = {
        "model": model_config.name,
        "model_type": model_config.model_type,
        "frame_step": 1,
        "input_file": str(example_image_path),
        "original_size": f"{image.width}x{image.height}",
        "id2label": id2label,
    }

    # Save segmentation data to Zarr
    zarr_storage = ZarrSegmentationStorage()
    zarr_path = zarr_storage.save_segmentation_data(
        dataset, dataset.attrs, expected_output_dir / "test_segmentation_data"
    )
    assert zarr_path.exists(), "Failed to save Zarr data"

    # Save analysis to Parquet
    analysis_path = SegmentationAnalyzer.analyze_segmentation_dataset(
        dataset, expected_output_dir / "test_segmentation_data"
    )
    assert analysis_path.exists(), "Failed to save analysis data"

    # Load and verify analysis data
    analysis_df = pd.read_parquet(analysis_path)
    assert not analysis_df.empty, "Analysis data is empty"
    assert "category_id" in analysis_df.columns, "Analysis missing category_id column"
    assert "pixel_count" in analysis_df.columns, "Analysis missing pixel_count column"
    assert "percentage" in analysis_df.columns, "Analysis missing percentage column"

    # Verify top categories
    top_categories = analysis_df.sort_values("percentage", ascending=False).head(3)
    assert len(top_categories) > 0, "No top categories found"

    # Compare with expected analysis
    expected_analysis_df = pd.read_parquet(generate_expected_outputs["analysis_path"])
    assert set(expected_analysis_df.columns) == set(analysis_df.columns), (
        "Analysis columns mismatch"
    )

    # Check that the same categories are present
    assert set(expected_analysis_df["category_id"]) == set(
        analysis_df["category_id"]
    ), "Category IDs mismatch"

    # Verify total pixel count matches
    assert (
        expected_analysis_df["pixel_count"].sum() == analysis_df["pixel_count"].sum()
    ), "Total pixel count mismatch"

    # Compare category percentages (allowing for small floating point differences)
    for category_id in expected_analysis_df["category_id"].unique():
        expected_pct = expected_analysis_df[
            expected_analysis_df["category_id"] == category_id
        ]["percentage"].values[0]
        actual_pct = analysis_df[analysis_df["category_id"] == category_id][
            "percentage"
        ].values[0]
        assert abs(expected_pct - actual_pct) < 0.01, (
            f"Percentage mismatch for category {category_id}: expected {expected_pct}, got {actual_pct}"
        )

    # Compare Zarr data
    expected_zarr_ds, _ = ZarrSegmentationStorage().load_segmentation_data(
        generate_expected_outputs["zarr_path"]
    )
    test_zarr_ds, _ = ZarrSegmentationStorage().load_segmentation_data(zarr_path)

    # Check that the datasets have the same structure
    assert set(expected_zarr_ds.data_vars) == set(test_zarr_ds.data_vars), (
        "Zarr dataset variables mismatch"
    )
    assert expected_zarr_ds.segmentation.shape == test_zarr_ds.segmentation.shape, (
        "Zarr segmentation shape mismatch"
    )

    # Compare segmentation data (allowing for small differences due to compression)
    seg_data_diff = np.abs(
        expected_zarr_ds.segmentation.values - test_zarr_ds.segmentation.values
    ).mean()
    assert seg_data_diff < 0.01, (
        f"Zarr segmentation data differs significantly: mean diff = {seg_data_diff}"
    )


def test_video_frame_processing(
    example_video_path, expected_output_dir, test_model_name, test_model_type
):
    """
    Test video frame extraction and processing using the component-based approach.

    This test verifies:
    1. Video metadata extraction
    2. Frame extraction
    3. Frame segmentation
    4. Visualization generation
    """
    # Create model configuration
    from cityseg.core.config import ModelConfig

    model_config = ModelConfig(
        name=test_model_name,
        model_type=test_model_type,
        device="cpu",  # Use CPU for testing
        max_size=640,  # Resize to this maximum dimension
        num_workers=0,  # No multiprocessing for testing
    )

    # Get video metadata
    metadata = VideoProcessor.get_metadata(example_video_path)
    assert metadata is not None, "Failed to get video metadata"
    assert "frame_count" in metadata, "Metadata missing frame_count"
    assert "fps" in metadata, "Metadata missing fps"
    assert "width" in metadata, "Metadata missing width"
    assert "height" in metadata, "Metadata missing height"

    # Extract specific frames (just the first frame for testing)
    frame_indices = [0]
    frames = VideoProcessor.get_frames(example_video_path, frame_indices)
    assert len(frames) == len(frame_indices), (
        f"Expected {len(frame_indices)} frames, got {len(frames)}"
    )

    # Create the segmentation pipeline
    segmentation_pipeline = SegmentationProcessor.create_pipeline(model_config)
    assert segmentation_pipeline is not None, "Failed to create segmentation pipeline"

    # Process the frame
    if frames:
        # Resize frame
        resized_frame = ImageProcessor.resize_image(frames[0], model_config.max_size)
        assert resized_frame is not None, "Failed to resize frame"
        assert (
            max(resized_frame.width, resized_frame.height) <= model_config.max_size
        ), "Frame not resized correctly"

        # Process with segmentation model
        result = SegmentationProcessor.process_image(
            resized_frame, segmentation_pipeline
        )
        assert result is not None, "Failed to process frame"
        assert "seg_map" in result, "Segmentation result missing seg_map"

        # Create overlay
        frame_overlay = VisualizationHandler.visualize_segmentation(
            np.array(resized_frame),
            result["seg_map"],
            result.get("palette"),
            colored_only=False,
            alpha=0.6,
        )
        assert frame_overlay is not None, "Failed to create frame overlay"

        # Create colored segmentation
        frame_segmentation = VisualizationHandler.visualize_segmentation(
            np.array(resized_frame),
            result["seg_map"],
            result.get("palette"),
            colored_only=True,
        )
        assert frame_segmentation is not None, "Failed to create frame segmentation"

        # Compare with expected outputs
        expected_frame_segmentation = np.array(
            Image.open(expected_output_dir / "frame0_segmentation.png")
        )
        expected_frame_overlay = np.array(
            Image.open(expected_output_dir / "frame0_overlay.png")
        )

        # Check shapes match
        assert frame_segmentation.shape == expected_frame_segmentation.shape, (
            "Frame segmentation shape mismatch"
        )
        assert frame_overlay.shape == expected_frame_overlay.shape, (
            "Frame overlay shape mismatch"
        )

        # Compare image content - using mean pixel difference (allows for small compression differences)
        frame_seg_diff = np.abs(
            frame_segmentation.astype(np.float32)
            - expected_frame_segmentation.astype(np.float32)
        ).mean()
        frame_overlay_diff = np.abs(
            frame_overlay.astype(np.float32) - expected_frame_overlay.astype(np.float32)
        ).mean()

        # Allow for small differences due to compression/decompression
        assert frame_seg_diff < 5.0, (
            f"Frame segmentation content differs significantly: mean diff = {frame_seg_diff}"
        )
        assert frame_overlay_diff < 5.0, (
            f"Frame overlay content differs significantly: mean diff = {frame_overlay_diff}"
        )

        # Compare segmentation classes distribution
        unique_classes, counts = np.unique(result["seg_map"], return_counts=True)
        class_distribution = {
            int(cls): int(count) for cls, count in zip(unique_classes, counts)
        }

        # Check that the major classes are present (at least the top 3 by pixel count)
        assert len(class_distribution) > 0, "No classes found in frame segmentation"

        # Log the class distribution for debugging
        top_classes = sorted(
            class_distribution.items(), key=lambda x: x[1], reverse=True
        )[:3]
        print(f"Top 3 classes in frame: {top_classes}")
