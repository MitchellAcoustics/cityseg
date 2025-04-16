"""Integration tests for CitySeg using real example inputs."""

import pytest
from pathlib import Path
import tempfile

from cityseg.core.config import Config, ModelConfig, VisualizationConfig


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


@pytest.mark.skip(reason="Hamilton workflow configuration issues need to be resolved")
def test_image_processing_workflow(example_image_file, test_output_dir):
    """Test processing a real image through the CitySeg pipeline.

    This test verifies the full integration of image processing, segmentation,
    and result generation using a real example image.

    Note: Currently skipped due to Hamilton configuration issues.
    """
    pytest.importorskip("torch")  # Skip if torch not installed

    from cityseg.workflow import process

    # Configure a simple test run
    config = Config(
        input=example_image_file,
        output_dir=test_output_dir,
        output_prefix="test_image",
        model=ModelConfig(
            name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            device="cpu",
            max_size=640,  # Limit size for faster testing
            num_workers=0,  # Avoid multiprocessing in tests
            pipe_batch=1,
        ),
        frame_step=1,
        batch_size=1,
        save_raw_segmentation=True,
        save_colored_segmentation=True,
        save_overlay=True,
        analyze_results=True,
        visualization=VisualizationConfig(alpha=0.5, colormap="default"),
        force_reprocess=True,
        disable_tqdm=True,
    )

    # Process the image
    result = process(config)

    # Verify basic results
    assert result is not None
    assert "segmentation_dataset" in result
    assert "overlay_path" in result
    assert "analysis_path" in result

    # Check output files exist
    overlay_path = result["overlay_path"]
    analysis_path = result["analysis_path"]

    assert overlay_path.exists(), f"Overlay image not created: {overlay_path}"
    assert analysis_path.exists(), f"Analysis file not created: {analysis_path}"

    # Check the segmentation dataset
    seg_dataset = result["segmentation_dataset"]
    assert "segmentation" in seg_dataset.data_vars
    assert "height" in seg_dataset.coords
    assert "width" in seg_dataset.coords

    print(f"\nSuccessfully processed {example_image_file.name}")
    print(f"Output files in: {test_output_dir}")


@pytest.mark.slow
@pytest.mark.skip(reason="Hamilton workflow configuration issues need to be resolved")
def test_video_processing_workflow(example_video_file, test_output_dir):
    """Test processing a real video through the CitySeg pipeline.

    This test verifies the full integration of video processing, frame extraction,
    segmentation, and result generation using a real example video.

    Note: Currently skipped due to Hamilton configuration issues.
    """
    pytest.importorskip("torch")  # Skip if torch not installed

    from cityseg.workflow import process

    # Configure a simple test run with minimal frames for speed
    config = Config(
        input=example_video_file,
        output_dir=test_output_dir,
        output_prefix="test_video",
        model=ModelConfig(
            name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            device="cpu",
            max_size=640,  # Limit size for faster testing
            num_workers=0,  # Avoid multiprocessing in tests
            pipe_batch=1,
        ),
        frame_step=30,  # Process only every 30th frame for speed
        batch_size=1,
        save_raw_segmentation=True,
        save_colored_segmentation=True,
        save_overlay=True,
        analyze_results=True,
        visualization=VisualizationConfig(alpha=0.5, colormap="default"),
        force_reprocess=True,
        disable_tqdm=True,
    )

    # Process the video
    result = process(config)

    # Verify basic results
    assert result is not None
    assert "segmentation_dataset" in result
    assert "overlay_path" in result
    assert "analysis_path" in result

    # Check output files exist
    overlay_path = result["overlay_path"]
    analysis_path = result["analysis_path"]

    assert overlay_path.exists(), f"Overlay video not created: {overlay_path}"
    assert analysis_path.exists(), f"Analysis file not created: {analysis_path}"

    # Check the segmentation dataset
    seg_dataset = result["segmentation_dataset"]
    assert "segmentation" in seg_dataset.data_vars
    assert "frames" in seg_dataset.coords
    assert "height" in seg_dataset.coords
    assert "width" in seg_dataset.coords

    print(f"\nSuccessfully processed {example_video_file.name}")
    print(f"Output files in: {test_output_dir}")
