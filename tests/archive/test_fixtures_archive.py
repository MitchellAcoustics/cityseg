"""Integration test fixture utilities."""

import pytest

from cityseg.core.config import Config, ModelConfig, VisualizationConfig


@pytest.fixture
def test_video_config(
    test_temp_dir, example_video_path, test_model_name, test_model_type
):
    """Create a complete test configuration for video processing."""
    # Configure temporary output directory
    output_dir = test_temp_dir / "video_output"
    output_dir.mkdir(exist_ok=True)

    return Config(
        input=example_video_path,
        output_dir=output_dir,
        output_prefix="test_video",
        model=ModelConfig(
            name=test_model_name,
            model_type=test_model_type,
            device="cpu",
            max_size=640,  # Smaller size for faster testing
            num_workers=0,  # Avoid multiprocessing for tests
            pipe_batch=1,
        ),
        frame_step=15,  # Process fewer frames for faster testing
        batch_size=1,  # Smaller batch size for testing
        output_fps=None,
        save_raw_segmentation=True,
        save_colored_segmentation=True,
        save_overlay=True,
        analyze_results=True,
        visualization=VisualizationConfig(alpha=0.5, colormap="default"),
        force_reprocess=True,
        disable_tqdm=True,  # Disable progress bars for cleaner test output
    )


@pytest.fixture
def test_image_config(
    test_temp_dir, example_image_path, test_model_name, test_model_type
):
    """Create a complete test configuration for image processing."""
    # Configure temporary output directory
    output_dir = test_temp_dir / "image_output"
    output_dir.mkdir(exist_ok=True)

    return Config(
        input=example_image_path,
        output_dir=output_dir,
        output_prefix="test_image",
        model=ModelConfig(
            name=test_model_name,
            model_type=test_model_type,
            device="cpu",
            max_size=640,  # Smaller size for faster testing
            num_workers=0,  # Avoid multiprocessing for tests
            pipe_batch=1,
        ),
        frame_step=1,  # Not used for images but required for Hamilton
        batch_size=1,
        output_fps=None,
        save_raw_segmentation=True,
        save_colored_segmentation=True,
        save_overlay=True,
        analyze_results=True,
        visualization=VisualizationConfig(alpha=0.5, colormap="default"),
        force_reprocess=True,
        disable_tqdm=True,  # Disable progress bars for cleaner test output
    )


@pytest.fixture
def test_directory_config(test_temp_dir, test_model_name, test_model_type):
    """Create a complete test configuration for directory processing."""
    # Create input and output directories
    input_dir = test_temp_dir / "directory_input"
    input_dir.mkdir(exist_ok=True)

    output_dir = test_temp_dir / "directory_output"
    output_dir.mkdir(exist_ok=True)

    return Config(
        input=input_dir,
        output_dir=output_dir,
        output_prefix="test_directory",
        model=ModelConfig(
            name=test_model_name,
            model_type=test_model_type,
            device="cpu",
            max_size=640,  # Smaller size for faster testing
            num_workers=0,  # Avoid multiprocessing for tests
            pipe_batch=1,
        ),
        frame_step=15,  # For video files in directory
        batch_size=1,
        output_fps=None,
        save_raw_segmentation=True,
        save_colored_segmentation=True,
        save_overlay=True,
        analyze_results=True,
        visualization=VisualizationConfig(alpha=0.5, colormap="default"),
        force_reprocess=True,
        disable_tqdm=True,  # Disable progress bars for cleaner test output
    )
