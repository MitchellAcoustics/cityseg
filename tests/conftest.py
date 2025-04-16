import pytest
import os
import tempfile
from pathlib import Path

from cityseg.core.config import Config, ModelConfig, VisualizationConfig


# --- Path fixtures ---

@pytest.fixture
def example_video_path():
    """Path to example video file."""
    return Path(
        "/Users/mitch/Documents/GitHub/cityseg/example_inputs/CaledonianPark1_15s_3840x2160.mov"
    )


@pytest.fixture
def example_image_path():
    """Path to example image file."""
    return Path(
        "/Users/mitch/Documents/GitHub/cityseg/example_inputs/EustonTap-Screenshot1.png"
    )


@pytest.fixture
def test_fixture_image_path():
    """Path to test image in fixtures directory."""
    path = Path(__file__).parent / "fixtures" / "images" / "test_image.png"
    if not path.exists():
        pytest.skip(f"Test fixture image not found: {path}")
    return path


@pytest.fixture
def test_fixture_video_path():
    """Path to test video in fixtures directory."""
    path = Path(__file__).parent / "fixtures" / "videos" / "test_video.mp4"
    if not path.exists():
        pytest.skip(f"Test fixture video not found: {path}")
    return path


# --- Directory fixtures ---

@pytest.fixture
def test_temp_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def test_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# --- Model fixtures ---

@pytest.fixture(scope="session")
def test_model_name():
    """Return the model name to use for testing."""
    return "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


@pytest.fixture(scope="session")
def test_model_type():
    """Return the model type to use for testing."""
    return "segformer"


# --- Configuration fixtures ---

@pytest.fixture
def test_video_config(test_temp_dir, example_video_path, test_model_name, test_model_type):
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
def test_image_config(test_temp_dir, example_image_path, test_model_name, test_model_type):
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
