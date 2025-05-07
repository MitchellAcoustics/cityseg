"""Tests for input type detection in configuration."""

import pytest
import sys
import shutil
from pathlib import Path

# Add parent directory to import path
sys.path.append(str(Path(__file__).parent.parent))

from cityseg.core.config import Config, ModelConfig, InputType


@pytest.fixture
def test_image_directory(test_temp_dir, example_image_path):
    """Create a test directory with multiple images."""
    # Create a directory to hold test images
    image_dir = test_temp_dir / "test_image_directory"
    image_dir.mkdir(exist_ok=True)

    # Copy the example image multiple times with different names
    original_img = example_image_path
    for i in range(2):
        target_path = image_dir / f"test_image_{i}.png"
        shutil.copy(original_img, target_path)

    return image_dir


def test_single_image_detection(
    example_image_path, test_temp_dir, test_model_name, test_model_type
):
    """Test detection of single image input type."""
    config = Config(
        input=example_image_path,
        output_dir=test_temp_dir,
        output_prefix="test_image",
        model=ModelConfig(
            name=test_model_name,
            model_type=test_model_type,
            device="cpu",
        ),
    )

    assert config.input_type == InputType.SINGLE_IMAGE, (
        "Failed to detect single image input"
    )
    assert config.get_output_path().parent == test_temp_dir, "Incorrect output path"


def test_single_video_detection(
    example_video_path, test_temp_dir, test_model_name, test_model_type
):
    """Test detection of single video input type."""
    config = Config(
        input=example_video_path,
        output_dir=test_temp_dir,
        output_prefix="test_video",
        model=ModelConfig(
            name=test_model_name,
            model_type=test_model_type,
            device="cpu",
        ),
    )

    assert config.input_type == InputType.SINGLE_VIDEO, (
        "Failed to detect single video input"
    )
    output_path = config.get_output_path()
    assert output_path.parent == test_temp_dir, "Incorrect output path"


def test_directory_detection(
    test_image_directory, test_temp_dir, test_model_name, test_model_type
):
    """Test detection of directory input type."""
    config = Config(
        input=test_image_directory,
        output_dir=test_temp_dir,
        output_prefix="test_directory",
        model=ModelConfig(
            name=test_model_name,
            model_type=test_model_type,
            device="cpu",
        ),
    )

    assert config.input_type == InputType.DIRECTORY, "Failed to detect directory input"

    # For directory inputs, the output path should include a model-specific subdirectory
    output_path = config.get_output_path()
    assert test_temp_dir in output_path.parents, "Incorrect output path"
    assert test_model_name.split("/")[-1] in str(output_path), (
        "Model name not in output path"
    )
