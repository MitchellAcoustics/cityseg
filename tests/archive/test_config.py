"""Tests for configuration objects used for integration tests."""

import sys
from pathlib import Path

# Add parent directory to import path
sys.path.append(str(Path(__file__).parent.parent))

from cityseg.core.config import Config, ModelConfig, InputType


def test_config_loading(
    example_image_path,
    example_video_path,
    test_temp_dir,
    test_model_name,
    test_model_type,
):
    """Test that the test configurations are properly loaded."""
    # Test image configuration
    config = Config(
        input=example_image_path,
        output_dir=test_temp_dir,
        output_prefix="test_image",
        model=ModelConfig(
            name=test_model_name,
            model_type=test_model_type,
            device="cpu",
            max_size=640,
            num_workers=0,
        ),
        frame_step=1,
        batch_size=1,
    )

    assert config.input_type == InputType.SINGLE_IMAGE, "Wrong input type detected"
    assert config.model.name == test_model_name, "Model name mismatch"
    assert config.model.model_type == test_model_type, "Model type mismatch"

    # Test video configuration
    config = Config(
        input=example_video_path,
        output_dir=test_temp_dir,
        output_prefix="test_video",
        model=ModelConfig(
            name=test_model_name,
            model_type=test_model_type,
            device="cpu",
            max_size=640,
            num_workers=0,
        ),
        frame_step=15,
        batch_size=1,
    )

    assert config.input_type == InputType.SINGLE_VIDEO, "Wrong input type detected"
    assert config.frame_step == 15, "Frame step mismatch"

    # Ensure get_output_path works
    output_path = config.get_output_path()
    assert output_path.parent == test_temp_dir, "Output path mismatch"
