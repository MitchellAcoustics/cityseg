"""Integration tests for directory processing workflow."""

import pytest
import os
import shutil
from pathlib import Path
import sys

# Add parent directory to import path
sys.path.append(str(Path(__file__).parent.parent))

from cityseg.workflow.hamilton import process

# Import test fixtures


@pytest.fixture
def test_image_directory(test_temp_dir, example_image_path):
    """Create a test directory with multiple images."""
    # Create a directory to hold test images
    image_dir = test_temp_dir / "image_inputs"
    image_dir.mkdir(exist_ok=True)

    # Copy the example image multiple times with different names
    original_img = example_image_path
    for i in range(2):  # Just copy twice for faster testing
        target_path = image_dir / f"test_image_{i}.png"
        shutil.copy(original_img, target_path)

    return image_dir


@pytest.fixture
def populated_directory_config(test_directory_config, test_image_directory):
    """Create a populated test configuration for directory processing."""
    # Update the config to use our populated directory
    test_directory_config.input = test_image_directory
    return test_directory_config


@pytest.mark.skip(reason="Needs complete Hamilton workflow implementation")
@pytest.mark.slow
def test_directory_image_workflow(populated_directory_config):
    """Test processing a directory of images."""
    # Process the directory
    result = process(populated_directory_config)

    # Verify outputs for each image in the directory
    output_dir = populated_directory_config.output_dir

    # Get all image files from the input directory
    input_dir = Path(populated_directory_config.input)
    image_files = [
        f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    assert len(image_files) > 0, "No image files found in test directory"

    # Check that the result contains processed_videos
    assert "processed_videos" in result, "Missing processed_videos in result"
    assert len(result["processed_videos"]) == len(image_files), (
        "Not all files were processed"
    )

    # Check that we have output files for each input image
    model_name = populated_directory_config.model.name.split("/")[-1]

    for image_file in image_files:
        base_name = image_file

        # Check segmentation output
        segmentation_path = output_dir / f"{base_name}_{model_name}_segmentation.zarr"
        assert os.path.exists(segmentation_path), (
            f"Segmentation file not created for {image_file}"
        )

        # Check overlay output
        overlay_path = output_dir / f"{base_name}_{model_name}_overlay.png"
        assert os.path.exists(overlay_path), (
            f"Overlay file not created for {image_file}"
        )

        # Check analysis output
        counts_path = output_dir / f"{base_name}_{model_name}_category_counts.csv"
        assert os.path.exists(counts_path), f"Counts file not created for {image_file}"
