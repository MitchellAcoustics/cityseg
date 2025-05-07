"""Demo test to verify basic CitySeg functionality."""

import pytest
import os
from pathlib import Path
import sys
import tempfile

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from cityseg.core.config import Config, ModelConfig, VisualizationConfig
from helpers.assertions import assert_image_valid


@pytest.fixture
def test_fixture_image_path():
    """Path to test image in fixtures directory."""
    return Path(__file__).parent.parent / "fixtures" / "images" / "test_image.png"


@pytest.fixture
def test_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_simple_segmentation(test_fixture_image_path, test_output_dir):
    """Test basic segmentation of a single image.

    This test uses the actual segformer-b0 model to verify the core functionality
    of CitySeg operates correctly.
    """
    # Skip this test if the user doesn't want to download the model
    pytest.importorskip("torch")

    # Import here to avoid importing if test is skipped
    from cityseg.workflow import process

    # Setup basic configuration
    config = Config(
        input=test_fixture_image_path,
        output_dir=test_output_dir,
        output_prefix="test_demo",
        model=ModelConfig(
            name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            device="cpu",
            max_size=320,  # Very small size for quick tests
            num_workers=0,
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

    # Verify we got valid results
    assert result is not None, "Processing returned None"

    # Check that the segmentation dataset was created
    assert "segmentation_dataset" in result, "Missing segmentation dataset in results"
    seg_dataset = result["segmentation_dataset"]

    # Verify the dataset has the expected attributes
    assert "segmentation" in seg_dataset.data_vars, "Missing segmentation data variable"
    assert (
        seg_dataset.attrs["model_name"]
        == "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    )

    # Check that output files were created
    assert "overlay_path" in result, "Missing overlay path in results"
    overlay_path = result["overlay_path"]
    assert os.path.exists(overlay_path), f"Overlay file not found: {overlay_path}"
    assert_image_valid(overlay_path)

    print(f"\nSuccessfully processed {test_fixture_image_path} with segformer-b0")
    print(f"Output files in: {test_output_dir}")
