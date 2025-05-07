"""Test demonstrating how to test CitySeg workflows without using real models."""

import pytest
import os
import numpy as np
from pathlib import Path
import sys
import tempfile
import xarray as xr
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from cityseg.core.config import Config, ModelConfig, VisualizationConfig


@pytest.fixture
def test_fixture_image_path():
    """Path to test image in fixtures directory."""
    return Path(__file__).parent.parent / "fixtures" / "images" / "test_image.png"


@pytest.fixture
def test_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_segmentation_result():
    """Create mock segmentation results for testing."""
    # Generate a grid pattern segmentation map
    height, width = 240, 320
    grid_size = 40  # Size of each grid cell
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)

    # Create a simple grid pattern with 10 classes
    grid_x = xx // grid_size % 10
    grid_y = yy // grid_size % 10
    seg_map = (grid_x + grid_y) % 10

    # Create the standard cityscapes class mapping (just the first 10)
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
    ]
    id2label = {i: name for i, name in enumerate(cityscapes_classes[:10])}
    label2id = {name: i for i, name in enumerate(cityscapes_classes[:10])}

    # Create a simple palette
    np.random.seed(42)
    palette = np.random.randint(0, 256, size=(10, 3), dtype=np.uint8)

    # Return segmentation result dict
    return {
        "seg_map": seg_map.astype(np.int32),
        "id2label": id2label,
        "label2id": label2id,
        "palette": palette,
    }


def test_workflow_with_mocked_model(
    test_fixture_image_path, test_output_dir, mock_segmentation_result
):
    """Test workflow with a mocked segmentation model.

    This shows how to test the CitySeg workflow pipeline without downloading
    or loading an actual model, by intercepting the model loading and
    segmentation functions.
    """
    from cityseg.components.segmentation import SegmentationProcessor
    from cityseg.workflow import process

    # Setup the config - normally this would load a model
    config = Config(
        input=test_fixture_image_path,
        output_dir=test_output_dir,
        output_prefix="mock_test",
        model=ModelConfig(
            name="mock/segformer-test",
            model_type="segformer",
            device="cpu",
            max_size=320,
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

    # Define our patch for the create_pipeline function
    def mock_create_pipeline(*args, **kwargs):
        """Mock implementation that returns a dictionary with a process function."""

        # Create a mock process function that returns our pre-defined result
        def mock_process_image(image, *args, **kwargs):
            return mock_segmentation_result

        # Return an object that has the expected structure
        return {
            "process_fn": mock_process_image,
            "id2label": mock_segmentation_result["id2label"],
            "label2id": mock_segmentation_result["label2id"],
            "palette": mock_segmentation_result["palette"],
        }

    # Apply our patch to intercept model loading
    with patch.object(SegmentationProcessor, "create_pipeline", mock_create_pipeline):
        # Process the image using our workflow with the mocked model
        result = process(config)

    # Verify we got valid results
    assert result is not None, "Processing returned None"
    assert "segmentation_dataset" in result, "Missing segmentation dataset in results"

    # Verify the segmentation dataset contains our mock data
    seg_dataset = result["segmentation_dataset"]
    assert isinstance(seg_dataset, xr.Dataset), "Result is not an xarray Dataset"
    assert "segmentation" in seg_dataset.data_vars, "Missing segmentation data variable"

    # Verify the segmentation map is what we expect
    seg_data = seg_dataset["segmentation"].values
    assert seg_data.shape[1:] == mock_segmentation_result["seg_map"].shape, (
        "Shape mismatch in segmentation map"
    )
    assert np.array_equal(seg_data[0], mock_segmentation_result["seg_map"]), (
        "Segmentation map data doesn't match"
    )

    # Check that output files were created
    assert "overlay_path" in result, "Missing overlay path in results"
    overlay_path = result["overlay_path"]
    assert os.path.exists(overlay_path), f"Overlay file not found: {overlay_path}"

    print("\nSuccessfully processed image with mocked segmentation model")
    print(f"Output files in: {test_output_dir}")
