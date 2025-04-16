"""Integration tests for image processing workflow."""

import pytest
import os
from pathlib import Path
import sys

# Add parent directory to import path
sys.path.append(str(Path(__file__).parent.parent))

from cityseg.workflow.hamilton import process
from cityseg.storage.storage import ZarrSegmentationStorage
from helpers.assertions import (
    assert_segmentation_data_valid,
    assert_analysis_data_valid,
    assert_image_valid,
)

# Import test fixtures


@pytest.mark.skip(reason="Needs complete Hamilton workflow implementation")
@pytest.mark.slow
def test_image_workflow_basic(test_image_config):
    """Test that basic image processing workflow completes successfully."""
    # Process the image
    result = process(test_image_config)

    # Verify we got valid results
    assert result is not None, "Processing returned None"
    assert "segmentation_dataset" in result, "Missing segmentation dataset in results"
    assert "analysis_path" in result, "Missing analysis path in results"
    assert "overlay_path" in result, "Missing overlay path in results"

    # Check segmentation dataset
    seg_dataset = result["segmentation_dataset"]
    assert_segmentation_data_valid(seg_dataset)

    # Check overlay path exists
    overlay_path = result["overlay_path"]
    assert os.path.exists(overlay_path), f"Overlay file not found: {overlay_path}"
    assert_image_valid(overlay_path)

    # Check analysis paths exist
    assert os.path.exists(result["analysis_path"]), (
        f"Analysis file not found: {result['analysis_path']}"
    )

    # Verify analysis data
    assert_analysis_data_valid(
        result["analysis_path"], expected_columns=["category_id", "count"]
    )


@pytest.mark.skip(reason="Needs complete Hamilton workflow implementation")
@pytest.mark.slow
def test_image_output_files(test_image_config):
    """Test that image processing creates expected output files."""
    # Process the image
    process(test_image_config)

    # Get expected file paths
    output_dir = test_image_config.output_dir
    base_name = os.path.basename(test_image_config.input)
    model_name = test_image_config.model.name.split("/")[-1]

    # Check segmentation file exists
    segmentation_path = output_dir / f"{base_name}_{model_name}_segmentation.zarr"
    assert os.path.exists(segmentation_path), "Segmentation file not created"

    # Check segmentation data is valid
    storage = ZarrSegmentationStorage(segmentation_path)
    data = storage.load_segmentation_data()
    assert_segmentation_data_valid(data, min_frames=1)

    # Check colored segmentation exists
    colored_path = output_dir / f"{base_name}_{model_name}_colored.png"
    assert os.path.exists(colored_path), "Colored segmentation not created"
    assert_image_valid(colored_path)

    # Check overlay image exists
    overlay_path = output_dir / f"{base_name}_{model_name}_overlay.png"
    assert os.path.exists(overlay_path), "Overlay image not created"
    assert_image_valid(overlay_path)

    # Check analysis files exist
    counts_path = output_dir / f"{base_name}_{model_name}_category_counts.csv"
    assert os.path.exists(counts_path), "Category counts file not created"
    assert_analysis_data_valid(counts_path, expected_columns=["category_id", "count"])
