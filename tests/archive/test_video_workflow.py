"""Integration tests for video processing workflow."""

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
    assert_video_valid,
    assert_processing_history_valid,
)

# Import test fixtures


@pytest.mark.skip(reason="Needs complete Hamilton workflow implementation")
@pytest.mark.slow
def test_video_workflow_basic(test_video_config):
    """Test that basic video processing workflow completes successfully."""
    # Process the video
    result = process(test_video_config)

    # Verify we got valid results
    assert result is not None, "Processing returned None"
    assert "segmentation_dataset" in result, "Missing segmentation dataset in results"
    assert "analysis_path" in result, "Missing analysis path in results"
    assert "overlay_path" in result, "Missing overlay path in results"

    # Check segmentation dataset
    seg_dataset = result["segmentation_dataset"]
    assert_segmentation_data_valid(seg_dataset)

    # Check overlay path exists and is a valid video
    overlay_path = result["overlay_path"]
    assert os.path.exists(overlay_path), f"Overlay file not found: {overlay_path}"
    assert_video_valid(overlay_path)

    # Check analysis paths exist
    assert os.path.exists(result["analysis_path"]), (
        f"Analysis file not found: {result['analysis_path']}"
    )

    # Verify analysis data
    assert_analysis_data_valid(
        result["analysis_path"], expected_columns=["frame", "category_id", "count"]
    )


@pytest.mark.skip(reason="Needs complete Hamilton workflow implementation")
@pytest.mark.slow
def test_video_output_files(test_video_config):
    """Test that video processing creates expected output files."""
    # Process the video
    process(test_video_config)

    # Get expected file paths
    output_dir = test_video_config.output_dir
    base_name = os.path.basename(test_video_config.input)
    model_name = test_video_config.model.name.split("/")[-1]
    frame_step = test_video_config.frame_step

    # Check segmentation file exists
    segmentation_path = (
        output_dir / f"{base_name}_{model_name}_step{frame_step}_segmentation.zarr"
    )
    assert os.path.exists(segmentation_path), "Segmentation file not created"

    # Check segmentation data is valid
    storage = ZarrSegmentationStorage(segmentation_path)
    data = storage.load_segmentation_data()
    assert_segmentation_data_valid(data)

    # Check colored segmentation exists
    colored_path = output_dir / f"{base_name}_{model_name}_step{frame_step}_colored.mp4"
    assert os.path.exists(colored_path), "Colored segmentation video not created"
    assert_video_valid(colored_path)

    # Check overlay video exists
    overlay_path = output_dir / f"{base_name}_{model_name}_step{frame_step}_overlay.mp4"
    assert os.path.exists(overlay_path), "Overlay video not created"
    assert_video_valid(overlay_path)

    # Check analysis files exist
    category_counts_path = (
        output_dir / f"{base_name}_{model_name}_step{frame_step}_category_counts.csv"
    )
    assert os.path.exists(category_counts_path), "Category counts file not created"
    assert_analysis_data_valid(
        category_counts_path, expected_columns=["frame", "category_id", "count"]
    )

    category_percentages_path = (
        output_dir
        / f"{base_name}_{model_name}_step{frame_step}_category_percentages.csv"
    )
    assert os.path.exists(category_percentages_path), (
        "Category percentages file not created"
    )
    assert_analysis_data_valid(
        category_percentages_path,
        expected_columns=["frame", "category_id", "percentage"],
    )

    # Check processing history file exists
    history_path = (
        output_dir
        / f"{base_name}_{model_name}_step{frame_step}_processing_history.json"
    )
    assert os.path.exists(history_path), "Processing history file not created"
    assert_processing_history_valid(history_path)
