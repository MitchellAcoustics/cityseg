from unittest.mock import MagicMock

import pytest

from cityseg.config import Config
from cityseg.processing_plan import ProcessingPlan


@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.force_reprocess = False
    config.save_colored_segmentation = True
    config.save_overlay = True
    config.analyze_results = True
    config.get_output_path.return_value = MagicMock()
    return config


def test_force_reprocess_enabled(mock_config):
    """
    Given force reprocessing is enabled
    When creating a processing plan
    Then all processing steps should be executed
    """
    mock_config.force_reprocess = True
    processing_plan = ProcessingPlan(mock_config)

    expected_plan = {
        "process_video": True,
        "generate_hdf": True,
        "generate_colored_video": True,
        "generate_overlay_video": True,
        "analyze_results": True,
    }

    assert processing_plan.plan == expected_plan


def test_existing_outputs_invalid(mock_config):
    """
    Given force reprocessing is disabled
    And existing outputs are invalid
    When creating a processing plan
    Then all processing steps should be executed
    """
    mock_config.force_reprocess = False

    # Mock the _check_existing_outputs method before creating the ProcessingPlan object
    ProcessingPlan._check_existing_outputs = MagicMock(
        return_value={
            "hdf_file_valid": False,
            "colored_video_valid": False,
            "overlay_video_valid": False,
            "analysis_files_valid": False,
        }
    )

    processing_plan = ProcessingPlan(mock_config)

    expected_plan = {
        "process_video": True,
        "generate_hdf": True,
        "generate_colored_video": True,
        "generate_overlay_video": True,
        "analyze_results": True,
    }

    assert processing_plan.plan == expected_plan


def test_existing_outputs_valid(mock_config):
    """
    Given force reprocessing is disabled
    And existing outputs are valid
    When creating a processing plan
    Then no processing steps should be executed
    """
    mock_config.force_reprocess = False

    # Mock the _check_existing_outputs method before creating the ProcessingPlan object
    ProcessingPlan._check_existing_outputs = MagicMock(
        return_value={
            "hdf_file_valid": True,
            "colored_video_valid": True,
            "overlay_video_valid": True,
            "analysis_files_valid": True,
        }
    )

    processing_plan = ProcessingPlan(mock_config)

    expected_plan = {
        "process_video": False,
        "generate_hdf": False,
        "generate_colored_video": False,
        "generate_overlay_video": False,
        "analyze_results": False,
    }

    assert processing_plan.plan == expected_plan
