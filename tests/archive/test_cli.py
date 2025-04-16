"""Integration tests for CLI functionality."""

import pytest
import os
import subprocess
import yaml
from pathlib import Path
import sys

# Add parent directory to import path
sys.path.append(str(Path(__file__).parent.parent))

# Import test fixtures


@pytest.fixture
def test_config_file(
    test_temp_dir, example_image_path, test_model_name, test_model_type
):
    """Create a temporary config file for CLI testing."""
    # Configure output directory
    output_dir = test_temp_dir / "cli_output"
    output_dir.mkdir(exist_ok=True)

    config = {
        "input": str(example_image_path),
        "output_dir": str(output_dir),
        "model": {
            "name": test_model_name,
            "model_type": test_model_type,
            "device": "cpu",
            "max_size": 640,  # Smaller size for faster testing
            "num_workers": 0,  # Avoid multiprocessing for tests
            "pipe_batch": 1,
        },
        "frame_step": 1,  # Not used for images but required
        "batch_size": 1,
        "save_raw_segmentation": True,
        "save_colored_segmentation": True,
        "save_overlay": True,
        "analyze_results": True,
        "visualization": {
            "alpha": 0.5,
            "colormap": "default",
        },
        "force_reprocess": True,
        "disable_tqdm": True,  # Disable progress bars for cleaner test output
    }

    config_path = test_temp_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.mark.skip(reason="Needs proper CLI test implementation")
@pytest.mark.slow
def test_cli_execution(test_config_file):
    """Test running cityseg through the command line interface."""
    result = subprocess.run(
        ["uv", "run", "python", "-m", "cityseg.main", str(test_config_file)],
        capture_output=True,
        text=True,
        cwd=str(Path(test_config_file).parent),
    )

    # Print the output for debugging if it fails
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")

    # Check command execution was successful
    assert result.returncode == 0, f"CLI execution failed: {result.stderr}"

    # Get the expected output directory
    with open(test_config_file, "r") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])

    # Check that output files were created
    output_files = os.listdir(output_dir)

    # Check for overlay file
    overlay_files = [f for f in output_files if f.endswith("_overlay.png")]
    assert len(overlay_files) > 0, f"No overlay output found in {output_files}"

    # Check for segmentation file
    segmentation_files = [f for f in output_files if f.endswith("_segmentation.zarr")]
    assert len(segmentation_files) > 0, (
        f"No segmentation output found in {output_files}"
    )

    # Check for analysis files
    analysis_files = [f for f in output_files if f.endswith("_category_counts.csv")]
    assert len(analysis_files) > 0, f"No analysis output found in {output_files}"
