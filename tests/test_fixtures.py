"""Tests for the test fixtures.

This file verifies that all test fixtures are working correctly and
pointing to valid files/directories.
"""

import os
import pytest
from pathlib import Path


# --- Tests for example data fixtures ---

def test_example_video_path(example_video_path):
    """Test that the example video path fixture returns a valid file."""
    assert os.path.exists(example_video_path), (
        f"Example video not found at {example_video_path}"
    )
    assert os.path.isfile(example_video_path), f"{example_video_path} is not a file"
    assert example_video_path.suffix in (".mp4", ".avi", ".mov"), (
        f"{example_video_path} is not a video file"
    )


def test_example_image_path(example_image_path):
    """Test that the example image path fixture returns a valid file."""
    assert os.path.exists(example_image_path), (
        f"Example image not found at {example_image_path}"
    )
    assert os.path.isfile(example_image_path), f"{example_image_path} is not a file"
    assert example_image_path.suffix in (".png", ".jpg", ".jpeg"), (
        f"{example_image_path} is not an image file"
    )


# --- Tests for fixtures directory fixtures ---

def test_fixture_image_path(test_fixture_image_path):
    """Test that the fixture image path points to a valid file."""
    assert os.path.exists(test_fixture_image_path), (
        f"Fixture image not found at {test_fixture_image_path}"
    )
    assert os.path.isfile(test_fixture_image_path), (
        f"{test_fixture_image_path} is not a file"
    )
    assert test_fixture_image_path.suffix in (".png", ".jpg", ".jpeg"), (
        f"{test_fixture_image_path} is not an image file"
    )


def test_fixture_video_path(test_fixture_video_path):
    """Test that the fixture video path points to a valid file."""
    assert os.path.exists(test_fixture_video_path), (
        f"Fixture video not found at {test_fixture_video_path}"
    )
    assert os.path.isfile(test_fixture_video_path), (
        f"{test_fixture_video_path} is not a file"
    )
    assert test_fixture_video_path.suffix in (".mp4", ".avi", ".mov"), (
        f"{test_fixture_video_path} is not a video file"
    )


# --- Tests for directory fixtures ---

def test_temp_dir_fixture(test_temp_dir):
    """Test that the temporary directory fixture creates a valid directory."""
    assert os.path.exists(test_temp_dir), (
        f"Temp directory not created at {test_temp_dir}"
    )
    assert os.path.isdir(test_temp_dir), f"{test_temp_dir} is not a directory"

    # Test we can write to the directory
    test_file = test_temp_dir / "test_file.txt"
    with open(test_file, "w") as f:
        f.write("Test content")

    assert os.path.exists(test_file), f"Could not write to {test_temp_dir}"


def test_output_dir_fixture(test_output_dir):
    """Test that the output directory fixture creates a valid directory."""
    assert os.path.exists(test_output_dir), (
        f"Output directory not created at {test_output_dir}"
    )
    assert os.path.isdir(test_output_dir), f"{test_output_dir} is not a directory"

    # Test we can write to the directory
    test_file = test_output_dir / "test_file.txt"
    with open(test_file, "w") as f:
        f.write("Test content")

    assert os.path.exists(test_file), f"Could not write to {test_output_dir}"


# --- Tests for configuration fixtures ---

def test_video_config_fixture(test_video_config):
    """Test that the video configuration fixture is valid."""
    assert test_video_config is not None
    assert test_video_config.input_type.value == "single_video"
    assert test_video_config.model.name is not None
    assert test_video_config.output_dir is not None


def test_image_config_fixture(test_image_config):
    """Test that the image configuration fixture is valid."""
    assert test_image_config is not None
    assert test_image_config.input_type.value == "single_image"
    assert test_image_config.model.name is not None
    assert test_image_config.output_dir is not None
