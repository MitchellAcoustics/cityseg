#!/usr/bin/env python
"""
Simple script to create test video files for CitySeg tests.

This generates a short test video file that is useful for testing
the video processing pipeline with predictable inputs.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import helpers
sys.path.insert(0, str(Path(__file__).parents[2]))
from helpers.test_data_generators import create_test_video

# Create the test videos directory
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

# Generate a short test video with a moving circle
create_test_video(
    OUTPUT_DIR / "test_video.mp4",
    frame_count=30,
    fps=30,
    width=320,
    height=240,
    pattern="moving_circle",
)
print(f"Created {OUTPUT_DIR / 'test_video.mp4'}")

# Generate a shorter test video with frame numbers
create_test_video(
    OUTPUT_DIR / "test_frames.mp4",
    frame_count=10,
    fps=5,
    width=320,
    height=240,
    pattern="text",
)
print(f"Created {OUTPUT_DIR / 'test_frames.mp4'}")

print("\nTest videos created successfully. Use these for CitySeg tests.")
