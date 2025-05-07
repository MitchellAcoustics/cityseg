#!/usr/bin/env python
"""
Simple script to create test pattern images for CitySeg tests.

This generates a few standard test images that are useful for testing
segmentation pipelines with predictable inputs.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import helpers
sys.path.insert(0, str(Path(__file__).parents[2]))
from helpers.test_data_generators import create_test_image

# Create the test images directory
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

# Generate a simple checkerboard pattern
checkerboard = create_test_image(height=480, width=640, pattern="checkerboard")
checkerboard.save(OUTPUT_DIR / "checkerboard.png")
print(f"Created {OUTPUT_DIR / 'checkerboard.png'}")

# Generate a simple gradient pattern
gradient = create_test_image(height=480, width=640, pattern="gradient")
gradient.save(OUTPUT_DIR / "gradient.png")
print(f"Created {OUTPUT_DIR / 'gradient.png'}")

# Create a small test image (faster processing)
small_test = create_test_image(height=240, width=320, pattern="checkerboard")
small_test.save(OUTPUT_DIR / "test_image.png")
print(f"Created {OUTPUT_DIR / 'test_image.png'}")

print("\nTest images created successfully. Use these for CitySeg tests.")
