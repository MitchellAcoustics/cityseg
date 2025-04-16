# Test Fixtures for CitySeg

This directory contains test fixtures for CitySeg tests:

## `/images`
Test image files for segmentation tests:
- Small test images with predictable patterns

## `/videos`
Test video files for video processing tests:
- Short test videos (10-30 frames) with predictable content

## `/config`
YAML configuration files for testing:
- `test_image_config.yaml`: Configuration for image processing tests
- `test_video_config.yaml`: Configuration for video processing tests
- `test_model_config.yaml`: Simple model configuration

## Usage

You can recreate the test images and videos by running:
```bash
cd tests/fixtures/images
python create_test_pattern.py

cd tests/fixtures/videos
python create_test_video.py
```

For tests, you can use these fixtures with the existing pytest fixtures. Example:

```python
import pytest
from pathlib import Path

@pytest.fixture
def test_image_path():
    """Path to a test image."""
    return Path(__file__).parent / "fixtures" / "images" / "test_image.png"
```