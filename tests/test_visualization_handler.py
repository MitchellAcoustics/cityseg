import numpy as np

from cityseg.visualization_handler import VisualizationHandler


def test_visualize_single_image_with_default_palette():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    seg_map = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    result = VisualizationHandler.visualize_segmentation(image, seg_map)
    assert result.shape == image.shape


def test_visualize_single_image_with_custom_palette():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    seg_map = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    palette = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
    result = VisualizationHandler.visualize_segmentation(image, seg_map, palette)
    assert result.shape == image.shape


def test_visualize_single_image_colored_only():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    seg_map = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    result = VisualizationHandler.visualize_segmentation(
        image, seg_map, colored_only=True
    )
    assert result.shape == image.shape


def test_visualize_multiple_images_with_default_palette():
    images = [
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)
    ]
    seg_maps = [np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(3)]
    results = VisualizationHandler.visualize_segmentation(images, seg_maps)
    assert len(results) == 3
    for result, image in zip(results, images):
        assert result.shape == image.shape


def test_visualize_multiple_images_with_custom_palette():
    images = [
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)
    ]
    seg_maps = [np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(3)]
    palette = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
    results = VisualizationHandler.visualize_segmentation(images, seg_maps, palette)
    assert len(results) == 3
    for result, image in zip(results, images):
        assert result.shape == image.shape


def test_visualize_multiple_images_colored_only():
    images = [
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)
    ]
    seg_maps = [np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(3)]
    results = VisualizationHandler.visualize_segmentation(
        images, seg_maps, colored_only=True
    )
    assert len(results) == 3
    for result, image in zip(results, images):
        assert result.shape == image.shape


def test_generate_palette_with_less_colors_than_default():
    palette = VisualizationHandler._generate_palette(10)
    assert palette.shape == (10, 3)


def test_generate_palette_with_more_colors_than_default():
    palette = VisualizationHandler._generate_palette(300)
    assert palette.shape == (300, 3)


def test_visualize_with_list_palette():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    seg_map = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    list_palette = [[i, i, i] for i in range(256)]
    result = VisualizationHandler.visualize_segmentation(image, seg_map, list_palette)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_visualize_with_float_images():
    image = np.random.random((100, 100, 3)).astype(np.float32)
    seg_map = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    result = VisualizationHandler.visualize_segmentation(image, seg_map)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_visualize_with_zero_segmentation():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    seg_map = np.zeros((100, 100), dtype=np.uint8)
    result = VisualizationHandler.visualize_segmentation(image, seg_map)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_visualize_with_different_image_dimensions():
    image = np.random.randint(0, 255, (50, 80, 3), dtype=np.uint8)
    seg_map = np.random.randint(0, 256, (50, 80), dtype=np.uint8)
    result = VisualizationHandler.visualize_segmentation(image, seg_map)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_visualize_with_float_segmentation_maps():
    """
    Test that visualization works correctly with floating-point segmentation maps.

    This tests the fix for the issue where floating-point segmentation maps
    caused an error when used as indices into the palette array.
    """
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Create a segmentation map with floating-point values
    seg_map = np.random.uniform(0, 5, (100, 100)).astype(np.float32)

    # Create a palette that covers the range of integer values we expect
    palette = np.random.randint(0, 255, (6, 3), dtype=np.uint8)

    # This should work without raising an error now
    result = VisualizationHandler.visualize_segmentation(image, seg_map, palette)

    # Check the result
    assert result.shape == image.shape
    assert result.dtype == np.uint8

    # Also test with multiple images
    images = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)]
    seg_maps = [np.random.uniform(0, 5, (50, 50)).astype(np.float32) for _ in range(3)]

    results = VisualizationHandler.visualize_segmentation(images, seg_maps, palette)
    assert len(results) == 3
    for result, image in zip(results, images):
        assert result.shape == image.shape
        assert result.dtype == np.uint8
