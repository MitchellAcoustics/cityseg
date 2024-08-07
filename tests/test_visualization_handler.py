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
