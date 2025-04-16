"""
This module provides functionality for processing image files.

It encapsulates methods for loading, preprocessing, and saving images for segmentation.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
from loguru import logger


class ImageProcessor:
    """
    A class for processing image files.

    This class provides methods to load and preprocess images for segmentation,
    including resizing, format conversion, and other transformations.

    Methods:
        load_image: Load an image from a file path
        resize_image: Resize an image to a maximum dimension
        save_image: Save an image to a file
    """

    @staticmethod
    def load_image(image_path: Path) -> Image.Image:
        """
        Load an image from a file path.

        Args:
            image_path: Path to the image file

        Returns:
            Loaded PIL Image object in RGB format
        """
        try:
            image = Image.open(image_path).convert("RGB")
            logger.debug(
                f"Loaded image from {image_path}: {image.width}x{image.height}"
            )
            return image
        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {str(e)}")
            raise

    @staticmethod
    def resize_image(image: Image.Image, max_size: Optional[int] = None) -> Image.Image:
        """
        Resize an image to a maximum dimension while preserving aspect ratio.

        Args:
            image: PIL Image to resize
            max_size: Maximum dimension (width or height) for the resized image

        Returns:
            Resized PIL Image
        """
        if max_size is None:
            return image

        # Make a copy to avoid modifying the original
        resized = image.copy()
        resized.thumbnail((max_size, max_size))
        logger.debug(
            f"Resized image: {image.width}x{image.height} -> {resized.width}x{resized.height}"
        )
        return resized

    @staticmethod
    def save_image(
        image: Union[Image.Image, np.ndarray],
        output_path: Path,
        format: Optional[str] = None,
    ) -> Path:
        """
        Save an image to a file.

        Args:
            image: PIL Image or numpy array to save
            output_path: Path where the image will be saved
            format: Optional format override (e.g., 'PNG', 'JPEG')

        Returns:
            Path to the saved image
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Handle different dtype ranges
            if image.dtype == np.float32 or image.dtype == np.float64:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)

            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = image

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the image
        pil_image.save(output_path, format=format)
        logger.debug(f"Saved image to {output_path}")

        return output_path
