from typing import Literal

import numpy as np
from PIL import Image

from xarray import DataArray
from xarray.backends.common import BackendArray

from .exceptions import ImageReadError, ImageWriteError

from pathlib import Path


class ImageArrayWrapper(BackendArray):
    """A wrapper around image dataset objects."""

    def __init__(self, shape):
        self._shape = shape
        self._dtype = np.dtype("uint8")

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape


def _open_image(
    filename,
    mode: Literal["r"] = "r",
    formats: list[str] | tuple[str, ...] | None = None,
):
    try:
        img = Image.open(filename, mode=mode, formats=formats)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    except Exception as e:
        raise ImageReadError(f"Failed to read image {filename}: {e}") from e


def _write_image(
    filename: str | Path,
    array,
    metadata: dict | None = None,
    format: str | None = None,
    **params,
):
    img = Image.fromarray(array.astype(np.uint8), mode="RGB")

    if metadata is not None:
        for key, value in metadata.items():
            img.info[key] = value
    try:
        img.save(filename, format=format, **params)
    except Exception as e:
        raise ImageWriteError(f"Failed to write image {filename}: {e}") from e

    img.close()  # Close the image file to free resources


def _resize_image(
    img: Image.Image, size: tuple[int, int] | list[int] | np.ndarray, **kwargs
) -> Image.Image:
    """
    Resize an image to the specified size using the given resampling method.

    Parameters
    ----------
    img : Image.Image
        The image to resize.
    size : tuple[int, int]
        The target size (width, height).
    **kwargs
        Additional keyword arguments passed to PIL.Image.resize.

    Returns
    -------
    Image.Image
        The resized image.
    """
    return img.resize(size, **kwargs)


def open_image(
    filename: str | Path,
    mode: Literal["r"] = "r",
    formats: list[str] | tuple[str, ...] | None = None,
    resize: tuple[int, int] | list[int] | np.ndarray | None = None,
    **resize_kwargs,
) -> DataArray:
    """
    Image file into an xarray DataSet.

    This reads an image file and returns it as an xarray DataSet.

    Parameters
    ----------
    filename : str | Path
        The path to the image file.
    **resize_kwargs
        Additional keyword arguments passed to PIL.Image.resize .
    Returns
    -------
    An xarray Dataset containing the image data.

    Raises
    ------
    ImageReadError
        If the image cannot be read or converted to RGB.
    """

    file_path = Path(filename)

    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    img = _open_image(file_path, mode=mode, formats=formats)

    if resize is not None:
        if not isinstance(resize, tuple) or len(resize) != 2:
            raise ValueError("Resize parameter must be a tuple of (width, height).")
        img = _resize_image(img, resize, **resize_kwargs)

    width = img.width
    height = img.height

    # Coordinates
    coords = {
        "channel": ["R", "G", "B"],
        "pixel_x": np.arange(width),
        "pixel_y": np.arange(height),
    }

    # Attributes
    attrs = {"filename": file_path.name, "_image": img.mode}

    # Data
    data = np.array(img, dtype=np.uint8)

    dataarray = DataArray(
        data=data,
        dims=("pixel_y", "pixel_x", "channel"),
        coords=coords,
        attrs=attrs,
    )

    img.close()  # Close the image file to free resources

    return dataarray
