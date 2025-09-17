import xarray
import numpy as np

from .exceptions import ImageError


@xarray.register_dataarray_accessor("image")
class ImageArray:
    """Image extension for class `xarray.DataArray`.

    Implements operations on a dataset which includes image data.

    Based on the VideoArray class from xarray_video: https://xarray-video.readthedocs.io/en/latest/index.html
    """

    def __init__(self, xarray_arr):
        self._arr = xarray_arr

    def _check_image(self, ndim=3):
        if len(self._arr.shape) != ndim or self._arr.shape[-1] != 3:
            raise ImageError(
                f"Expected an image with {ndim} dimensions and 3 channels, "
                f"but got shape {self._arr.shape}."
            )
        if self._arr.dtype != np.uint8:
            raise ImageError(
                f"Expected image data type to be uint8, but got {self._arr.dtype}."
            )
        return np.prod(self._arr.shape[:-1])

    def plot(self, **kwargs):
        """
        Plot DataArray as an image using matplotlib.imshow.

        Raises:
            ImageError: If the DataArray does not represent a valid image.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImageError("matplotlib is required for plotting images.")
        self._check_image(3)
        plt.imshow(self._arr.values, **kwargs)
        plt.show()
