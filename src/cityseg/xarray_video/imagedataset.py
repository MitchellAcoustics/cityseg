import xarray

from .exceptions import ImageWriteError
from .image_backend import _write_image


@xarray.register_dataset_accessor("image")
class ImageDataset:
    """Image extension for :class:`xarray.Dataset`.

    Implements operations on a dataset which includes image data
    """

    def __init__(self, xarray_dset):
        self._dset = xarray_dset

    def to_image(self, filename, data_var=None, format: str | None = None):
        """Write Dataset to an image file.

        Args:
            filename (string): name of output file
            data_var (string, Optional): Data variable to write as image.

        Raises:
            ImageWriteError: Incompatible DataArray or file write error
        """
        try:
            _write_image(filename, self._dset[data_var].values, format=format)
        except Exception as e:
            raise ImageWriteError(f"Error writing image: {e}")
