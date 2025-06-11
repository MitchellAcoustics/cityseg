from numcodecs.registry import register_codec

from .codecs.h264 import H264

register_codec(H264)


from .video_backend import open_video  # noqa: E402
from .videoarray import VideoArray  # noqa: E402
from .videodataset import VideoDataset  # noqa: E402

from .image_backend import open_image  # noqa: E402
from .imagearray import ImageArray  # noqa: E402
from .imagedataset import ImageDataset  # noqa: E402

"""Top-level package for xarray-video."""

__author__ = """Oceanum Developers"""
__email__ = "developers@oceanum.science"
__version__ = "0.2.11"

__all__ = [
    "VideoDataset",
    "VideoArray",
    "open_video",
    "ImageDataset",
    "ImageArray",
    "open_image",
]
