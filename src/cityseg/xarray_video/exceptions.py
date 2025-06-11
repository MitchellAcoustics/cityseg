class VideoError(Exception):
    pass


class VideoReadError(VideoError):
    pass


class VideoWriteError(VideoError):
    pass


class VideoProcessingError(VideoError):
    pass


class VideoDisplayError(VideoError):
    pass


class ImageError(Exception):
    pass


class ImageReadError(ImageError):
    pass


class ImageWriteError(ImageError):
    pass


class ImageProcessingError(ImageError):
    pass


class ImageDisplayError(ImageError):
    pass
