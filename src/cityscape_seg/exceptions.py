class CityScapeSegError(Exception):
    """Base exception class for CityScapeSeg project."""
    pass

class ConfigurationError(CityScapeSegError):
    """Raised when there's an error in the configuration."""
    pass

class ProcessingError(CityScapeSegError):
    """Raised when there's an error during image or video processing."""
    pass

class ModelError(CityScapeSegError):
    """Raised when there's an error related to the segmentation model."""
    pass

class InputError(CityScapeSegError):
    """Raised when there's an error with the input data."""
    pass