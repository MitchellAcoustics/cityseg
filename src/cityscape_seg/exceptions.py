class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""

    pass


class InputError(Exception):
    """Exception raised for errors with input data."""

    pass


class ModelError(Exception):
    """Exception raised for errors related to the segmentation model."""

    pass


class ProcessingError(Exception):
    """Exception raised for errors during data processing."""

    pass
