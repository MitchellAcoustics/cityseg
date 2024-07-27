"""
This module defines custom exception classes for the Semantic Segmentation Pipeline.

These exceptions are used to provide more specific error handling and
messaging throughout the application.
"""

class ConfigurationError(Exception):
    """
    Exception raised for errors in the configuration.

    This exception is raised when there are issues with the pipeline's
    configuration, such as missing or invalid configuration parameters.
    """

    pass


class InputError(Exception):
    """
    Exception raised for errors with input data.

    This exception is raised when there are issues with the input data,
    such as invalid file formats or missing input files.
    """

    pass


class ModelError(Exception):
    """
    Exception raised for errors related to the segmentation model.

    This exception is raised when there are issues with the segmentation model,
    such as incompatible model types or failures during model loading.
    """

    pass


class ProcessingError(Exception):
    """
    Exception raised for errors during data processing.

    This exception is raised when there are issues during the actual
    processing of data, such as failures in segmentation or output generation.
    """

    pass