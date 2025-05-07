"""
Core components for the CitySeg package.

This module provides the core functionality and definitions used throughout
the package, including configuration handling and exception definitions.
"""

from .config import (
    Config,
    ConfigHasher,
    InputType,
    ModelConfig,
    VisualizationConfig,
)
from .exceptions import (
    ConfigurationError,
    InputError,
    ModelError,
    ProcessingError,
)

__all__ = [
    # Configuration
    "Config",
    "ConfigHasher",
    "InputType",
    "ModelConfig",
    "VisualizationConfig",
    # Exceptions
    "ConfigurationError",
    "InputError",
    "ModelError",
    "ProcessingError",
]
