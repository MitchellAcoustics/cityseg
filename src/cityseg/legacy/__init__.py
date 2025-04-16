"""
Legacy components for backward compatibility.

This module contains legacy adapters that maintain the original interface
while delegating to the new component-based architecture.
"""

from .processors import (
    ImageProcessorLegacy,
    VideoProcessorLegacy,
    DirectoryProcessorLegacy,
    create_processor,
)

__all__ = [
    "ImageProcessorLegacy",
    "VideoProcessorLegacy",
    "DirectoryProcessorLegacy",
    "create_processor",
]
