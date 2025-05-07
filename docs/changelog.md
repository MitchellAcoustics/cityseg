# Changelog

All notable changes to CitySeg will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - Development

### Added
- Comprehensive test fixtures system for reproducible testing
- Integration tests focused on direct component interactions
- Test helpers for common assertions and data generation
- New modular component-based architecture

### Changed
- Complete codebase reorganization into modular structure
- Refactored into component-based architecture with clear separation of concerns
- Moved configuration and exceptions to core/ directory
- Split processing functionality into specialized components
- Improved legacy adapter layer for backward compatibility
- Simplified test structure with focused integration tests
- Enhanced storage layer with Zarr and Parquet adapters
- Improved video resource management

### Removed
- Deprecated monolithic processor classes (moved to legacy namespace)
- Redundant test files (archived for reference)

## [0.3.1] - 2024-04-15

### Added
- Initial release of CitySeg
- Support for OneFormer models
- Image and video processing capabilities
- Multi-video processing for directories
- Comprehensive configuration system
- Output generation including segmentation maps, visualizations, and CSV reports