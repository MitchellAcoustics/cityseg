# CitySeg Test Fixtures

This document explains the organization and usage of test fixtures in the CitySeg project.

## Overview

The test fixtures in CitySeg serve several purposes:

1. Provide access to real example data for integration tests
2. Supply standardized test data for consistent test results
3. Create test configurations for different processing scenarios
4. Set up temporary directories and other test resources

## Fixture Organization

Our test fixtures are organized in three ways:

### 1. Python Fixtures (in `conftest.py`)

The file `conftest.py` contains pytest fixtures available to all tests:

- **Path fixtures**: Point to example files or test fixtures
  - `example_video_path`: Real example video from example_inputs
  - `example_image_path`: Real example image from example_inputs
  - `test_fixture_image_path`: Test image from fixtures directory
  - `test_fixture_video_path`: Test video from fixtures directory

- **Directory fixtures**: Create temporary directories
  - `test_temp_dir`: Uses pytest's tmp_path
  - `test_output_dir`: Creates a temporary directory

- **Model fixtures**: Provide model configuration
  - `test_model_name`: Standard test model name
  - `test_model_type`: Model type for tests

- **Configuration fixtures**: Complete test configurations
  - `test_video_config`: Full config for video processing
  - `test_image_config`: Full config for image processing

### 2. Static Test Data (in `fixtures/` directory)

The `fixtures/` directory contains static test data files:

- **`config/`**: YAML configuration files
  - Test configurations for different processing scenarios

- **`images/`**: Test image files
  - Small, predictable images for testing
  - Created by `create_test_pattern.py`

- **`videos/`**: Test video files
  - Short videos for testing
  - Created by `create_test_video.py`

- **`models/`**: Model-related fixtures
  - Placeholder for model-specific fixtures if needed

### 3. Fixture Validation (in `test_fixtures.py`)

The file `test_fixtures.py` validates all fixtures:

- Tests that path fixtures point to valid files
- Verifies directory fixtures create writable directories
- Ensures configuration fixtures are properly set up

## Using Test Fixtures

When writing tests, you can use these fixtures in several ways:

### Integration Tests with Real Data

```python
def test_real_video_processing(example_video_path, test_output_dir):
    # Test with a real video from the example_inputs directory
    ...
```

### Tests with Controlled Test Data

```python
def test_image_processing(test_fixture_image_path, test_output_dir):
    # Test with a controlled test image from fixtures directory
    ...
```

### Tests with Pre-configured Settings

```python
def test_workflow(test_video_config):
    # Test using a complete pre-configured test configuration
    ...
```

## Creating New Test Fixtures

When adding new test fixtures, follow these guidelines:

1. **Python Fixtures**: Add to `conftest.py` if needed by multiple tests
2. **Test Data**: Add to appropriate subdirectory in `fixtures/`
3. **Validation**: Add corresponding tests to `test_fixtures.py`

## Generators vs. Static Files

The test framework uses both approaches:

- **Generator functions** (in `helpers/test_data_generators.py`) create dynamic test data
- **Static files** (in `fixtures/`) provide consistent test data

Use the approach that makes the most sense for your test case.