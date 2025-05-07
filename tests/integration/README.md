# CitySeg Integration Tests

This directory contains streamlined integration tests for the CitySeg package, focusing on verifying that the components work together correctly with real-world inputs.

## Simplified Test Approach

Integration tests have been simplified to focus on the most important aspects of the system:

1. **Direct Component Integration** (`test_direct_components.py`):
   - Tests components directly working together without Hamilton workflow
   - Image loading, segmentation, and visualization
   - Video frame extraction, processing, and visualization
   - Uses real example inputs and real models
   - Verifies core functionality in isolation

2. **End-to-End Workflow** (`test_basic_workflow.py`):
   - Tests the complete pipeline using Hamilton workflow engine
   - Covers both image and video processing workflows
   - Some tests may be skipped if Hamilton configuration issues exist
   - Verifies the entire system working together

## Running Tests

Tests can be run using pytest through UV:

```bash
# Run all integration tests
uv run pytest tests/integration/

# Run direct component tests only
uv run pytest tests/integration/test_direct_components.py

# Run a specific test function
uv run pytest tests/integration/test_direct_components.py::test_image_segmentation_direct

# Skip slow tests (video processing)
uv run pytest tests/integration/ -k "not slow"
```

## Test Data

Tests use real example data from `example_inputs/` for authentic testing. Tests will be skipped if this data is unavailable.

## Key Components Tested

1. **Image Processing Pipeline**:
   - Image loading and preprocessing
   - Segmentation model integration
   - Result visualization and analysis

2. **Video Processing Pipeline**:
   - Video frame extraction
   - Batch segmentation processing
   - Result aggregation and analysis

## Testing Philosophy

Our integration tests prioritize:

1. **Simplicity**: Straightforward tests that directly verify components work together
2. **Authenticity**: Using real example data rather than synthetic data
3. **Practicality**: Focusing on the most important workflows first

This approach provides comprehensive validation of core functionality while remaining maintainable and efficient.