# CitySeg Testing Framework

This directory contains tests for the CitySeg project. The testing framework has been completely overhauled to align with the new modular architecture.

## Test Organization

The test suite is organized as follows:

```
tests/
├── conftest.py                       # Shared fixtures and test utilities
├── helpers/                          # Test helper modules
│   ├── assertions.py                 # Custom assertion helpers
│   └── test_data_generators.py       # Test data generation utilities
├── fixtures/                         # Test fixtures and data
│   ├── config/                       # Test configuration files
│   ├── images/                       # Test image files
│   └── videos/                       # Test video files
├── integration/                      # Integration tests
│   ├── README.md                     # Integration testing strategy
│   ├── test_direct_components.py     # Component integration tests
│   └── test_basic_workflow.py        # End-to-end workflow tests
└── test_utils.py                     # Core utility function tests
```

## Test Approach

The testing framework follows these principles:

1. **Simplicity**: Focus on straightforward tests that verify core functionality
2. **Real-world Testing**: Use actual example inputs rather than synthetic data where possible
3. **Component Integration**: Test that components work together properly
4. **Modularity**: Helper modules and fixtures promote code reuse across tests

## Current Test Status

Currently implemented tests:

| Category | Status | Description |
|----------|--------|-------------|
| Utils | ✅ | Core utility functions like segmentation batch handling and logging |
| Direct Component Integration | ✅ | Components working together without Hamilton workflow |
| End-to-End Workflow | 🚧 | Full pipeline tests (some skipped pending Hamilton fixes) |
| CLI Tests | 📝 | Command-line interface tests (planned) |

## Running Tests

The project uses pytest configured in pyproject.toml with standard test discovery and reporting. To run the tests:

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/integration/  # Run all integration tests
uv run pytest tests/test_utils.py  # Run utility tests

# Skip slow tests (marked with @pytest.mark.slow)
uv run pytest -k "not slow"

# Run only specific tests by marker
uv run pytest -m "slow"  # Run only slow tests
```

By default, tests will run with:
- Verbose output (-v)
- Coverage report for src/cityseg
- Short traceback format
- Timing information for slow tests

## Next Steps

See `tests/integration/next_steps.md` for recommendations on expanding the test suite.

## Legacy Tests

The legacy test files have been removed as they no longer align with the new modular architecture. Their functionality has been or will be incorporated into the new test structure.