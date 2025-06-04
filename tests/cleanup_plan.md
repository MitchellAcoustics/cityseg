# CitySeg Integration Test Cleanup Plan

## Files to Keep

1. **`test_direct_components.py`** ✅
   - Contains core integration tests that directly test components working together
   - Tests both image and video processing in a straightforward way
   - Uses real example inputs and real models
   - Focuses on component interaction rather than the Hamilton workflow

2. **`test_basic_workflow.py`** ✅
   - Contains proper integration tests using the Hamilton workflow
   - Uses real example inputs and real models
   - Tests the end-to-end processing pipeline
   - Currently has issues with Hamilton configuration (some tests skipped)

## Files to Remove/Merge

1. **`test_config.py`** and **`test_config_input_types.py`** ❌
   - These are essentially unit tests for the Config class, not integration tests
   - They should be moved to a unit test file or removed if redundant

2. **`test_dummy.py`** ❌
   - Contains a simple test using generated data rather than real inputs
   - Not a true integration test as it doesn't test components working together
   - Can be removed or moved to a unit test file

3. **`test_demo.py`** ❌
   - Similar to test_basic_workflow.py but with less coverage
   - Redundant with the more comprehensive tests in test_basic_workflow.py

4. **`test_mock_model.py`** ❌
   - Uses a mocked model rather than testing real integration
   - Not focused on testing the actual components working together
   - More appropriate as a unit test or example

5. **`test_file_handler.py`** ❌
   - Likely a unit test for file handling, not a true integration test

6. **`test_storage.py`** ❌
   - Likely a unit test for storage components, not a true integration test

7. **`test_image_workflow.py`**, **`test_video_workflow.py`**, **`test_directory_workflow.py`** ❌
   - These appear to be redundant with test_basic_workflow.py which already tests these workflows
   - Should be consolidated into a single comprehensive workflow test file

8. **`test_fixtures.py`** ❌
   - Likely just contains fixture definitions, not actual tests
   - Should be moved to conftest.py or a helper module

9. **`test_cli.py`** ❌
   - CLI tests are typically separate from integration tests
   - Should be kept separate or implemented properly

## Implementation Plan

1. Keep only `test_direct_components.py` and `test_basic_workflow.py`

2. Ensure all necessary test fixtures are properly defined in conftest.py

3. Update these files to be more comprehensive if needed:
   - Make sure test_direct_components.py covers the core components working together
   - Make sure test_basic_workflow.py covers all three input types (image, video, directory)
   - Ensure test coverage is sufficient and focused on integration points

4. Remove unnecessary test files that are redundant or not true integration tests