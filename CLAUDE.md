# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands
- Install: `pip install -e .`
- Run tests: `pytest` 
- Run single test: `pytest tests/test_file.py::TestClass::test_function -v`
- Run with coverage: `pytest --cov=src/cityseg --cov-report=term`

## Code Style Guidelines
- Follow PEP 8 for Python code style
- Use type hints for all function parameters and return values
- Import organization: standard library, third-party, local modules (alphabetically within groups)
- Use docstrings for all modules, classes, and functions (including parameters and return values)
- Error handling: Use custom exceptions from exceptions.py
- Naming: snake_case for variables/functions, PascalCase for classes
- Path handling: Use pathlib.Path instead of strings
- Logging: Use loguru.logger with appropriate levels
- Testing: Use pytest fixtures and parametrize for test cases