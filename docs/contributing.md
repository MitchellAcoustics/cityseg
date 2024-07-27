# Contributing to CitySeg

We welcome contributions to CitySeg! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally.
3. Create a new branch for your feature or bug fix.

## Making Changes

1. Make your changes in your branch.
2. Add or update tests as necessary.
3. Run the test suite to ensure all tests pass.
4. Update the documentation if you've made changes to the API or added new features.

## Submitting Changes

1. Push your changes to your fork on GitHub.
2. Submit a pull request to the main repository.
3. Describe your changes in the pull request description.

## Code Style

We follow PEP 8 for Python code style. Please ensure your code adheres to these standards. We recommend using [Ruff](https://docs.astral.sh/ruff/) for quick, automatic code formatting, but this is not required. 

### Environment and dependency management

We use [Rye](https://docs.astral.sh/rye/) for managing Python environments and dependencies. If using Rye, you can easily add dependencies to your environment by running `rye add <package>`. This will automatically update the `requirements.txt` and `pyproject.toml` files. If you are not using Rye, please ensure you update these files manually.

## Reporting Issues

If you find a bug or have a suggestion for improvement, please open an issue on the GitHub repository.

Thank you for contributing to CitySeg!