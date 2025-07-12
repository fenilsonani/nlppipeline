# Contributing to NLP Pipeline

First off, thank you for considering contributing to the NLP Pipeline! It's people like you that make this project such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps to reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed after following the steps**
* **Explain which behavior you expected to see instead and why**
* **Include logs and error messages**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior and explain which behavior you expected**
* **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code follows the existing code style
6. Issue that pull request!

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/nlp-pipeline.git
cd nlp-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## Style Guidelines

### Python Style Guide

* Follow PEP 8
* Use Black for code formatting
* Use isort for import sorting
* Maximum line length: 88 characters
* Use type hints where appropriate

### Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

Example:
```
feat: Add batch processing support for entity extraction

- Implement parallel processing for large document sets
- Add progress tracking with tqdm
- Optimize memory usage for batches > 1000 docs

Closes #123
```

### Documentation

* Use docstrings for all public modules, functions, classes, and methods
* Follow Google style for docstrings
* Update README.md with any new features
* Add examples for new functionality

## Testing

* Write unit tests for new functionality
* Ensure all tests pass before submitting PR
* Aim for >80% code coverage for new code
* Include integration tests for complex features

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_models.py
```

## Project Structure

```
nlp-pipeline/
├── src/           # Main source code
├── tests/         # Test files
├── scripts/       # Utility scripts
├── docs/          # Documentation
├── notebooks/     # Jupyter notebooks
└── docker/        # Docker configuration
```

## Questions?

Feel free to open an issue with the tag "question" or reach out to the maintainers directly.

Thank you for contributing! 🎉