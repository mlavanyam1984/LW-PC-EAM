# Contributing to LW-PC-EAM

Thank you for your interest in contributing!

## Reporting Issues
- Search existing issues before opening a new one
- Include Python version, OS, and full error traceback
- For dataset issues, confirm your folder structure matches `data/README.md`

## Pull Requests
1. Fork the repository and create a feature branch (`git checkout -b feature/my-fix`)
2. Run the test suite and ensure all tests pass: `python -m pytest tests/ -v`
3. Add tests for any new functionality
4. Update `CHANGELOG.md` under an `[Unreleased]` section
5. Submit a pull request with a clear description of the change

## Code Style
- Follow PEP 8
- Add type hints to all public functions
- Add docstrings referencing the relevant paper equation (e.g., `[Equation 4]`)

## Dataset
Do NOT commit any part of the MVTec AD dataset to the repository.
It is subject to a non-commercial license and is too large for GitHub.
