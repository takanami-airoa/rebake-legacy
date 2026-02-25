# Contributing to Robot Data Pipeline

We welcome contributions to make robot data conversion more accessible and support more platforms! 

## Quick Start for Contributors

### Development Setup

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/your-username/robot_data_pipeline.git
   cd robot_data_pipeline
   ```

2. Install dependencies:
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 uv sync
   ```

3. Run pre-commit checks to ensure everything works:
   ```bash
   make format && make lint && make test
   ```

## Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and add tests**
   - Add unit tests for new functionality
   - Update documentation as needed

3. **Run quality checks:**
   ```bash
   make format    # Format code with ruff
   make lint      # Check linting and types with ruff + mypy
   make test      # Run test suite
   ```

4. **Commit your changes:**
   ```bash
   git commit -m "Add your descriptive commit message"
   ```
   - Use clear, descriptive commit messages
   - Reference issue numbers when applicable

5. **Push and create a Pull Request:**
   ```bash
   git push origin feature/your-feature-name
   ```

## Pull Request Guidelines

- **Title**: Clear and descriptive
- **Description**: Explain what changes you made and why
- **Tests**: Include tests for new functionality
- **Quality**: All checks must pass (`make format && make lint && make test`)
- **Documentation**: Update README.md if needed
- **Reference**: Link any related issues

## Reporting Issues

### Bug Reports
- Use clear, descriptive titles
- Include steps to reproduce
- Provide environment details (OS, Python version, etc.)
- Include relevant log outputs

### Feature Requests
- Describe the feature and its use case
- Explain how it would benefit the community
- Consider implementation complexity

## Types of Contributions

### Code Contributions
- Bug fixes
- New features
- Performance improvements
- Code refactoring

### Robot Platform Support
- New robot platform plugins
- Improvements to existing robot support
- Configuration examples

### Documentation
- README improvements
- Code documentation
- Usage examples

## Code Standards

- **Python 3.10+** compatible
- **Type hints** required for new code
- **Tests** required for new functionality
- **Format**: Code must pass `make format`
- **Linting**: Code must pass `make lint`

## Questions & Support

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion

## License

By contributing to this project, you agree that your contributions will be licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

Thank you for contributing to Robot Data Pipeline! 🤖
