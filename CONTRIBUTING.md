# Contributing to Realtime Duplex Voice Demo

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Ensure you have Python 3.13+ and [uv](https://github.com/astral-sh/uv) installed
2. Clone the repository
3. Install dependencies: `uv sync --all-extras`
4. Install pre-commit hooks (if available)

## Code Quality Standards

All contributions must pass our quality checks:

```bash
# Run all checks
just ci

# Individual checks
just lint        # Ruff linting
just typecheck   # Mypy type checking
just test        # Pytest tests
```

### Code Style

- Use **ruff** for linting and formatting
- Follow PEP 8 with 100 character line length
- Use type hints for all function signatures
- Write docstrings for public APIs (Google style)

### Type Checking

- All code must pass **mypy** in strict mode
- Use explicit type annotations
- Avoid `Any` types where possible

### Testing

- Write unit tests for new functionality
- Integration tests for end-to-end features
- Maintain or improve code coverage
- Use meaningful test names

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature/fix: `git checkout -b feature/my-feature`
3. **Make changes** following code quality standards
4. **Run tests**: `just ci`
5. **Commit** with clear messages following [Conventional Commits](https://www.conventionalcommits.org/)
6. **Push** to your fork
7. **Open a Pull Request** with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/examples if applicable

## Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Build process or tooling changes

Example:
```
feat(tts): add CosyVoice 2 adapter with streaming support

Implements the CosyVoice 2 TTS adapter conforming to the unified
streaming interface. Includes 20ms frame repacketization and
loudness normalization.

Closes #42
```

## Development Workflow

### Running Tests

```bash
# All tests
just test

# Specific test file
uv run pytest tests/unit/test_vad.py -v

# With coverage
uv run pytest --cov=src tests/
```

### Code Generation

After modifying `.proto` files:
```bash
just gen-proto
```

### Profiling

```bash
# CPU profiling
just spy-record <PID>

# GPU profiling
just nsys-tts
```

## Architecture Guidelines

### Adding New TTS Adapters

1. Create adapter in `src/tts/adapters/adapter_<name>.py`
2. Implement `TTSAdapter` protocol from `src/tts/tts_base.py`
3. Support streaming synthesis with 20ms frames at 48kHz
4. Respect PAUSE/RESUME/STOP commands with < 50ms latency
5. Normalize loudness to target LUFS
6. Add unit tests and integration tests
7. Update documentation

### Adding New Features

1. Check if feature aligns with PRD and TDD
2. Discuss approach in an issue first for large changes
3. Follow milestone implementation plan
4. Update tests and documentation
5. Ensure backward compatibility

## Documentation

- Update README.md for user-facing changes
- Update CLAUDE.md for development guidance
- Add docstrings for all public APIs
- Update configuration examples if adding new settings

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for questions and ideas
- Review existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
