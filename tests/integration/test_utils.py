"""Test utilities for integration tests.

This module provides utility functions and pytest markers for integration tests,
including environment detection for gRPC compatibility.
"""

import os
import platform
import sys
from typing import Final

import pytest

# Environment detection constants
_WSL_INDICATORS: Final[tuple[str, ...]] = ("microsoft", "wsl")


def is_wsl2() -> bool:
    """Detect if running in Windows Subsystem for Linux 2.

    WSL2 is a known problematic environment for grpc-python due to threading
    and event loop interactions that cause segmentation faults.

    Returns:
        True if running in WSL2, False otherwise
    """
    if platform.system() != "Linux":
        return False

    try:
        with open("/proc/version", encoding="utf-8") as f:
            version_info = f.read().lower()
            return any(indicator in version_info for indicator in _WSL_INDICATORS)
    except (FileNotFoundError, OSError, PermissionError):
        return False


def is_grpc_safe_environment() -> bool:
    """Detect if the current environment is safe for grpc-python tests.

    Known unsafe environments:
    - WSL2 (Windows Subsystem for Linux 2)
    - Certain Linux kernel versions with threading issues

    Environment variable overrides:
    - GRPC_TESTS_ENABLED=1: Force enable tests (ignore environment detection)
    - GRPC_TESTS_ENABLED=0: Force disable tests (skip regardless of environment)
    - GRPC_TESTS_FORKED=1: Indicates tests are running with --forked flag

    Returns:
        True if environment is safe for grpc tests, False otherwise

    Examples:
        >>> # WSL2 environment, tests disabled by default
        >>> is_grpc_safe_environment()
        False

        >>> # Force enable in WSL2 (risky, may segfault)
        >>> os.environ["GRPC_TESTS_ENABLED"] = "1"
        >>> is_grpc_safe_environment()
        True

        >>> # Running with --forked flag, safe even in WSL2
        >>> os.environ["GRPC_TESTS_FORKED"] = "1"
        >>> is_grpc_safe_environment()
        True
    """
    # Check for explicit override first
    grpc_enabled = os.getenv("GRPC_TESTS_ENABLED")
    if grpc_enabled is not None:
        return grpc_enabled == "1"

    # Check if running with --forked flag (process isolation)
    # When using --forked, tests are safe even in problematic environments
    if os.getenv("GRPC_TESTS_FORKED") == "1":
        return True

    # Check for known problematic environments
    if is_wsl2():
        return False

    # Default: assume safe for native Linux, macOS, etc.
    return True


def get_skip_reason() -> str:
    """Get detailed reason why gRPC tests are being skipped.

    Returns:
        Human-readable explanation of why tests are skipped
    """
    if is_wsl2():
        return (
            "gRPC tests segfault in WSL2 environment. "
            "Options: (1) Run with GRPC_TESTS_ENABLED=1 to force enable (risky), "
            "(2) Use 'just test-integration' which runs with --forked flag, "
            "(3) Run in native Linux/macOS environment, "
            "(4) Use Docker to run tests. "
            "See GRPC_SEGFAULT_WORKAROUND.md for details."
        )

    grpc_enabled = os.getenv("GRPC_TESTS_ENABLED")
    if grpc_enabled == "0":
        return (
            "gRPC tests disabled by GRPC_TESTS_ENABLED=0. "
            "Set GRPC_TESTS_ENABLED=1 to enable."
        )

    return "gRPC tests disabled in this environment."


# Pytest markers for conditional test skipping
skip_if_grpc_unsafe = pytest.mark.skipif(
    not is_grpc_safe_environment(),
    reason=get_skip_reason(),
)

# Alias for backward compatibility
requires_grpc_safe_environment = skip_if_grpc_unsafe


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers with pytest.

    This function is automatically called by pytest during configuration.

    Args:
        config: Pytest configuration object
    """
    config.addinivalue_line(
        "markers",
        "grpc_unsafe: Tests that may segfault in certain environments (WSL2, etc.)"
    )
    config.addinivalue_line(
        "markers",
        "requires_grpc: Tests that require a gRPC-safe environment to run"
    )


# Module-level diagnostics for debugging
def print_environment_info() -> None:
    """Print diagnostic information about the test environment.

    Useful for debugging test skipping behavior.
    """
    print("\n" + "=" * 60)
    print("gRPC Test Environment Diagnostics")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"WSL2 Detected: {is_wsl2()}")
    print(f"GRPC_TESTS_ENABLED: {os.getenv('GRPC_TESTS_ENABLED', '(not set)')}")
    print(f"GRPC_TESTS_FORKED: {os.getenv('GRPC_TESTS_FORKED', '(not set)')}")
    print(f"Is Safe Environment: {is_grpc_safe_environment()}")
    if not is_grpc_safe_environment():
        print(f"\nSkip Reason: {get_skip_reason()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Allow running this module directly for diagnostics
    print_environment_info()
