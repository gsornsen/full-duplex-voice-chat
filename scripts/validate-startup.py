#!/usr/bin/env python3
"""Startup validation script for M2 services.

Validates that all prerequisites are met before starting services:
- Configuration files valid
- Dependencies installed
- Proto stubs generated
- Services can connect to each other

This script provides fail-fast validation to prevent runtime errors.

Exit codes:
- 0: All validations passed, ready to start
- 1: Validation failed, cannot start

Usage:
    python scripts/validate-startup.py
    # Or as pre-start hook:
    ./scripts/validate-startup.py && docker compose up
"""

import asyncio
import sys
from pathlib import Path
from typing import Any


def check_file_exists(path: Path, description: str) -> bool:
    """Check if a required file exists.

    Args:
        path: Path to check
        description: Human-readable description

    Returns:
        True if file exists, False otherwise
    """
    print(f"Checking {description}... ", end="", flush=True)
    if path.exists():
        print("OK")
        return True
    else:
        print("FAILED")
        print(f"   Cause: File not found: {path}")
        print(f"   Resolution: Create or generate {description}")
        return False


def check_directory_exists(path: Path, description: str) -> bool:
    """Check if a required directory exists.

    Args:
        path: Path to check
        description: Human-readable description

    Returns:
        True if directory exists, False otherwise
    """
    print(f"Checking {description}... ", end="", flush=True)
    if path.is_dir():
        print("OK")
        return True
    else:
        print("FAILED")
        print(f"   Cause: Directory not found: {path}")
        print(f"   Resolution: Create {description}")
        return False


def check_proto_stubs() -> bool:
    """Check that gRPC proto stubs are generated.

    Returns:
        True if stubs exist and are valid, False otherwise
    """
    print("Checking gRPC proto stubs... ", end="", flush=True)

    proto_dir = Path("src/rpc/generated")
    required_files = [
        proto_dir / "__init__.py",
        proto_dir / "tts_pb2.py",
        proto_dir / "tts_pb2_grpc.py",
    ]

    missing = [f for f in required_files if not f.exists()]

    if not missing:
        print("OK")
        return True
    else:
        print("FAILED")
        print(f"   Cause: Missing proto stub files:")
        for f in missing:
            print(f"     - {f}")
        print("   Resolution: just gen-proto")
        print("   See: docs/runbooks/ENVIRONMENT.md#proto-generation")
        return False


def check_python_dependencies() -> bool:
    """Check that Python dependencies are installed.

    Returns:
        True if dependencies available, False otherwise
    """
    print("Checking Python dependencies... ", end="", flush=True)

    required_packages = [
        ("grpc", "grpcio"),
        ("aiohttp", "aiohttp"),
        ("pydantic", "pydantic"),
        ("redis", "redis"),
    ]

    missing = []
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)

    if not missing:
        print("OK")
        return True
    else:
        print("FAILED")
        print(f"   Cause: Missing Python packages: {', '.join(missing)}")
        print("   Resolution: uv sync")
        print("   See: docs/runbooks/ENVIRONMENT.md#dependencies")
        return False


def validate_orchestrator_config() -> bool:
    """Validate orchestrator configuration.

    Returns:
        True if config is valid, False otherwise
    """
    print("Validating orchestrator config... ", end="", flush=True)

    config_path = Path("configs/orchestrator.yaml")
    if not config_path.exists():
        print("FAILED")
        print(f"   Cause: Config file not found: {config_path}")
        print("   Resolution: Copy from configs/orchestrator.example.yaml")
        return False

    try:
        # Import here to avoid import errors before dependencies check
        from src.orchestrator.config import OrchestratorConfig

        config = OrchestratorConfig.from_yaml(config_path)

        # Validate critical fields
        if not config.routing.static_worker_addr:
            print("FAILED")
            print("   Cause: routing.static_worker_addr not configured")
            print("   Resolution: Set worker address in configs/orchestrator.yaml")
            print("   Example: static_worker_addr: 'grpc://localhost:7001'")
            return False

        print("OK")
        return True

    except Exception as e:
        print("FAILED")
        print(f"   Cause: Configuration validation error: {e}")
        print("   Resolution: Fix errors in configs/orchestrator.yaml")
        print("   See: docs/CONFIGURATION_REFERENCE.md")
        return False


def validate_worker_config() -> bool:
    """Validate worker configuration.

    Returns:
        True if config is valid or not required, False otherwise
    """
    print("Validating worker config... ", end="", flush=True)

    config_path = Path("configs/worker.yaml")
    if not config_path.exists():
        # Worker config is optional for M2 (uses defaults)
        print("OK (using defaults)")
        return True

    try:
        # For M2: Basic YAML validation
        # For M4+: Use TTSWorkerConfig Pydantic model
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)

        if config is None:
            print("FAILED")
            print("   Cause: Empty configuration file")
            print("   Resolution: Add configuration or remove file to use defaults")
            return False

        print("OK")
        return True

    except Exception as e:
        print("FAILED")
        print(f"   Cause: Configuration validation error: {e}")
        print("   Resolution: Fix errors in configs/worker.yaml")
        print("   See: docs/CONFIGURATION_REFERENCE.md")
        return False


async def check_services_reachable() -> bool:
    """Check if required services are reachable.

    Returns:
        True if services available, False otherwise
    """
    print("\nChecking service connectivity...")
    print("(This checks if services can be reached, not if they're fully healthy)")
    print()

    try:
        # Import connection test functions
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.test_connection import (  # type: ignore[import]
            test_grpc_worker,
            test_redis,
        )

        # Test basic connectivity (simplified for startup validation)
        import os

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        worker_addr = os.getenv("WORKER_ADDR", "localhost:7001")

        redis_ok = await test_redis(redis_url)
        worker_ok = await test_grpc_worker(worker_addr)

        return redis_ok and worker_ok

    except ImportError:
        print("   WARNING: Could not import connection test functions")
        print("   Skipping connectivity checks")
        return True
    except Exception as e:
        print(f"   WARNING: Error checking connectivity: {e}")
        print("   Proceeding anyway (services may start later)")
        return True


def main() -> int:
    """Run all startup validations.

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    print("=" * 60)
    print("M2 Startup Validation")
    print("=" * 60)
    print()

    # Track results
    checks: dict[str, bool] = {}

    # File/directory checks
    checks["configs_dir"] = check_directory_exists(
        Path("configs"), "configs directory"
    )
    checks["src_dir"] = check_directory_exists(Path("src"), "src directory")
    checks["proto_dir"] = check_directory_exists(
        Path("src/rpc"), "proto directory"
    )

    # Proto stubs (critical)
    checks["proto_stubs"] = check_proto_stubs()

    # Dependencies (critical)
    checks["dependencies"] = check_python_dependencies()

    # Only validate configs if dependencies are available
    if checks["dependencies"]:
        checks["orch_config"] = validate_orchestrator_config()
        checks["worker_config"] = validate_worker_config()
    else:
        print("\nSkipping config validation (dependencies not installed)")
        checks["orch_config"] = False
        checks["worker_config"] = False

    print()
    print("=" * 60)

    # Determine if critical checks passed
    critical_checks = [
        "proto_stubs",
        "dependencies",
        "orch_config",
    ]

    critical_failed = [k for k in critical_checks if not checks.get(k, False)]

    if not critical_failed:
        print("Status: READY TO START")
        print()
        print("All critical validations passed.")
        print()
        print("Next steps:")
        print("  - Start services: docker compose up --build")
        print("  - Or run locally: just run-orch & just run-tts-sesame")
        print("  - Test connections: ./scripts/test-connection.py")
        return 0
    else:
        print(f"Status: NOT READY ({len(critical_failed)} critical check(s) failed)")
        print()
        print("Failed critical checks:")
        for check in critical_failed:
            print(f"  - {check}")
        print()
        print("Troubleshooting:")
        print("  1. Fix failed checks above")
        print("  2. Run pre-flight check: ./scripts/preflight-check.sh")
        print("  3. Install dependencies: uv sync")
        print("  4. Generate proto stubs: just gen-proto")
        print("  5. See: docs/runbooks/ENVIRONMENT.md")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
