#!/usr/bin/env python3
"""Validate M2 configuration files.

This tool validates both orchestrator.yaml and worker.yaml configuration files
using Pydantic models, catching all configuration errors before runtime.

Exit codes:
    0: All configurations valid
    1: Configuration validation failed
    2: File not found or YAML parse error

Usage:
    ./scripts/validate-config.py
    ./scripts/validate-config.py --orchestrator configs/orchestrator.yaml
    ./scripts/validate-config.py --worker configs/worker.yaml
    ./scripts/validate-config.py --help
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pydantic import ValidationError

try:
    from orchestrator.config import OrchestratorConfig
    from tts.config import TTSWorkerConfig
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("   Resolution: Ensure you're running from the project root and dependencies are installed.")
    print("   Run: uv sync")
    sys.exit(2)


def format_validation_errors(e: ValidationError) -> str:
    """Format Pydantic validation errors in a user-friendly way.

    Args:
        e: Pydantic ValidationError

    Returns:
        Formatted error message with resolution steps
    """
    errors = []
    for error in e.errors():
        loc = ".".join(str(x) for x in error["loc"])
        msg = error["msg"]
        error_type = error["type"]
        errors.append(f"      Field: {loc}")
        errors.append(f"      Error: {msg}")
        errors.append(f"      Type: {error_type}")

        # Add context-specific resolution hints
        if "greater than" in msg.lower() or "less than" in msg.lower():
            errors.append(f"      Resolution: Check valid range in configuration comments")
        elif "not a valid" in msg.lower():
            errors.append(f"      Resolution: Ensure value matches expected type")
        elif "field required" in msg.lower():
            errors.append(f"      Resolution: Add required field '{loc}' to configuration")
        errors.append("")

    return "\n".join(errors)


def validate_orchestrator_config(path: Path, verbose: bool = False) -> bool:
    """Validate orchestrator.yaml configuration.

    Args:
        path: Path to orchestrator.yaml
        verbose: Show full configuration on success

    Returns:
        True if valid, False otherwise
    """
    try:
        config = OrchestratorConfig.from_yaml(path)
        print(f"✅ {path}: Valid orchestrator configuration")

        if verbose:
            print("\n   Configuration loaded successfully:")
            print(f"   - WebSocket: {config.transport.websocket.host}:{config.transport.websocket.port}")
            print(f"   - LiveKit: {'enabled' if config.transport.livekit.enabled else 'disabled'}")
            print(f"   - Redis: {config.redis.url}")
            print(f"   - Worker: {config.routing.static_worker_addr}")
            print(f"   - VAD: {'enabled' if config.vad.enabled else 'disabled'}")
            print(f"   - Log Level: {config.log_level}")

        return True

    except FileNotFoundError:
        print(f"❌ {path}: File not found")
        print(f"   Resolution: Create orchestrator configuration file at {path}")
        print("   Example: cp configs/orchestrator.yaml.example configs/orchestrator.yaml")
        return False

    except ValidationError as e:
        print(f"❌ {path}: Invalid orchestrator configuration")
        print("\n   Validation Errors:")
        print(format_validation_errors(e))
        print("   Resolution: Fix configuration errors listed above")
        print("   Reference: docs/CONFIGURATION_REFERENCE.md")
        return False

    except Exception as e:
        print(f"❌ {path}: YAML parse error")
        print(f"   Error: {e}")
        print("   Resolution: Check YAML syntax (indentation, quotes, colons)")
        print("   Validate YAML: https://www.yamllint.com/")
        return False


def validate_worker_config(path: Path, verbose: bool = False) -> bool:
    """Validate worker.yaml configuration.

    Args:
        path: Path to worker.yaml
        verbose: Show full configuration on success

    Returns:
        True if valid, False otherwise
    """
    try:
        config = TTSWorkerConfig.from_yaml(path)
        print(f"✅ {path}: Valid worker configuration")

        if verbose:
            print("\n   Configuration loaded successfully:")
            print(f"   - Worker: {config.worker.name}")
            print(f"   - gRPC: {config.worker.grpc_host}:{config.worker.grpc_port}")
            print(f"   - Default Model: {config.model_manager.default_model_id}")
            print(f"   - Preload Models: {config.model_manager.preload_model_ids or 'none'}")
            print(f"   - Languages: {', '.join(config.worker.capabilities.languages)}")
            print(f"   - Max Sessions: {config.worker.capabilities.max_concurrent_sessions}")
            print(f"   - Audio: {config.audio.output_sample_rate}Hz, {config.audio.frame_duration_ms}ms frames")
            print(f"   - Redis: {config.redis.url}")
            print(f"   - Metrics: {config.metrics.prometheus_port if config.metrics.enabled else 'disabled'}")

        return True

    except FileNotFoundError:
        print(f"❌ {path}: File not found")
        print(f"   Resolution: Create worker configuration file at {path}")
        print("   Example: cp configs/worker.yaml.example configs/worker.yaml")
        return False

    except ValidationError as e:
        print(f"❌ {path}: Invalid worker configuration")
        print("\n   Validation Errors:")
        print(format_validation_errors(e))
        print("   Resolution: Fix configuration errors listed above")
        print("   Reference: docs/CONFIGURATION_REFERENCE.md")
        return False

    except Exception as e:
        print(f"❌ {path}: YAML parse error")
        print(f"   Error: {e}")
        print("   Resolution: Check YAML syntax (indentation, quotes, colons)")
        print("   Validate YAML: https://www.yamllint.com/")
        return False


def main() -> int:
    """Main entry point for config validation tool.

    Returns:
        Exit code (0=success, 1=validation failed, 2=file error)
    """
    parser = argparse.ArgumentParser(
        description="Validate M2 Realtime Duplex Voice Demo configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate both configs (default)
  ./scripts/validate-config.py

  # Validate specific config
  ./scripts/validate-config.py --orchestrator configs/orchestrator.yaml
  ./scripts/validate-config.py --worker configs/worker.yaml

  # Verbose output
  ./scripts/validate-config.py --verbose

  # Custom paths
  ./scripts/validate-config.py --orchestrator /path/to/orch.yaml --worker /path/to/worker.yaml
        """
    )

    parser.add_argument(
        "--orchestrator",
        type=Path,
        default=Path("configs/orchestrator.yaml"),
        help="Path to orchestrator.yaml (default: configs/orchestrator.yaml)"
    )

    parser.add_argument(
        "--worker",
        type=Path,
        default=Path("configs/worker.yaml"),
        help="Path to worker.yaml (default: configs/worker.yaml)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full configuration details on success"
    )

    args = parser.parse_args()

    print("=== M2 Configuration Validation ===\n")

    # Validate orchestrator config
    orch_ok = validate_orchestrator_config(args.orchestrator, verbose=args.verbose)
    print()

    # Validate worker config
    worker_ok = validate_worker_config(args.worker, verbose=args.verbose)
    print()

    # Summary
    print("=" * 40)
    if orch_ok and worker_ok:
        print("✅ All configurations valid")
        print("\nNext steps:")
        print("  1. Start Redis: just redis")
        print("  2. Generate protos: just gen-proto")
        print("  3. Start worker: just run-tts-sesame")
        print("  4. Start orchestrator: just run-orch")
        return 0
    else:
        print("❌ Configuration validation failed")
        print("\nFailed validations:")
        if not orch_ok:
            print(f"  - {args.orchestrator}")
        if not worker_ok:
            print(f"  - {args.worker}")
        print("\nFix errors above and re-run validation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
