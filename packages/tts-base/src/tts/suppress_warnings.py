"""Suppress harmless third-party library warnings for cleaner logs.

This module suppresses known warnings from:
- Pydantic (protected namespace conflicts)
- Diffusers (deprecation warnings)
- PyTorch (weight_norm deprecation)
- ONNXRuntime (optimization messages)
- DeepSpeed (verbose logging)
- gRPC (trace configuration)
"""

import logging
import os
import warnings


def suppress_third_party_warnings() -> None:
    """Suppress known harmless warnings from third-party libraries.

    Call this at module import to clean up log output.
    """
    # Suppress diffusers deprecation warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
    warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear.*")
    warnings.filterwarnings("ignore", message=".*Sliding Window Attention.*")

    # Suppress PyTorch deprecations
    warnings.filterwarnings("ignore", message=".*weight_norm is deprecated.*")

    # Suppress pydantic warnings
    warnings.filterwarnings("ignore", message=".*protected namespace.*")

    # Suppress ONNXRuntime verbose warnings
    logging.getLogger("onnxruntime").setLevel(logging.ERROR)

    # Suppress DeepSpeed info logs
    logging.getLogger("deepspeed").setLevel(logging.WARNING)

    # Set gRPC environment variables (suppress trace warnings)
    os.environ.setdefault("GRPC_TRACE", "")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

    logging.info("Third-party warnings suppressed for cleaner logs")


# Auto-suppress on import
suppress_third_party_warnings()
