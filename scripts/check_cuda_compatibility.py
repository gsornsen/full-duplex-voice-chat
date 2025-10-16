#!/usr/bin/env python3
"""CUDA/cuDNN compatibility checker.

This script detects CUDA and cuDNN version mismatches before runtime
to help diagnose issues with GPU-accelerated libraries like WhisperX.

Usage:
    python scripts/check_cuda_compatibility.py

    Or via uv:
    uv run python scripts/check_cuda_compatibility.py
"""

import sys
from typing import Optional


def get_nvidia_driver_info() -> tuple[Optional[str], Optional[str]]:
    """Get NVIDIA driver and CUDA driver version from nvidia-smi."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,cuda_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        driver_version, cuda_version = result.stdout.strip().split(", ")
        return driver_version.strip(), cuda_version.strip()
    except Exception as e:
        print(f"⚠️  Could not get NVIDIA driver info: {e}")
        return None, None


def get_pytorch_cuda_info() -> tuple[Optional[str], Optional[str], Optional[int]]:
    """Get PyTorch CUDA and cuDNN versions."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None, None

        cuda_version = torch.version.cuda
        cudnn_version = None

        try:
            import torch.backends.cudnn as cudnn
            if cudnn.is_available():
                cudnn_version = cudnn.version()
        except Exception as e:
            print(f"⚠️  Could not get cuDNN version: {e}")

        return cuda_version, str(cudnn_version), cudnn_version
    except ImportError:
        print("⚠️  PyTorch not installed")
        return None, None, None


def get_ctranslate2_info() -> tuple[bool, Optional[str]]:
    """Get CTranslate2 CUDA support info."""
    try:
        import ctranslate2

        # Check if CUDA is available in CTranslate2
        cuda_available = ctranslate2.get_cuda_device_count() > 0
        version = ctranslate2.__version__
        return cuda_available, version
    except ImportError:
        return False, None


def format_cudnn_version(version: int) -> str:
    """Format cuDNN version number to readable string.

    Example: 91002 -> 9.1.0.2
    """
    major = version // 10000
    minor = (version % 10000) // 100
    patch = version % 100
    return f"{major}.{minor}.{patch // 10}.{patch % 10}"


def check_compatibility() -> int:
    """Check CUDA/cuDNN compatibility and return exit code."""
    print("=" * 80)
    print("CUDA/cuDNN Compatibility Check")
    print("=" * 80)
    print()

    # 1. NVIDIA Driver and CUDA Driver
    print("📊 NVIDIA Driver & CUDA Driver")
    print("-" * 80)
    driver_version, cuda_driver_version = get_nvidia_driver_info()
    if driver_version:
        print(f"  NVIDIA Driver Version:  {driver_version}")
        print(f"  CUDA Driver Version:    {cuda_driver_version}")
    else:
        print("  ❌ NVIDIA driver not found (nvidia-smi not available)")
        print("  This is expected if running on CPU-only system")
    print()

    # 2. PyTorch CUDA and cuDNN
    print("🔥 PyTorch CUDA & cuDNN")
    print("-" * 80)
    pytorch_cuda, cudnn_version_str, cudnn_version_int = get_pytorch_cuda_info()
    if pytorch_cuda:
        print(f"  PyTorch CUDA Version:   {pytorch_cuda}")
        if cudnn_version_int:
            formatted_cudnn = format_cudnn_version(cudnn_version_int)
            print(f"  cuDNN Version:          {formatted_cudnn} (raw: {cudnn_version_int})")
        else:
            print(f"  cuDNN Version:          Not available")

        # Check if PyTorch can access GPU
        try:
            import torch
            gpu_count = torch.cuda.device_count()
            print(f"  GPUs Detected:          {gpu_count}")
            if gpu_count > 0:
                print(f"  GPU 0:                  {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"  ⚠️  Could not query GPU: {e}")
    else:
        print("  ❌ PyTorch CUDA not available")
        print("  PyTorch installed without CUDA support or CUDA not working")
    print()

    # 3. CTranslate2 (used by faster-whisper/WhisperX)
    print("⚡ CTranslate2 (faster-whisper backend)")
    print("-" * 80)
    ct2_cuda, ct2_version = get_ctranslate2_info()
    if ct2_version:
        print(f"  CTranslate2 Version:    {ct2_version}")
        print(f"  CUDA Support:           {'✅ Yes' if ct2_cuda else '❌ No'}")
        if ct2_cuda:
            import ctranslate2
            cuda_devices = ctranslate2.get_cuda_device_count()
            print(f"  CUDA Devices:           {cuda_devices}")
    else:
        print("  ❌ CTranslate2 not installed")
        print("  Install with: pip install ctranslate2")
    print()

    # 4. Compatibility Analysis
    print("🔍 Compatibility Analysis")
    print("-" * 80)

    issues = []
    warnings = []

    # Check if CUDA versions match
    if cuda_driver_version and pytorch_cuda:
        driver_major = int(cuda_driver_version.split(".")[0])
        pytorch_major = int(pytorch_cuda.split(".")[0])

        if driver_major != pytorch_major:
            issues.append(
                f"CUDA version mismatch: Driver={cuda_driver_version}, "
                f"PyTorch={pytorch_cuda}"
            )
        else:
            print(f"  ✅ CUDA versions compatible (Driver: {cuda_driver_version}, PyTorch: {pytorch_cuda})")

    # Check cuDNN availability
    if pytorch_cuda and not cudnn_version_int:
        warnings.append("cuDNN not available in PyTorch (required for some operations)")
    elif cudnn_version_int:
        print(f"  ✅ cuDNN available (version {format_cudnn_version(cudnn_version_int)})")

    # Check CTranslate2 CUDA support
    if pytorch_cuda and ct2_version and not ct2_cuda:
        warnings.append(
            "CTranslate2 installed without CUDA support (WhisperX will be CPU-only)"
        )
    elif ct2_cuda:
        print(f"  ✅ CTranslate2 has CUDA support (WhisperX can use GPU)")

    # WSL2 specific warnings
    try:
        with open("/proc/version", "r") as f:
            if "microsoft" in f.read().lower():
                warnings.append(
                    "WSL2 detected: cuDNN may have compatibility issues. "
                    "Consider using device='cpu' for WhisperX"
                )
                print("  ⚠️  WSL2 environment detected (potential cuDNN issues)")
    except Exception:
        pass

    print()

    # 5. Summary
    print("📝 Summary")
    print("-" * 80)

    if issues:
        print("  ❌ CRITICAL ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
        print()

    if warnings:
        print("  ⚠️  WARNINGS:")
        for warning in warnings:
            print(f"    - {warning}")
        print()

    if not issues and not warnings:
        print("  ✅ No compatibility issues detected!")
        print()

    # Recommendations
    print("💡 Recommendations")
    print("-" * 80)

    if issues or warnings:
        print("  For WhisperX/faster-whisper usage:")
        if "WSL2" in str(warnings):
            print("    1. Use device='cpu' in agent.py to avoid cuDNN issues")
            print("    2. CPU inference is slower but more stable in WSL2")
            print("    3. For GPU: Consider native Linux or proper cuDNN installation")

        if not ct2_cuda and pytorch_cuda:
            print("    1. Install CTranslate2 with CUDA support:")
            print("       pip install ctranslate2 --force-reinstall")
            print("    2. Or accept CPU-only inference")

        if not cudnn_version_int and pytorch_cuda:
            print("    1. Install cuDNN matching your CUDA version")
            print(f"       CUDA {pytorch_cuda} requires cuDNN 9.x")
            print("    2. Or use device='cpu' to avoid cuDNN dependency")
    else:
        print("  System is properly configured for GPU-accelerated inference!")
        print("  You can use device='auto' or device='cuda' in WhisperX")

    print()
    print("=" * 80)

    # Return exit code
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(check_compatibility())
