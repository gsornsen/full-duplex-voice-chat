#!/usr/bin/env python3
"""WhisperX GPU test in Docker container.

This script tests if WhisperX/faster-whisper can successfully use GPU
in a Docker container with NVIDIA runtime on WSL2.

Tests:
1. Environment detection (CUDA, cuDNN, GPU availability)
2. Model loading on GPU
3. Sample transcription
4. Performance benchmarking (RTF comparison)
"""

import time
import sys
import numpy as np

def print_header(title: str) -> None:
    """Print formatted section header."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def check_environment() -> bool:
    """Check CUDA/cuDNN environment."""
    print_header("Environment Check")

    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ cuDNN version: {torch.backends.cudnn.version()}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            print(f"✅ GPU 0: {torch.cuda.get_device_name(0)}")

            # Test GPU memory access
            torch.cuda.init()
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("❌ CUDA not available")
            return False

    except Exception as e:
        print(f"❌ Error checking environment: {e}")
        return False


def check_ctranslate2() -> bool:
    """Check CTranslate2 CUDA support."""
    print_header("CTranslate2 Check")

    try:
        import ctranslate2
        print(f"✅ CTranslate2 version: {ctranslate2.__version__}")

        cuda_devices = ctranslate2.get_cuda_device_count()
        print(f"✅ CUDA devices detected: {cuda_devices}")

        if cuda_devices > 0:
            return True
        else:
            print("❌ No CUDA devices detected by CTranslate2")
            return False

    except Exception as e:
        print(f"❌ Error checking CTranslate2: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading(device: str = "cuda") -> bool:
    """Test loading WhisperX model on specified device."""
    print_header(f"Model Loading Test (device={device})")

    try:
        from faster_whisper import WhisperModel

        print(f"Loading Whisper small model on {device}...")
        start_time = time.time()

        model = WhisperModel(
            "small",
            device=device,
            compute_type="float16" if device == "cuda" else "int8"
        )

        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f}s")

        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_test_audio(duration_seconds: float = 10.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate test audio (sine wave with speech-like characteristics)."""
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples)

    # Mix of frequencies to simulate speech
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
        0.3 * np.sin(2 * np.pi * 800 * t) +  # Mid frequency
        0.2 * np.sin(2 * np.pi * 2000 * t)   # High frequency
    )

    # Add some amplitude modulation
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    audio = audio * modulation

    # Normalize to -1 to 1 range
    audio = audio / np.max(np.abs(audio))

    return audio.astype(np.float32)


def benchmark_transcription(device: str, num_runs: int = 3) -> dict:
    """Benchmark transcription performance."""
    print_header(f"Transcription Benchmark (device={device})")

    try:
        from faster_whisper import WhisperModel

        # Load model
        print(f"Loading model on {device}...")
        model = WhisperModel(
            "small",
            device=device,
            compute_type="float16" if device == "cuda" else "int8"
        )

        # Generate test audio (10 seconds)
        audio_duration = 10.0
        print(f"Generating {audio_duration}s test audio...")
        audio = generate_test_audio(duration_seconds=audio_duration, sample_rate=16000)

        # Warmup run
        print("Warmup run...")
        segments, info = model.transcribe(audio, language="en", beam_size=5)
        _ = list(segments)  # Consume generator

        # Benchmark runs
        print(f"Running {num_runs} benchmark iterations...")
        timings = []

        for i in range(num_runs):
            start_time = time.time()
            segments, info = model.transcribe(audio, language="en", beam_size=5)
            _ = list(segments)  # Consume generator
            elapsed = time.time() - start_time
            timings.append(elapsed)

            rtf = elapsed / audio_duration
            print(f"  Run {i+1}: {elapsed:.3f}s (RTF: {rtf:.3f})")

        # Calculate statistics
        avg_time = np.mean(timings)
        std_time = np.std(timings)
        min_time = np.min(timings)
        max_time = np.max(timings)
        avg_rtf = avg_time / audio_duration

        print()
        print(f"Results for {audio_duration}s audio on {device}:")
        print(f"  Average time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"  Min time:     {min_time:.3f}s")
        print(f"  Max time:     {max_time:.3f}s")
        print(f"  Average RTF:  {avg_rtf:.3f}")
        print(f"  {'✅ Faster than realtime' if avg_rtf < 1.0 else '⚠️  Slower than realtime'}")

        return {
            "device": device,
            "audio_duration": audio_duration,
            "avg_time": avg_time,
            "std_time": std_time,
            "min_time": min_time,
            "max_time": max_time,
            "avg_rtf": avg_rtf,
            "success": True
        }

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "device": device,
            "success": False,
            "error": str(e)
        }


def main() -> int:
    """Run all tests."""
    print("=" * 80)
    print("WhisperX GPU Test in Docker Container (WSL2)")
    print("=" * 80)

    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed - CUDA not available")
        return 1

    # Check CTranslate2
    if not check_ctranslate2():
        print("\n❌ CTranslate2 check failed - CUDA support missing")
        return 1

    # Test model loading
    if not test_model_loading(device="cuda"):
        print("\n❌ Model loading failed on GPU")
        return 1

    print("\n✅ All checks passed! GPU is accessible.")

    # Benchmark GPU performance
    gpu_results = benchmark_transcription(device="cuda", num_runs=3)

    if not gpu_results["success"]:
        print("\n❌ GPU benchmark failed")
        return 1

    # Summary
    print_header("Summary")
    print("✅ WhisperX successfully running on GPU in Docker container!")
    print(f"✅ Performance: RTF {gpu_results['avg_rtf']:.3f} ({gpu_results['avg_time']:.3f}s for {gpu_results['audio_duration']:.1f}s audio)")
    print()
    print("📊 Comparison to CPU baseline:")
    print("  CPU (WSL2 native): RTF ~0.095")
    print(f"  GPU (Docker):      RTF {gpu_results['avg_rtf']:.3f}")
    print(f"  Speedup:           {0.095 / gpu_results['avg_rtf']:.1f}x faster")
    print()
    print("💡 Recommendation:")
    if gpu_results['avg_rtf'] < 0.07:  # Significantly faster than CPU
        print("  ✅ GPU performance is excellent! Consider using GPU in Docker for production.")
    elif gpu_results['avg_rtf'] < 0.095:  # Faster but not dramatically
        print("  ⚠️  GPU is faster but not dramatically. CPU may be acceptable.")
    else:
        print("  ❌ GPU is slower than CPU. Continue using CPU.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
