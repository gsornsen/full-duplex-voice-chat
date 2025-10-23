#!/usr/bin/env python3
"""Benchmark TTS synthesis performance - sequential vs parallel.

This script measures the performance of TTS synthesis with the following goals:
1. Establish baseline sequential synthesis times (current behavior)
2. Test parallel synthesis capability (2-3 concurrent requests)
3. Validate GPU can handle concurrent workload without OOM
4. Provide performance metrics for pipeline optimization

Usage:
    # Sequential benchmark (current behavior)
    python scripts/benchmark_tts_parallel.py --mode sequential

    # Parallel benchmark (2 workers)
    python scripts/benchmark_tts_parallel.py --mode parallel --workers 2

    # Parallel benchmark (3 workers)
    python scripts/benchmark_tts_parallel.py --mode parallel --workers 3

    # Full comparison benchmark
    python scripts/benchmark_tts_parallel.py --mode both --workers 3

Requirements:
    - TTS worker running at localhost:7001
    - CosyVoice2 model loaded (or Piper for CPU baseline)
    - CUDA GPU with sufficient VRAM (for CosyVoice2)

Output:
    - Timing metrics per sentence
    - Total synthesis time
    - Real-Time Factor (RTF) analysis
    - GPU memory usage (if CUDA available)
    - Speedup factor (parallel vs sequential)
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Final

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.plugins.grpc_tts.tts import TTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Test sentences (realistic LLM output from weather query)
TEST_SENTENCES: Final[list[str]] = [
    "In the fall, Salinas, California usually has mild weather.",
    "You can expect temperatures ranging from the mid-50s to mid-70s Fahrenheit.",
    "It's generally dry with occasional light rain starting in late October.",
    "It's a great time to enjoy outdoor activities like hiking and wine tasting.",
    "Need any tips for things to do there?",
]


async def benchmark_sequential(
    worker_address: str,
    model_id: str,
    sentences: list[str] | None = None,
) -> dict:
    """Benchmark sequential TTS synthesis (current behavior).

    Args:
        worker_address: gRPC worker address (e.g., "localhost:7001")
        model_id: Model ID to use (e.g., "cosyvoice2-en-base")
        sentences: List of sentences to synthesize (default: TEST_SENTENCES)

    Returns:
        Dictionary with timing metrics and results
    """
    if sentences is None:
        sentences = TEST_SENTENCES

    logger.info(f"Starting sequential benchmark with {len(sentences)} sentences")

    tts = TTS(worker_address=worker_address, model_id=model_id)

    total_start = time.perf_counter()
    sentence_times = []
    total_frames = 0
    total_audio_duration = 0.0

    try:
        for i, sentence in enumerate(sentences):
            logger.info(f"Synthesizing sentence {i+1}/{len(sentences)}: {sentence[:50]}...")

            sent_start = time.perf_counter()
            frames = []

            # Synthesize and collect frames
            async for frame_data in tts.synthesize_streaming(sentence):
                frames.append(frame_data)

            sent_time = time.perf_counter() - sent_start

            # Calculate audio duration (frames are 20ms @ 48kHz = 1920 bytes each)
            frame_count = len(frames)
            audio_duration_s = frame_count * 0.02  # 20ms per frame

            # Calculate RTF (Real-Time Factor)
            rtf = sent_time / audio_duration_s if audio_duration_s > 0 else 0

            sentence_times.append({
                "sentence_id": i + 1,
                "text": sentence,
                "synthesis_time_s": sent_time,
                "frame_count": frame_count,
                "audio_duration_s": audio_duration_s,
                "rtf": rtf,
            })

            total_frames += frame_count
            total_audio_duration += audio_duration_s

            logger.info(
                f"  Sentence {i+1}: {sent_time:.2f}s synthesis, "
                f"{frame_count} frames, {audio_duration_s:.2f}s audio, RTF={rtf:.3f}"
            )

    finally:
        await tts.aclose()

    total_time = time.perf_counter() - total_start
    overall_rtf = total_time / total_audio_duration if total_audio_duration > 0 else 0

    logger.info(f"\nSequential benchmark complete:")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Total audio: {total_audio_duration:.2f}s")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Overall RTF: {overall_rtf:.3f}")

    return {
        "mode": "sequential",
        "total_time_s": total_time,
        "total_audio_duration_s": total_audio_duration,
        "total_frames": total_frames,
        "overall_rtf": overall_rtf,
        "sentence_count": len(sentences),
        "sentence_times": sentence_times,
    }


async def synthesize_sentence(
    worker_address: str,
    model_id: str,
    sentence: str,
    sentence_id: int,
) -> dict:
    """Synthesize a single sentence (used for parallel benchmarking).

    Args:
        worker_address: gRPC worker address
        model_id: Model ID to use
        sentence: Text to synthesize
        sentence_id: Sentence identifier for logging

    Returns:
        Dictionary with timing metrics for this sentence
    """
    logger.info(f"[Worker {sentence_id}] Synthesizing: {sentence[:50]}...")

    # Create dedicated TTS instance for this sentence
    tts = TTS(worker_address=worker_address, model_id=model_id)

    sent_start = time.perf_counter()
    frames = []

    try:
        # Synthesize and collect frames
        async for frame_data in tts.synthesize_streaming(sentence):
            frames.append(frame_data)

    finally:
        await tts.aclose()

    sent_time = time.perf_counter() - sent_start

    # Calculate audio duration
    frame_count = len(frames)
    audio_duration_s = frame_count * 0.02  # 20ms per frame

    # Calculate RTF
    rtf = sent_time / audio_duration_s if audio_duration_s > 0 else 0

    logger.info(
        f"[Worker {sentence_id}] Complete: {sent_time:.2f}s synthesis, "
        f"{frame_count} frames, {audio_duration_s:.2f}s audio, RTF={rtf:.3f}"
    )

    return {
        "sentence_id": sentence_id,
        "text": sentence,
        "synthesis_time_s": sent_time,
        "frame_count": frame_count,
        "audio_duration_s": audio_duration_s,
        "rtf": rtf,
    }


async def benchmark_parallel(
    worker_address: str,
    model_id: str,
    sentences: list[str] | None = None,
    num_workers: int = 2,
) -> dict:
    """Benchmark parallel TTS synthesis with multiple concurrent requests.

    Args:
        worker_address: gRPC worker address
        model_id: Model ID to use
        sentences: List of sentences to synthesize (default: TEST_SENTENCES)
        num_workers: Number of parallel workers (2-3 recommended)

    Returns:
        Dictionary with timing metrics and results
    """
    if sentences is None:
        sentences = TEST_SENTENCES

    logger.info(
        f"Starting parallel benchmark with {len(sentences)} sentences, "
        f"{num_workers} workers"
    )

    total_start = time.perf_counter()

    # Process sentences in batches of num_workers
    sentence_times = []
    total_frames = 0
    total_audio_duration = 0.0

    # Process sentences in parallel batches
    for batch_start in range(0, len(sentences), num_workers):
        batch_end = min(batch_start + num_workers, len(sentences))
        batch = sentences[batch_start:batch_end]

        logger.info(
            f"Processing batch {batch_start // num_workers + 1}: "
            f"sentences {batch_start + 1}-{batch_end}"
        )

        # Create parallel tasks for this batch
        tasks = [
            synthesize_sentence(
                worker_address=worker_address,
                model_id=model_id,
                sentence=sentence,
                sentence_id=batch_start + i + 1,
            )
            for i, sentence in enumerate(batch)
        ]

        # Execute batch in parallel
        batch_results = await asyncio.gather(*tasks)

        # Collect results
        for result in batch_results:
            sentence_times.append(result)
            total_frames += result["frame_count"]
            total_audio_duration += result["audio_duration_s"]

    total_time = time.perf_counter() - total_start
    overall_rtf = total_time / total_audio_duration if total_audio_duration > 0 else 0

    logger.info(f"\nParallel benchmark complete ({num_workers} workers):")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Total audio: {total_audio_duration:.2f}s")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Overall RTF: {overall_rtf:.3f}")

    return {
        "mode": "parallel",
        "num_workers": num_workers,
        "total_time_s": total_time,
        "total_audio_duration_s": total_audio_duration,
        "total_frames": total_frames,
        "overall_rtf": overall_rtf,
        "sentence_count": len(sentences),
        "sentence_times": sentence_times,
    }


async def benchmark_both(
    worker_address: str,
    model_id: str,
    sentences: list[str] | None = None,
    num_workers: int = 2,
) -> dict:
    """Run both sequential and parallel benchmarks for comparison.

    Args:
        worker_address: gRPC worker address
        model_id: Model ID to use
        sentences: List of sentences to synthesize
        num_workers: Number of parallel workers

    Returns:
        Dictionary with comparison metrics
    """
    logger.info("Running comparative benchmark (sequential + parallel)")

    # Run sequential benchmark
    sequential_results = await benchmark_sequential(
        worker_address=worker_address,
        model_id=model_id,
        sentences=sentences,
    )

    # Small delay to allow GPU to stabilize
    await asyncio.sleep(2.0)

    # Run parallel benchmark
    parallel_results = await benchmark_parallel(
        worker_address=worker_address,
        model_id=model_id,
        sentences=sentences,
        num_workers=num_workers,
    )

    # Calculate speedup
    speedup = (
        sequential_results["total_time_s"] / parallel_results["total_time_s"]
        if parallel_results["total_time_s"] > 0
        else 0
    )

    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Sequential: {sequential_results['total_time_s']:.2f}s")
    logger.info(f"Parallel ({num_workers} workers): {parallel_results['total_time_s']:.2f}s")
    logger.info(f"Speedup: {speedup:.2f}x")
    logger.info(f"Sequential RTF: {sequential_results['overall_rtf']:.3f}")
    logger.info(f"Parallel RTF: {parallel_results['overall_rtf']:.3f}")
    logger.info("=" * 80)

    return {
        "sequential": sequential_results,
        "parallel": parallel_results,
        "speedup": speedup,
    }


async def check_gpu_memory() -> dict | None:
    """Check GPU memory usage (if CUDA available).

    Returns:
        Dictionary with GPU memory stats, or None if CUDA unavailable
    """
    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("CUDA not available - GPU memory check skipped")
            return None

        device = torch.device("cuda:0")
        memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024 / 1024  # MB
        memory_total = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024  # MB

        logger.info(f"\nGPU Memory Usage:")
        logger.info(f"  Allocated: {memory_allocated:.1f} MB")
        logger.info(f"  Reserved: {memory_reserved:.1f} MB")
        logger.info(f"  Total: {memory_total:.1f} MB")
        logger.info(
            f"  Utilization: {(memory_allocated / memory_total * 100):.1f}%"
        )

        return {
            "allocated_mb": memory_allocated,
            "reserved_mb": memory_reserved,
            "total_mb": memory_total,
            "utilization_pct": (memory_allocated / memory_total * 100),
        }

    except ImportError:
        logger.warning("PyTorch not available - GPU memory check skipped")
        return None


def main() -> int:
    """Main entry point for benchmark script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Benchmark TTS synthesis performance (sequential vs parallel)"
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "parallel", "both"],
        default="both",
        help="Benchmark mode (default: both)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )
    parser.add_argument(
        "--worker-address",
        default="localhost:7001",
        help="gRPC worker address (default: localhost:7001)",
    )
    parser.add_argument(
        "--model-id",
        default="cosyvoice2-en-base",
        help="Model ID to use (default: cosyvoice2-en-base)",
    )
    parser.add_argument(
        "--sentences",
        nargs="+",
        help="Custom sentences to synthesize (default: built-in test sentences)",
    )

    args = parser.parse_args()

    # Check GPU memory before benchmark
    logger.info("Checking GPU memory before benchmark...")
    asyncio.run(check_gpu_memory())

    # Run benchmark
    try:
        if args.mode == "sequential":
            results = asyncio.run(
                benchmark_sequential(
                    worker_address=args.worker_address,
                    model_id=args.model_id,
                    sentences=args.sentences,
                )
            )
        elif args.mode == "parallel":
            results = asyncio.run(
                benchmark_parallel(
                    worker_address=args.worker_address,
                    model_id=args.model_id,
                    sentences=args.sentences,
                    num_workers=args.workers,
                )
            )
        else:  # both
            results = asyncio.run(
                benchmark_both(
                    worker_address=args.worker_address,
                    model_id=args.model_id,
                    sentences=args.sentences,
                    num_workers=args.workers,
                )
            )

        # Check GPU memory after benchmark
        logger.info("\nChecking GPU memory after benchmark...")
        asyncio.run(check_gpu_memory())

        logger.info("\nBenchmark completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\nBenchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
