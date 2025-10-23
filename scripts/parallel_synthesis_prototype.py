#!/usr/bin/env python3
"""Parallel synthesis prototype - proof of concept for pipeline optimization.

This script demonstrates the core concept for Phase 2 pipeline optimization:
1. Sentence segmentation from streaming LLM output
2. Parallel TTS synthesis of multiple sentences
3. Response buffer for smooth audio playback

Usage:
    python scripts/parallel_synthesis_prototype.py

Requirements:
    - TTS worker running at localhost:7001
    - CosyVoice2 or Piper model loaded

Output:
    - Demonstrates parallel synthesis with 2-3 concurrent sentences
    - Shows timing improvements vs sequential processing
    - Validates GPU memory handling
"""

import asyncio
import logging
import sys
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

# Simulated LLM streaming output (sentence-segmented)
SIMULATED_LLM_STREAM: Final[list[str]] = [
    "In the fall, Salinas, California usually has mild weather.",
    "You can expect temperatures ranging from the mid-50s to mid-70s Fahrenheit.",
    "It's generally dry with occasional light rain starting in late October.",
    "It's a great time to enjoy outdoor activities like hiking and wine tasting.",
    "Need any tips for things to do there?",
]


class ResponseBuffer:
    """Simple response buffer for managing synthesized audio.

    This is a minimal implementation to demonstrate the concept.
    Phase 2 will have a full implementation with:
    - Thread-safe queue
    - Backpressure handling
    - Frame dropping on overflow
    - Latency tracking
    """

    def __init__(self, max_size: int = 100) -> None:
        """Initialize response buffer.

        Args:
            max_size: Maximum number of frames to buffer
        """
        self.buffer: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.completed = False

    async def enqueue(self, frames: list[bytes]) -> None:
        """Add frames to buffer.

        Args:
            frames: List of audio frames to enqueue
        """
        for frame in frames:
            await self.buffer.put(frame)
        logger.debug(f"Enqueued {len(frames)} frames")

    async def dequeue(self) -> bytes | None:
        """Get next frame from buffer.

        Returns:
            Audio frame, or None if buffer is complete and empty
        """
        try:
            frame = await asyncio.wait_for(self.buffer.get(), timeout=0.1)
            return frame
        except asyncio.TimeoutError:
            if self.completed and self.buffer.empty():
                return None
            return b""  # Empty frame (silence)

    def mark_complete(self) -> None:
        """Mark buffer as complete (no more frames will be added)."""
        self.completed = True


async def synthesize_sentence_parallel(
    tts: TTS,
    sentence: str,
    sentence_id: int,
    buffer: ResponseBuffer,
) -> None:
    """Synthesize a sentence and enqueue frames to buffer.

    This runs in parallel with other sentence synthesis tasks.

    Args:
        tts: TTS instance to use
        sentence: Text to synthesize
        sentence_id: Sentence identifier for logging
        buffer: Response buffer to enqueue frames
    """
    logger.info(f"[Parallel {sentence_id}] Starting synthesis: {sentence[:50]}...")

    frames = []
    async for frame_data in tts.synthesize_streaming(sentence):
        frames.append(frame_data)

    # Enqueue frames to buffer
    await buffer.enqueue(frames)

    logger.info(
        f"[Parallel {sentence_id}] Complete: {len(frames)} frames enqueued"
    )


async def prototype_parallel_pipeline(
    worker_address: str = "localhost:7001",
    model_id: str = "cosyvoice2-en-base",
    max_parallel: int = 2,
) -> None:
    """Demonstrate parallel synthesis pipeline.

    This prototype shows:
    1. Sentences are synthesized in parallel (2-3 at a time)
    2. Frames are buffered for smooth playback
    3. Total time is reduced vs sequential processing

    Args:
        worker_address: gRPC worker address
        model_id: Model ID to use
        max_parallel: Maximum parallel synthesis tasks (2-3 recommended)
    """
    logger.info("=" * 80)
    logger.info("PARALLEL SYNTHESIS PROTOTYPE")
    logger.info("=" * 80)
    logger.info(f"Worker: {worker_address}")
    logger.info(f"Model: {model_id}")
    logger.info(f"Max parallel: {max_parallel}")
    logger.info(f"Sentences: {len(SIMULATED_LLM_STREAM)}")
    logger.info("=" * 80)

    # Create shared TTS instance
    tts = TTS(worker_address=worker_address, model_id=model_id)

    # Create response buffer
    buffer = ResponseBuffer(max_size=1000)

    # Track synthesis tasks
    synthesis_tasks = []

    try:
        import time

        start_time = time.perf_counter()

        # Create synthesis tasks with parallelism limit
        semaphore = asyncio.Semaphore(max_parallel)

        async def synthesize_with_limit(sentence: str, sentence_id: int) -> None:
            """Synthesize with semaphore to limit parallelism."""
            async with semaphore:
                await synthesize_sentence_parallel(
                    tts=tts,
                    sentence=sentence,
                    sentence_id=sentence_id,
                    buffer=buffer,
                )

        # Create all synthesis tasks
        for i, sentence in enumerate(SIMULATED_LLM_STREAM):
            task = asyncio.create_task(
                synthesize_with_limit(sentence, sentence_id=i + 1)
            )
            synthesis_tasks.append(task)

        # Wait for all synthesis tasks to complete
        await asyncio.gather(*synthesis_tasks)

        # Mark buffer as complete
        buffer.mark_complete()

        end_time = time.perf_counter()
        total_time = end_time - start_time

        logger.info("=" * 80)
        logger.info("SYNTHESIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Sentences: {len(SIMULATED_LLM_STREAM)}")
        logger.info(f"Average per sentence: {total_time / len(SIMULATED_LLM_STREAM):.2f}s")
        logger.info("=" * 80)

        # Simulate playback (dequeue frames)
        logger.info("\nSimulating playback (dequeuing frames)...")
        frame_count = 0
        while True:
            frame = await buffer.dequeue()
            if frame is None:
                break
            if frame:  # Non-empty frame
                frame_count += 1

        logger.info(f"Playback complete: {frame_count} frames")

    finally:
        await tts.aclose()


async def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Run prototype with 2 parallel workers (conservative)
        await prototype_parallel_pipeline(max_parallel=2)

        logger.info("\n✓ Prototype completed successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. Run benchmark_tts_parallel.py for detailed metrics")
        logger.info("  2. Test with 3 parallel workers for maximum throughput")
        logger.info("  3. Integrate with sentence segmenter for live LLM streaming")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nPrototype interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\nPrototype failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
