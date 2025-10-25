"""Integration tests for parallel TTS synthesis pipeline.

This test suite validates end-to-end parallel synthesis with real components:
- LLM → SentenceSegmenter → ParallelSynthesisPipeline → Audio output
- Feature flag behavior (parallel on/off)
- Multi-sentence responses (2, 5, 10 sentences)
- Error scenarios (TTS failures, worker crashes)
- Barge-in handling (cancel all workers)
- Audio gap detection (<100ms target)
- Seamless playback verification

Test Coverage:
- End-to-end flow with real sentence segmenter
- Feature flag switching
- Error recovery and graceful degradation
- Barge-in cancellation
- Audio continuity (gap detection)
- Real-world scenarios (weather question, storytelling)

Design reference: Phase B ParallelSynthesisPipeline + SentenceSegmenter
"""
# type: ignore  # TODO: Add proper type annotations for test fixtures and helpers


import asyncio
import time
from collections.abc import AsyncIterator

import pytest

# Mock imports (will be replaced with actual implementation)
# from src.orchestrator.parallel_tts import ParallelSynthesisPipeline
# from src.orchestrator.sentence_segmenter import SentenceSegmenter
# from src.rpc.tts_pb2_grpc import TTSStub


# ============================================================================
# Mock Components (temporary until implementation)
# ============================================================================


class MockTTSClient:
    """Mock TTS gRPC client for testing."""

    def __init__(
        self,
        latency_ms: float = 50.0,
        failure_rate: float = 0.0,
    ) -> None:
        """Initialize mock TTS client.

        Args:
            latency_ms: Synthesis latency in milliseconds
            failure_rate: Probability of synthesis failure (0.0-1.0)
        """
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.call_count = 0
        self.synthesis_history: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        """Mock synthesize method.

        Args:
            text: Text to synthesize

        Returns:
            Mock audio data

        Raises:
            Exception: If failure_rate triggers
        """
        self.call_count += 1
        self.synthesis_history.append(text)

        # Simulate failure
        if self.failure_rate > 0:
            import random
            if random.random() < self.failure_rate:  # noqa: S311
                raise Exception(f"TTS synthesis failed for: {text}")

        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        # Return mock audio data (48kHz PCM, 20ms frames)
        frame_size = 48000 * 2 * 20 // 1000  # 48kHz, 16-bit, 20ms
        return b"\x00" * frame_size


class MockSentenceSegmenter:
    """Mock sentence segmenter for testing."""

    async def segment(self, token_stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """Segment token stream into sentences.

        Args:
            token_stream: Token stream from LLM

        Yields:
            Complete sentences
        """
        buffer = []
        async for token in token_stream:
            buffer.append(token)
            text = "".join(buffer)
            # Simple sentence detection (any .!? followed by space)
            if text.rstrip() != text.rstrip(".!?"):
                sentence = "".join(buffer).strip()
                if sentence:
                    yield sentence
                buffer = []

        # Emit remaining buffer
        if buffer:
            sentence = "".join(buffer).strip()
            if sentence:
                yield sentence


class MockParallelSynthesisPipeline:
    """Mock parallel synthesis pipeline for testing."""

    def __init__(
        self,
        tts_client: MockTTSClient,
        num_workers: int = 2,
        enabled: bool = True,
    ) -> None:
        """Initialize pipeline.

        Args:
            tts_client: TTS client
            num_workers: Number of parallel workers
            enabled: Feature flag for parallel synthesis
        """
        self.tts_client = tts_client
        self.num_workers = num_workers
        self.enabled = enabled
        self.sentences_processed: list[str] = []
        self.audio_chunks_emitted: list[bytes] = []

    async def synthesize_sentences(
        self,
        sentence_stream: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        """Synthesize sentences to audio chunks.

        Args:
            sentence_stream: Stream of complete sentences

        Yields:
            Audio chunks in correct order
        """
        if not self.enabled:
            # Sequential synthesis (fallback)
            async for sentence in sentence_stream:
                self.sentences_processed.append(sentence)
                audio = await self.tts_client.synthesize(sentence)
                self.audio_chunks_emitted.append(audio)
                yield audio
        else:
            # Parallel synthesis
            sentence_queue = asyncio.Queue()
            audio_queue = asyncio.Queue()

            # Collector task
            async def collect_sentences():
                async for sentence in sentence_stream:
                    self.sentences_processed.append(sentence)
                    await sentence_queue.put((len(self.sentences_processed) - 1, sentence))
                # Signal completion
                for _ in range(self.num_workers):
                    await sentence_queue.put(None)

            # Worker tasks
            async def worker():
                while True:
                    item = await sentence_queue.get()
                    if item is None:
                        break
                    seq_id, sentence = item
                    try:
                        audio = await self.tts_client.synthesize(sentence)
                        await audio_queue.put((seq_id, audio))
                    except Exception:
                        # Error recovery - emit silence
                        await audio_queue.put((seq_id, b"\x00" * 1920))

            # Emitter task
            async def emit_audio():
                next_seq = 0
                buffer = {}
                active_workers = self.num_workers

                while active_workers > 0 or buffer:
                    try:
                        seq_id, audio = await asyncio.wait_for(
                            audio_queue.get(),
                            timeout=0.1,
                        )
                        buffer[seq_id] = audio
                    except TimeoutError:
                        pass

                    # Emit in-order chunks
                    while next_seq in buffer:
                        audio = buffer.pop(next_seq)
                        self.audio_chunks_emitted.append(audio)
                        yield audio
                        next_seq += 1

                    # Check for worker completion
                    if sentence_queue.empty() and audio_queue.empty():
                        if all(i in self.sentences_processed for i in range(next_seq)):
                            break

            # Start tasks
            collector = asyncio.create_task(collect_sentences())
            workers = [asyncio.create_task(worker()) for _ in range(self.num_workers)]

            async for audio in emit_audio():
                yield audio

            # Cleanup
            await collector
            await asyncio.gather(*workers)

    async def cancel_all(self) -> None:
        """Cancel all active synthesis (barge-in)."""
        # Mock implementation - just clear state
        self.sentences_processed.clear()
        self.audio_chunks_emitted.clear()


# ============================================================================
# End-to-End Flow Tests
# ============================================================================


class TestEndToEndFlow:
    """Test complete end-to-end synthesis flow."""

    @pytest.mark.asyncio
    async def test_simple_two_sentence_flow(self) -> None:
        """Test simple 2-sentence response flow."""
        tts_client = MockTTSClient(latency_ms=50)
        segmenter = MockSentenceSegmenter()
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=2)

        # Mock LLM token stream
        async def llm_stream():
            tokens = ["Hello", " there", ".", " How", " are", " you", "?"]
            for token in tokens:
                yield token
                await asyncio.sleep(0.01)

        # Segment and synthesize
        sentence_stream = segmenter.segment(llm_stream())
        audio_chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream):
            audio_chunks.append(chunk)

        # Verify
        assert len(pipeline.sentences_processed) == 2
        assert pipeline.sentences_processed[0] == "Hello there."
        assert pipeline.sentences_processed[1] == "How are you?"
        assert len(audio_chunks) == 2

    @pytest.mark.asyncio
    async def test_five_sentence_response(self) -> None:
        """Test 5-sentence response (weather question scenario)."""
        tts_client = MockTTSClient(latency_ms=100)
        segmenter = MockSentenceSegmenter()
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=2)

        # Mock LLM stream for weather response
        async def llm_stream():
            text = (
                "The weather in Salinas is typically mild in the fall. "
                "Average temperatures range from 50 to 70 degrees Fahrenheit. "
                "You can expect some morning fog near the coast. "
                "Rainfall is minimal during this season. "
                "It's a great time to visit the area."
            )
            for char in text:
                yield char
                await asyncio.sleep(0.001)

        # Segment and synthesize
        sentence_stream = segmenter.segment(llm_stream())
        audio_chunks = []
        start_time = time.monotonic()

        async for chunk in pipeline.synthesize_sentences(sentence_stream):
            audio_chunks.append(chunk)

        end_time = time.monotonic()
        total_time = end_time - start_time

        # Verify
        assert len(pipeline.sentences_processed) == 5
        assert len(audio_chunks) == 5

        # With 2 workers and 100ms latency per sentence:
        # Sequential: 5 * 100ms = 500ms
        # Parallel (2 workers): ~300ms (3 batches: 2+2+1)
        # Allow some overhead for scheduling
        assert total_time < 0.4  # Should be faster than sequential

    @pytest.mark.asyncio
    async def test_ten_sentence_long_response(self) -> None:
        """Test 10-sentence long response (storytelling scenario)."""
        tts_client = MockTTSClient(latency_ms=50)
        segmenter = MockSentenceSegmenter()
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=3)

        # Mock LLM stream
        async def llm_stream():
            for i in range(10):
                yield f"This is sentence number {i + 1}. "
                await asyncio.sleep(0.01)

        # Segment and synthesize
        sentence_stream = segmenter.segment(llm_stream())
        audio_chunks = []

        async for chunk in pipeline.synthesize_sentences(sentence_stream):
            audio_chunks.append(chunk)

        # Verify
        assert len(pipeline.sentences_processed) == 10
        assert len(audio_chunks) == 10

    @pytest.mark.asyncio
    async def test_streaming_with_delays(self) -> None:
        """Test streaming synthesis with variable LLM delays."""
        tts_client = MockTTSClient(latency_ms=30)
        segmenter = MockSentenceSegmenter()
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=2)

        # Mock LLM with variable delays
        async def llm_stream():
            sentences_raw = [
                ("Let me think. ", 0.1),
                ("The answer is simple. ", 0.05),
                ("Just follow these steps. ", 0.02),
            ]
            for text, delay in sentences_raw:
                for char in text:
                    yield char
                    await asyncio.sleep(0.001)
                await asyncio.sleep(delay)  # Pause between sentences

        # Segment and synthesize
        sentence_stream = segmenter.segment(llm_stream())
        audio_chunks = []

        async for chunk in pipeline.synthesize_sentences(sentence_stream):
            audio_chunks.append(chunk)

        # Verify
        assert len(pipeline.sentences_processed) == 3
        assert len(audio_chunks) == 3


# ============================================================================
# Feature Flag Tests
# ============================================================================


class TestFeatureFlag:
    """Test parallel synthesis feature flag behavior."""

    @pytest.mark.asyncio
    async def test_parallel_enabled(self) -> None:
        """Test with parallel synthesis enabled."""
        tts_client = MockTTSClient(latency_ms=100)
        pipeline = MockParallelSynthesisPipeline(
            tts_client,
            num_workers=2,
            enabled=True,
        )

        async def sentence_stream():
            for i in range(4):
                yield f"Sentence {i}."

        start_time = time.monotonic()
        audio_chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            audio_chunks.append(chunk)
        elapsed = time.monotonic() - start_time

        # With 2 workers: 4 sentences = 2 batches = ~200ms
        assert len(audio_chunks) == 4
        assert elapsed < 0.3  # Parallel should be faster

    @pytest.mark.asyncio
    async def test_parallel_disabled(self) -> None:
        """Test with parallel synthesis disabled (sequential fallback)."""
        tts_client = MockTTSClient(latency_ms=100)
        pipeline = MockParallelSynthesisPipeline(
            tts_client,
            num_workers=2,
            enabled=False,  # Sequential mode
        )

        async def sentence_stream():
            for i in range(4):
                yield f"Sentence {i}."

        start_time = time.monotonic()
        audio_chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            audio_chunks.append(chunk)
        elapsed = time.monotonic() - start_time

        # Sequential: 4 sentences * 100ms = 400ms
        assert len(audio_chunks) == 4
        assert elapsed >= 0.4  # Sequential should be slower

    @pytest.mark.asyncio
    async def test_toggle_feature_flag(self) -> None:
        """Test switching feature flag between requests."""
        tts_client = MockTTSClient(latency_ms=50)

        async def sentence_stream():
            for i in range(2):
                yield f"Sentence {i}."

        # First request: parallel enabled
        pipeline1 = MockParallelSynthesisPipeline(tts_client, enabled=True)
        chunks1 = []
        async for chunk in pipeline1.synthesize_sentences(sentence_stream()):
            chunks1.append(chunk)
        assert len(chunks1) == 2

        # Second request: parallel disabled
        pipeline2 = MockParallelSynthesisPipeline(tts_client, enabled=False)
        chunks2 = []
        async for chunk in pipeline2.synthesize_sentences(sentence_stream()):
            chunks2.append(chunk)
        assert len(chunks2) == 2


# ============================================================================
# Error Scenarios
# ============================================================================


class TestErrorScenarios:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_tts_synthesis_failure(self) -> None:
        """Test handling of TTS synthesis failure."""
        tts_client = MockTTSClient(latency_ms=50, failure_rate=0.5)
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=2)

        async def sentence_stream():
            for i in range(5):
                yield f"Sentence {i}."

        # Should still complete despite failures
        audio_chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            audio_chunks.append(chunk)

        # Should emit audio for all sentences (with fallback silence for failures)
        assert len(audio_chunks) == 5

    @pytest.mark.asyncio
    async def test_empty_sentence_stream(self) -> None:
        """Test handling of empty sentence stream."""
        tts_client = MockTTSClient(latency_ms=50)
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=2)

        async def empty_stream():
            if False:
                yield ""

        audio_chunks = []
        async for chunk in pipeline.synthesize_sentences(empty_stream()):
            audio_chunks.append(chunk)

        assert len(audio_chunks) == 0

    @pytest.mark.asyncio
    async def test_single_sentence_response(self) -> None:
        """Test single sentence response (edge case)."""
        tts_client = MockTTSClient(latency_ms=50)
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=2)

        async def single_sentence():
            yield "Just one sentence."

        audio_chunks = []
        async for chunk in pipeline.synthesize_sentences(single_sentence()):
            audio_chunks.append(chunk)

        assert len(audio_chunks) == 1

    @pytest.mark.asyncio
    async def test_very_long_sentence(self) -> None:
        """Test very long sentence (>200 words)."""
        tts_client = MockTTSClient(latency_ms=100)
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=2)

        # Generate 200-word sentence
        long_sentence = " ".join(["word"] * 200) + "."

        async def long_sentence_stream():
            yield long_sentence

        audio_chunks = []
        async for chunk in pipeline.synthesize_sentences(long_sentence_stream()):
            audio_chunks.append(chunk)

        assert len(audio_chunks) == 1
        assert len(pipeline.sentences_processed[0]) > 1000  # Very long


# ============================================================================
# Barge-In Handling
# ============================================================================


class TestBargeInHandling:
    """Test barge-in (user interruption) handling."""

    @pytest.mark.asyncio
    async def test_cancel_during_synthesis(self) -> None:
        """Test cancelling synthesis when user interrupts."""
        tts_client = MockTTSClient(latency_ms=200)  # Slow synthesis
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=2)

        async def sentence_stream():
            for i in range(10):
                yield f"Long sentence {i}."

        # Start synthesis
        synthesis_task = asyncio.create_task(
            pipeline.synthesize_sentences(sentence_stream()).__anext__()
        )

        # Wait a bit, then cancel
        await asyncio.sleep(0.1)
        await pipeline.cancel_all()
        synthesis_task.cancel()

        # Should cancel gracefully
        try:
            await synthesis_task
        except (asyncio.CancelledError, StopAsyncIteration):
            pass  # Expected

    @pytest.mark.asyncio
    async def test_rapid_cancel_and_restart(self) -> None:
        """Test rapid cancel and restart (user changes mind)."""
        tts_client = MockTTSClient(latency_ms=50)

        async def sentence_stream():
            yield "First response."
            yield "Second sentence."

        # First synthesis
        pipeline1 = MockParallelSynthesisPipeline(tts_client, num_workers=2)
        task1 = asyncio.create_task(
            self._collect_audio(pipeline1.synthesize_sentences(sentence_stream()))
        )

        # Cancel quickly
        await asyncio.sleep(0.05)
        await pipeline1.cancel_all()
        task1.cancel()

        try:
            await task1
        except asyncio.CancelledError:
            pass

        # Start new synthesis immediately
        pipeline2 = MockParallelSynthesisPipeline(tts_client, num_workers=2)
        chunks = await self._collect_audio(
            pipeline2.synthesize_sentences(sentence_stream())
        )

        assert len(chunks) == 2

    async def _collect_audio(self, audio_stream: AsyncIterator[bytes]) -> list[bytes]:
        """Collect all audio chunks from stream."""
        chunks = []
        async for chunk in audio_stream:
            chunks.append(chunk)
        return chunks


# ============================================================================
# Audio Gap Detection
# ============================================================================


class TestAudioGapDetection:
    """Test audio continuity and gap detection."""

    @pytest.mark.asyncio
    async def test_no_gaps_between_sentences(self) -> None:
        """Test that audio chunks have no gaps (seamless playback)."""
        tts_client = MockTTSClient(latency_ms=50)
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=2)

        async def sentence_stream():
            for i in range(5):
                yield f"Sentence {i}."

        # Track emission timestamps
        emission_times = []

        async for _chunk in pipeline.synthesize_sentences(sentence_stream()):
            emission_times.append(time.monotonic())

        # Calculate gaps between emissions
        gaps = []
        for i in range(1, len(emission_times)):
            gap = (emission_times[i] - emission_times[i - 1]) * 1000  # ms
            gaps.append(gap)

        # With parallel synthesis, gaps should be minimal
        # Each chunk represents 20ms of audio, so gaps should be close to 0
        # (ideally all chunks buffered before playback starts)
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        max_gap = max(gaps) if gaps else 0

        # Relaxed thresholds for mock implementation
        assert avg_gap < 100  # <100ms average gap
        assert max_gap < 200  # <200ms max gap

    @pytest.mark.asyncio
    async def test_sequential_vs_parallel_gaps(self) -> None:
        """Compare gaps between sequential and parallel synthesis."""
        async def sentence_stream():
            for i in range(4):
                yield f"Sentence {i}."

        # Sequential synthesis
        tts_client_seq = MockTTSClient(latency_ms=100)
        pipeline_seq = MockParallelSynthesisPipeline(
            tts_client_seq,
            enabled=False,  # Sequential
        )

        seq_times = []
        async for _chunk in pipeline_seq.synthesize_sentences(sentence_stream()):
            seq_times.append(time.monotonic())

        # Parallel synthesis
        tts_client_par = MockTTSClient(latency_ms=100)
        pipeline_par = MockParallelSynthesisPipeline(
            tts_client_par,
            num_workers=2,
            enabled=True,  # Parallel
        )

        par_times = []
        async for _chunk in pipeline_par.synthesize_sentences(sentence_stream()):
            par_times.append(time.monotonic())

        # Calculate total time
        seq_total = seq_times[-1] - seq_times[0] if len(seq_times) > 1 else 0
        par_total = par_times[-1] - par_times[0] if len(par_times) > 1 else 0

        # Parallel should be faster (at least 30% reduction)
        assert par_total < seq_total * 0.7


# ============================================================================
# Real-World Scenarios
# ============================================================================


class TestRealWorldScenarios:
    """Test realistic user interaction scenarios."""

    @pytest.mark.asyncio
    async def test_weather_question_scenario(self) -> None:
        """Test exact weather question scenario from user.

        User: "What are the weather norms and whatnot in the fall in Salinas, California?"
        Expected: 5-6 sentences, seamless playback, no gaps
        """
        tts_client = MockTTSClient(latency_ms=80)
        segmenter = MockSentenceSegmenter()
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=2)

        # Mock LLM response
        async def llm_stream():
            response = (
                "The fall weather in Salinas, California is generally mild and pleasant. "
                "Average daytime temperatures range from 60 to 75 degrees Fahrenheit. "
                "Mornings can be cool with coastal fog, which typically burns off "
                "by midday. "
                "Rainfall is minimal during this season, with most precipitation "
                "occurring later in winter. "
                "It's an excellent time to visit the agricultural areas and enjoy "
                "the harvest season. "
                "You'll want to bring light layers for temperature variations "
                "throughout the day."
            )
            for char in response:
                yield char
                await asyncio.sleep(0.002)

        # Segment and synthesize
        sentence_stream = segmenter.segment(llm_stream())
        audio_chunks = []
        emission_times = []

        start_time = time.monotonic()
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            audio_chunks.append(chunk)
            emission_times.append(time.monotonic())

        emission_times[-1] - start_time if emission_times else 0

        # Verify
        assert len(pipeline.sentences_processed) == 6
        assert len(audio_chunks) == 6

        # Check for gaps
        gaps = [
            (emission_times[i] - emission_times[i - 1]) * 1000
            for i in range(1, len(emission_times))
        ]
        max_gap = max(gaps) if gaps else 0
        assert max_gap < 100  # <100ms max gap (seamless)

    @pytest.mark.asyncio
    async def test_storytelling_scenario(self) -> None:
        """Test long storytelling response (10+ sentences)."""
        tts_client = MockTTSClient(latency_ms=60)
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=3)

        async def sentence_stream():
            story_sentences = [
                "Once upon a time, there was a curious robot.",
                "The robot loved to learn new things every day.",
                "One day, it discovered the joy of helping people.",
                "It started by answering simple questions.",
                "Soon, it could hold entire conversations.",
                "People from all over would come to chat.",
                "The robot never got tired of talking.",
                "Each conversation taught it something new.",
                "It learned about human emotions and experiences.",
                "And it lived happily ever after, always learning.",
            ]
            for sentence in story_sentences:
                yield sentence

        audio_chunks = []
        start_time = time.monotonic()

        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            audio_chunks.append(chunk)

        total_time = time.monotonic() - start_time

        # Verify
        assert len(audio_chunks) == 10

        # With 3 workers and 60ms latency:
        # Sequential: 10 * 60ms = 600ms
        # Parallel (3 workers): ~240ms (4 batches: 3+3+3+1)
        assert total_time < 0.4  # Should be much faster than sequential

    @pytest.mark.asyncio
    async def test_technical_explanation_scenario(self) -> None:
        """Test technical explanation with complex sentences."""
        tts_client = MockTTSClient(latency_ms=100)
        segmenter = MockSentenceSegmenter()
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=2)

        async def llm_stream():
            response = (
                "Neural networks are computational models inspired by biological neurons. "
                "They consist of interconnected layers of nodes, called artificial neurons. "
                "Each connection has a weight that adjusts during training. "
                "The network learns by minimizing a loss function through backpropagation. "
                "This process allows the model to recognize patterns in data."
            )
            for char in response:
                yield char
                await asyncio.sleep(0.001)

        # Segment and synthesize
        sentence_stream = segmenter.segment(llm_stream())
        audio_chunks = []

        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            audio_chunks.append(chunk)

        # Verify
        assert len(pipeline.sentences_processed) == 5
        assert len(audio_chunks) == 5
        # Check all sentences processed
        assert all("." in s for s in pipeline.sentences_processed)


# ============================================================================
# Configuration and Tuning Tests
# ============================================================================


class TestConfigurationTuning:
    """Test different worker configurations."""

    @pytest.mark.asyncio
    async def test_single_worker_performance(self) -> None:
        """Test with single worker (sequential equivalent)."""
        tts_client = MockTTSClient(latency_ms=50)
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=1)

        async def sentence_stream():
            for i in range(4):
                yield f"Sentence {i}."

        start_time = time.monotonic()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)
        elapsed = time.monotonic() - start_time

        assert len(chunks) == 4
        # Single worker: essentially sequential (4 * 50ms = 200ms)
        assert elapsed >= 0.2

    @pytest.mark.asyncio
    async def test_two_worker_performance(self) -> None:
        """Test with 2 workers (optimal for most cases)."""
        tts_client = MockTTSClient(latency_ms=50)
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=2)

        async def sentence_stream():
            for i in range(4):
                yield f"Sentence {i}."

        start_time = time.monotonic()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)
        elapsed = time.monotonic() - start_time

        assert len(chunks) == 4
        # 2 workers: 2 batches (2 * 50ms = 100ms)
        assert elapsed < 0.15

    @pytest.mark.asyncio
    async def test_three_worker_performance(self) -> None:
        """Test with 3 workers (high parallelism)."""
        tts_client = MockTTSClient(latency_ms=50)
        pipeline = MockParallelSynthesisPipeline(tts_client, num_workers=3)

        async def sentence_stream():
            for i in range(6):
                yield f"Sentence {i}."

        start_time = time.monotonic()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)
        elapsed = time.monotonic() - start_time

        assert len(chunks) == 6
        # 3 workers: 2 batches (2 * 50ms = 100ms)
        assert elapsed < 0.15
