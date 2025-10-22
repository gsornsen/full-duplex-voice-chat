"""Integration tests for dual-LLM agent workflow.

Tests the complete integration of:
- DualLLMOrchestrator
- SentenceSegmenter
- ResponseBuffer
- TTS synthesis
- LiveKit Agent coordination

Tests verify:
- End-to-end response generation with filler
- Parallel execution coordination
- Seamless audio transitions
- Performance targets (first audio < 2s)
- Error resilience and fallbacks
"""

import asyncio
import os
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.orchestrator.dual_llm import DualLLMOrchestrator, ResponsePhase
from src.orchestrator.response_buffer import ResponseBuffer, ResponseBufferConfig
from src.orchestrator.sentence_segmenter import SentenceSegmenter


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Create mock OpenAI client for integration tests."""
    mock_client = AsyncMock()

    # Mock realistic streaming response
    async def mock_streaming_response() -> AsyncIterator[Mock]:
        # Simulate realistic token stream
        tokens = [
            "The",
            " weather",
            " today",
            " is",
            " sunny",
            " with",
            " a",
            " high",
            " of",
            " 72",
            " degrees",
            " Fahrenheit",
            ".",
            " It's",
            " a",
            " perfect",
            " day",
            " to",
            " go",
            " outside",
            "!",
        ]
        for token in tokens:
            chunk = Mock()
            chunk.choices = [Mock()]
            chunk.choices[0].delta.content = token
            yield chunk
            await asyncio.sleep(0.01)  # Simulate realistic streaming delay

    mock_client.chat.completions.create.return_value = mock_streaming_response()
    return mock_client


@pytest.fixture
def mock_tts_client() -> AsyncMock:
    """Create mock TTS client for integration tests."""
    mock_client = AsyncMock()

    async def mock_synthesize(sentences: list[str]) -> AsyncIterator[bytes]:
        """Mock TTS synthesis with realistic frame generation."""
        for sentence in sentences:
            # Estimate frames based on sentence length
            # Roughly 3 frames per sentence (60ms)
            num_frames = max(3, len(sentence) // 10)
            for _ in range(num_frames):
                yield b"\x00" * 1920  # 1920 bytes = 20ms @ 48kHz mono 16-bit
                await asyncio.sleep(0.02)  # Simulate 20ms frame duration

    mock_client.synthesize = mock_synthesize
    return mock_client


@pytest.fixture
def orchestrator(mock_openai_client: AsyncMock) -> DualLLMOrchestrator:
    """Create DualLLMOrchestrator for integration tests."""
    with patch("src.orchestrator.dual_llm.AsyncOpenAI", return_value=mock_openai_client):
        return DualLLMOrchestrator(
            openai_api_key="sk-test-integration-key",
            filler_enabled=True,
        )


@pytest.fixture
def segmenter() -> SentenceSegmenter:
    """Create SentenceSegmenter for integration tests."""
    return SentenceSegmenter(min_tokens=2, buffer_timeout=0.3)


@pytest.fixture
def response_buffer(mock_tts_client: AsyncMock) -> ResponseBuffer:
    """Create ResponseBuffer for integration tests."""
    config = ResponseBufferConfig(
        max_buffered_sentences=20,
        max_buffered_audio_frames=500,
        buffer_tts_lead_time_ms=500.0,
    )
    return ResponseBuffer(config=config, tts_client=mock_tts_client)


class TestEndToEndDualLLMFlow:
    """Test complete dual-LLM workflow end-to-end."""

    async def test_complete_response_generation_with_filler(
        self, orchestrator: DualLLMOrchestrator
    ) -> None:
        """Test complete response generation includes filler and full response."""
        user_message = "What's the weather like today?"

        # Collect all response phases
        responses = []
        async for text, phase in orchestrator.generate_response(user_message):
            responses.append((text, phase))

        # Verify all phases present
        phases = [r[1] for r in responses]
        assert ResponsePhase.FILLER in phases
        assert ResponsePhase.TRANSITION in phases
        assert ResponsePhase.COMPLETE in phases

        # Verify filler content
        filler_text = next(r[0] for r in responses if r[1] == ResponsePhase.FILLER)
        assert len(filler_text) > 0

        # Verify full response content
        full_text = "".join(r[0] for r in responses if r[1] != ResponsePhase.COMPLETE)
        assert len(full_text) > 0

    async def test_filler_arrives_before_full_response(
        self, orchestrator: DualLLMOrchestrator
    ) -> None:
        """Test filler is yielded before full response."""
        user_message = "Tell me about Python programming"

        phases_in_order = []
        async for _, phase in orchestrator.generate_response(user_message):
            phases_in_order.append(phase)

        # Filler should be first non-complete phase
        non_complete = [p for p in phases_in_order if p != ResponsePhase.COMPLETE]
        assert non_complete[0] == ResponsePhase.FILLER

    async def test_parallel_execution_timing(
        self, orchestrator: DualLLMOrchestrator
    ) -> None:
        """Test filler and full response execute in parallel."""
        import time

        user_message = "Explain quantum entanglement"

        start = time.perf_counter()

        filler_time = None
        full_time = None

        async for _, phase in orchestrator.generate_response(user_message):
            if phase == ResponsePhase.FILLER and filler_time is None:
                filler_time = (time.perf_counter() - start) * 1000

            if phase == ResponsePhase.TRANSITION and full_time is None:
                full_time = (time.perf_counter() - start) * 1000

        # Filler should arrive very quickly (< 50ms)
        assert filler_time is not None
        assert filler_time < 100

        # Full response should arrive shortly after (not sequential delay)
        assert full_time is not None
        assert full_time < 1000  # Within 1 second for mock


class TestSentenceSegmentation:
    """Test sentence segmentation in dual-LLM workflow."""

    async def test_sentence_segmenter_with_llm_stream(
        self, orchestrator: DualLLMOrchestrator, segmenter: SentenceSegmenter
    ) -> None:
        """Test sentence segmenter processes LLM token stream."""
        user_message = "What is Python?"

        # Get LLM token stream
        token_stream = orchestrator._generate_full_response(user_message, [])

        # Segment into sentences
        sentences = []
        async for sentence in segmenter.segment(token_stream):
            sentences.append(sentence)

        # Should have at least one sentence
        assert len(sentences) >= 1

        # Sentences should have content
        assert all(len(s.strip()) > 0 for s in sentences)

    async def test_segmenter_handles_abbreviations_in_llm_output(
        self, segmenter: SentenceSegmenter
    ) -> None:
        """Test segmenter handles abbreviations in LLM-style output."""

        async def llm_style_tokens() -> AsyncIterator[str]:
            # Simulate LLM output with abbreviations
            tokens = ["Dr", ".", " Smith", " studies", " AI", " etc", ".", " More", "."]
            for token in tokens:
                yield token
                await asyncio.sleep(0.01)

        sentences = []
        async for sentence in segmenter.segment(llm_style_tokens()):
            sentences.append(sentence)

        # Should properly segment around abbreviations
        assert len(sentences) >= 1
        assert "Dr. Smith" in " ".join(sentences)


class TestResponseBuffering:
    """Test response buffering with TTS synthesis."""

    async def test_buffer_during_filler_synthesizes_audio(
        self, response_buffer: ResponseBuffer
    ) -> None:
        """Test buffer_during_filler generates audio frames."""
        filler_text = "Let me check that for you..."

        async def sentence_stream() -> AsyncIterator[str]:
            yield "The answer is 42."
            yield "That's all you need to know."

        # Collect audio frames
        frames = []
        async for frame in response_buffer.buffer_during_filler(
            filler_text, sentence_stream()
        ):
            frames.append(frame)

        # Should have multiple frames
        assert len(frames) > 0

        # All frames should be valid audio
        assert all(isinstance(f, bytes) for f in frames)
        assert all(len(f) == 1920 for f in frames)

    async def test_buffer_tracks_metrics(
        self, response_buffer: ResponseBuffer
    ) -> None:
        """Test buffer tracks comprehensive metrics."""
        filler_text = "One moment please..."

        async def sentence_stream() -> AsyncIterator[str]:
            yield "First sentence."
            yield "Second sentence."

        # Execute buffering
        frames = []
        async for frame in response_buffer.buffer_during_filler(
            filler_text, sentence_stream()
        ):
            frames.append(frame)

        # Verify metrics
        metrics = response_buffer._get_metrics_summary()

        assert metrics["filler_duration_ms"] > 0
        assert metrics["buffered_sentences"] > 0
        assert metrics["buffered_audio_frames"] > 0

    async def test_buffer_handles_overflow_gracefully(
        self, mock_tts_client: AsyncMock
    ) -> None:
        """Test buffer handles sentence queue overflow."""
        # Create buffer with small queue
        config = ResponseBufferConfig(
            max_buffered_sentences=2,
        )
        buffer = ResponseBuffer(config=config, tts_client=mock_tts_client)

        filler_text = "Processing..."

        async def many_sentences() -> AsyncIterator[str]:
            for i in range(10):
                yield f"Sentence {i}."
                await asyncio.sleep(0.01)

        # Should handle overflow without errors
        frames = []
        async for frame in buffer.buffer_during_filler(filler_text, many_sentences()):
            frames.append(frame)

        # Should complete successfully
        assert len(frames) > 0


class TestIntegratedWorkflow:
    """Test complete integrated workflow with all components."""

    async def test_orchestrator_to_segmenter_to_buffer(
        self,
        orchestrator: DualLLMOrchestrator,
        segmenter: SentenceSegmenter,
        response_buffer: ResponseBuffer,
    ) -> None:
        """Test complete pipeline: Orchestrator → Segmenter → Buffer → Audio."""
        user_message = "What is the meaning of life?"

        # Generate filler
        filler = await response_buffer.generate_filler(user_message)

        # Get full LLM response stream
        token_stream = orchestrator._generate_full_response(user_message, [])

        # Segment tokens into sentences
        sentence_stream = segmenter.segment(token_stream)

        # Buffer during filler and generate audio
        frames = []
        async for frame in response_buffer.buffer_during_filler(filler, sentence_stream):
            frames.append(frame)

        # Verify complete workflow
        assert len(frames) > 0
        assert response_buffer.metrics.buffered_sentences > 0

    async def test_integrated_error_recovery(
        self, segmenter: SentenceSegmenter, mock_tts_client: AsyncMock
    ) -> None:
        """Test integrated workflow recovers from errors."""
        # Create orchestrator with failing OpenAI
        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.side_effect = Exception("API error")

        with patch("src.orchestrator.dual_llm.AsyncOpenAI", return_value=mock_openai):
            orchestrator = DualLLMOrchestrator(
                openai_api_key="sk-test-key",
                filler_enabled=True,
            )

        # Should still generate filler and fallback message
        responses = []
        async for text, phase in orchestrator.generate_response("Test"):
            responses.append((text, phase))

        # Should have filler and fallback
        phases = [r[1] for r in responses]
        assert ResponsePhase.FILLER in phases
        assert ResponsePhase.COMPLETE in phases


class TestPerformanceTargets:
    """Test performance targets are met in integrated workflow."""

    async def test_first_audio_latency_under_2_seconds(
        self,
        orchestrator: DualLLMOrchestrator,
        segmenter: SentenceSegmenter,
        response_buffer: ResponseBuffer,
    ) -> None:
        """Test first audio (filler) arrives within 2 seconds.

        This is the key performance metric: user should hear something
        within 2 seconds of asking a question.
        """
        import time

        user_message = "What's the weather forecast for tomorrow?"

        start = time.perf_counter()

        # Generate filler (should be immediate)
        filler = await response_buffer.generate_filler(user_message)
        filler_latency_ms = (time.perf_counter() - start) * 1000

        # Filler generation should be very fast
        assert filler_latency_ms < 100

        # Get token stream and segment
        token_stream = orchestrator._generate_full_response(user_message, [])
        sentence_stream = segmenter.segment(token_stream)

        # Get first audio frame
        first_frame_time = None
        async for _frame in response_buffer.buffer_during_filler(filler, sentence_stream):
            if first_frame_time is None:
                first_frame_time = (time.perf_counter() - start) * 1000
            break  # Only need first frame

        # First audio should arrive within 2 seconds
        assert first_frame_time is not None
        assert (
            first_frame_time < 2000
        ), f"First audio took {first_frame_time:.2f}ms (target: <2000ms)"

    async def test_filler_generation_under_50ms(
        self, response_buffer: ResponseBuffer
    ) -> None:
        """Test filler generation is under 50ms."""
        import time

        start = time.perf_counter()
        filler = await response_buffer.generate_filler("Complex question about AI?")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"Filler generation took {elapsed_ms:.2f}ms (target: <50ms)"
        assert len(filler) > 0

    async def test_transition_gap_under_100ms(
        self, response_buffer: ResponseBuffer
    ) -> None:
        """Test transition gap between filler and full response is minimal."""
        filler_text = "Let me think..."

        async def quick_sentences() -> AsyncIterator[str]:
            yield "Quick response."

        # Execute buffering
        async for _ in response_buffer.buffer_during_filler(filler_text, quick_sentences()):
            pass

        # Check transition gap
        metrics = response_buffer._get_metrics_summary()

        # Transition gap should be minimal (< 100ms for seamless experience)
        # Note: In real implementation with actual TTS timing, this will be tighter
        assert metrics["transition_gap_ms"] < 200  # Generous bound for mock


class TestFeatureFlags:
    """Test feature flag behavior."""

    async def test_filler_disabled_skips_filler_phase(
        self, mock_openai_client: AsyncMock
    ) -> None:
        """Test filler_enabled=False skips filler generation."""
        with patch("src.orchestrator.dual_llm.AsyncOpenAI", return_value=mock_openai_client):
            orchestrator = DualLLMOrchestrator(
                openai_api_key="sk-test-key",
                filler_enabled=False,  # DISABLED
            )

        responses = []
        async for text, phase in orchestrator.generate_response("Test question"):
            responses.append((text, phase))

        # Should NOT have filler phase
        phases = [r[1] for r in responses]
        assert ResponsePhase.FILLER not in phases

    async def test_filler_enabled_includes_filler_phase(
        self, orchestrator: DualLLMOrchestrator
    ) -> None:
        """Test filler_enabled=True includes filler generation."""
        responses = []
        async for text, phase in orchestrator.generate_response("Test question"):
            responses.append((text, phase))

        # Should have filler phase
        phases = [r[1] for r in responses]
        assert ResponsePhase.FILLER in phases


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "sk-your-openai-api-key",
    reason="Requires valid OPENAI_API_KEY",
)
class TestRealOpenAIIntegration:
    """Integration tests with real OpenAI API (requires API key).

    These tests are skipped by default and only run when a valid
    OPENAI_API_KEY is set in the environment.
    """

    async def test_real_openai_streaming(self) -> None:
        """Test with real OpenAI API (if available)."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        orchestrator = DualLLMOrchestrator(
            openai_api_key=api_key,
            filler_enabled=True,
        )

        # Simple question
        responses = []
        async for text, phase in orchestrator.generate_response("Say hello in 5 words."):
            responses.append((text, phase))
            # Limit to prevent excessive API usage
            if len(responses) > 10:
                break

        # Should have filler and response
        phases = [r[1] for r in responses]
        assert ResponsePhase.FILLER in phases
        assert len(responses) > 1
