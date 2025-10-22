"""Unit tests for DualLLMOrchestrator.

Tests cover:
- Filler generation with context-aware template selection
- Parallel LLM execution (fast filler + full OpenAI)
- Response buffering and streaming
- Error handling and fallbacks
- Feature flag (filler_enabled)
- Performance validation (latency targets)
"""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.orchestrator.dual_llm import (
    DualLLMOrchestrator,
    LLMResponse,
    ResponsePhase,
)


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Create mock OpenAI client."""
    mock_client = AsyncMock()
    return mock_client


@pytest.fixture
def orchestrator(mock_openai_client: AsyncMock) -> DualLLMOrchestrator:
    """Create DualLLMOrchestrator with mocked OpenAI client."""
    with patch("src.orchestrator.dual_llm.AsyncOpenAI", return_value=mock_openai_client):
        return DualLLMOrchestrator(
            openai_api_key="sk-test-key-123",
            filler_enabled=True,
        )


@pytest.fixture
def orchestrator_no_filler(mock_openai_client: AsyncMock) -> DualLLMOrchestrator:
    """Create DualLLMOrchestrator with fillers disabled."""
    with patch("src.orchestrator.dual_llm.AsyncOpenAI", return_value=mock_openai_client):
        return DualLLMOrchestrator(
            openai_api_key="sk-test-key-123",
            filler_enabled=False,
        )


class TestInitialization:
    """Test orchestrator initialization."""

    def test_init_with_valid_api_key(self) -> None:
        """Test initialization with valid API key."""
        with patch("src.orchestrator.dual_llm.AsyncOpenAI"):
            orchestrator = DualLLMOrchestrator(openai_api_key="sk-valid-key")
            assert orchestrator.filler_enabled is True
            assert orchestrator.full_model == "gpt-4o-mini"
            assert orchestrator.filler_max_tokens == 30
            assert orchestrator.full_max_tokens == 500

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        with patch("src.orchestrator.dual_llm.AsyncOpenAI"):
            orchestrator = DualLLMOrchestrator(
                openai_api_key="sk-valid-key",
                fast_model="gpt-3.5-turbo",
                full_model="gpt-4",
                filler_max_tokens=50,
                full_max_tokens=1000,
                filler_enabled=False,
            )
            assert orchestrator.fast_model == "gpt-3.5-turbo"
            assert orchestrator.full_model == "gpt-4"
            assert orchestrator.filler_max_tokens == 50
            assert orchestrator.full_max_tokens == 1000
            assert orchestrator.filler_enabled is False

    def test_init_with_invalid_api_key_raises_error(self) -> None:
        """Test initialization with invalid API key raises ValueError."""
        with pytest.raises(ValueError, match="Invalid OpenAI API key"):
            DualLLMOrchestrator(openai_api_key="")

        with pytest.raises(ValueError, match="Invalid OpenAI API key"):
            DualLLMOrchestrator(openai_api_key="sk-your-openai-api-key")


class TestFillerGeneration:
    """Test filler template generation."""

    def test_select_filler_weather_context(self, orchestrator: DualLLMOrchestrator) -> None:
        """Test context-aware filler selection for weather queries."""
        filler = orchestrator._select_filler("What's the weather like today?")
        assert any(
            keyword in filler.lower() for keyword in ["weather", "forecast", "conditions"]
        )

    def test_select_filler_time_context(self, orchestrator: DualLLMOrchestrator) -> None:
        """Test context-aware filler selection for time queries."""
        filler = orchestrator._select_filler("What time is it?")
        assert any(keyword in filler.lower() for keyword in ["time", "current"])

    def test_select_filler_calculation_context(self, orchestrator: DualLLMOrchestrator) -> None:
        """Test context-aware filler selection for calculation queries."""
        filler = orchestrator._select_filler("What is 123 * 456?")
        assert any(keyword in filler.lower() for keyword in ["calculate", "math", "computing"])

    def test_select_filler_search_context(self, orchestrator: DualLLMOrchestrator) -> None:
        """Test context-aware filler selection for search queries."""
        filler = orchestrator._select_filler("Can you search for Python tutorials?")
        assert any(keyword in filler.lower() for keyword in ["search", "looking", "looking"])

    def test_select_filler_generic_fallback(self, orchestrator: DualLLMOrchestrator) -> None:
        """Test generic filler selection for non-matching queries."""
        filler = orchestrator._select_filler("Tell me about quantum mechanics")
        # Should return a valid filler template
        assert len(filler) > 0
        assert isinstance(filler, str)

    def test_filler_generation_variety(self, orchestrator: DualLLMOrchestrator) -> None:
        """Test that filler selection provides variety."""
        fillers = {orchestrator._select_filler("generic question") for _ in range(100)}
        # Should have some variety (not always the same filler)
        # With 20+ templates, we should see at least 3 different ones in 100 attempts
        assert len(fillers) >= 3

    async def test_generate_filler_metadata(self, orchestrator: DualLLMOrchestrator) -> None:
        """Test filler generation returns proper metadata."""
        filler_response = await orchestrator._generate_filler("What's the weather?")

        assert isinstance(filler_response, LLMResponse)
        assert len(filler_response.content) > 0
        assert filler_response.latency_ms >= 0
        assert filler_response.latency_ms < 50  # Should be very fast (template selection)
        assert filler_response.token_count > 0
        assert filler_response.is_filler is True
        assert filler_response.phase == ResponsePhase.FILLER


class TestFullResponseGeneration:
    """Test full LLM response generation."""

    async def test_generate_full_response_streaming(
        self, orchestrator: DualLLMOrchestrator, mock_openai_client: AsyncMock
    ) -> None:
        """Test full response streaming from OpenAI."""
        # Mock OpenAI streaming response
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta.content = "Hello"

        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta.content = " world"

        mock_chunk3 = Mock()
        mock_chunk3.choices = [Mock()]
        mock_chunk3.choices[0].delta.content = "!"

        async def mock_stream() -> AsyncIterator[Mock]:
            yield mock_chunk1
            yield mock_chunk2
            yield mock_chunk3

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        # Collect response chunks
        chunks = []
        async for chunk in orchestrator._generate_full_response(
            "Test message", conversation_history=[]
        ):
            chunks.append(chunk)

        # Verify
        assert chunks == ["Hello", " world", "!"]
        assert mock_openai_client.chat.completions.create.called

    async def test_generate_full_response_with_history(
        self, orchestrator: DualLLMOrchestrator, mock_openai_client: AsyncMock
    ) -> None:
        """Test full response generation includes conversation history."""
        # Mock streaming response
        async def mock_stream() -> AsyncIterator[Mock]:
            mock_chunk = Mock()
            mock_chunk.choices = [Mock()]
            mock_chunk.choices[0].delta.content = "Response"
            yield mock_chunk

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        chunks = []
        async for chunk in orchestrator._generate_full_response(
            "How are you?", conversation_history=conversation_history
        ):
            chunks.append(chunk)

        # Verify history was included in API call
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 3  # 2 history + 1 new
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Hi there!"
        assert messages[2]["content"] == "How are you?"

    async def test_generate_full_response_error_fallback(
        self, orchestrator: DualLLMOrchestrator, mock_openai_client: AsyncMock
    ) -> None:
        """Test error fallback for full response generation."""
        # Mock API error
        mock_openai_client.chat.completions.create.side_effect = Exception("API error")

        chunks = []
        async for chunk in orchestrator._generate_full_response("Test", []):
            chunks.append(chunk)

        # Should return fallback message
        assert len(chunks) == 1
        assert "apologize" in chunks[0].lower()
        assert "trouble" in chunks[0].lower()


class TestParallelExecution:
    """Test parallel filler + full LLM execution."""

    async def test_parallel_execution_timing(
        self, orchestrator: DualLLMOrchestrator, mock_openai_client: AsyncMock
    ) -> None:
        """Test that filler and full response execute in parallel."""
        # Mock slow full response (500ms)
        async def slow_stream() -> AsyncIterator[Mock]:
            await asyncio.sleep(0.5)
            mock_chunk = Mock()
            mock_chunk.choices = [Mock()]
            mock_chunk.choices[0].delta.content = "Slow response"
            yield mock_chunk

        mock_openai_client.chat.completions.create.return_value = slow_stream()

        # Measure total time
        import time

        start = time.perf_counter()

        responses = []
        async for text, phase in orchestrator.generate_response("Test question"):
            responses.append((text, phase))

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Filler should arrive immediately (< 50ms)
        # Full response should arrive after ~500ms
        # Total should be dominated by full response time, not sum of both
        assert elapsed_ms < 700  # Parallel execution, not sequential

        # Verify filler came first
        assert responses[0][1] == ResponsePhase.FILLER
        assert len(responses[0][0]) > 0  # Has content

    async def test_buffer_full_response_during_filler(
        self, orchestrator: DualLLMOrchestrator, mock_openai_client: AsyncMock
    ) -> None:
        """Test that full response is buffered during filler phase."""
        # Mock streaming response with multiple chunks
        async def mock_stream() -> AsyncIterator[Mock]:
            for text in ["Chunk1", " Chunk2", " Chunk3"]:
                mock_chunk = Mock()
                mock_chunk.choices = [Mock()]
                mock_chunk.choices[0].delta.content = text
                yield mock_chunk

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        # Collect all buffered chunks
        buffered_chunks = await orchestrator._buffer_full_response("Test", [])

        assert buffered_chunks == ["Chunk1", " Chunk2", " Chunk3"]


class TestResponseStreaming:
    """Test complete response streaming workflow."""

    async def test_generate_response_complete_flow(
        self, orchestrator: DualLLMOrchestrator, mock_openai_client: AsyncMock
    ) -> None:
        """Test complete response flow with all phases."""
        # Mock full response
        async def mock_stream() -> AsyncIterator[Mock]:
            mock_chunk = Mock()
            mock_chunk.choices = [Mock()]
            mock_chunk.choices[0].delta.content = "Full response text"
            yield mock_chunk

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        # Collect all responses
        responses = []
        async for text, phase in orchestrator.generate_response("Test question"):
            responses.append((text, phase))

        # Verify phases
        phases = [r[1] for r in responses]
        assert ResponsePhase.FILLER in phases
        assert ResponsePhase.TRANSITION in phases
        assert ResponsePhase.COMPLETE in phases

        # Verify content
        assert any(len(r[0]) > 0 for r in responses if r[1] == ResponsePhase.FILLER)

    async def test_generate_response_without_filler_flag(
        self, orchestrator_no_filler: DualLLMOrchestrator, mock_openai_client: AsyncMock
    ) -> None:
        """Test response generation with filler disabled."""
        # Mock full response
        async def mock_stream() -> AsyncIterator[Mock]:
            mock_chunk = Mock()
            mock_chunk.choices = [Mock()]
            mock_chunk.choices[0].delta.content = "Direct response"
            yield mock_chunk

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        # Collect responses
        responses = []
        async for text, phase in orchestrator_no_filler.generate_response("Test"):
            responses.append((text, phase))

        # Should NOT have filler phase
        phases = [r[1] for r in responses]
        assert ResponsePhase.FILLER not in phases
        assert ResponsePhase.FULL in phases or ResponsePhase.TRANSITION in phases

    async def test_generate_response_without_filler_method(
        self, orchestrator: DualLLMOrchestrator, mock_openai_client: AsyncMock
    ) -> None:
        """Test direct response generation without filler."""
        # Mock full response
        async def mock_stream() -> AsyncIterator[Mock]:
            mock_chunk = Mock()
            mock_chunk.choices = [Mock()]
            mock_chunk.choices[0].delta.content = "Direct"
            yield mock_chunk

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        responses = []
        async for text, phase in orchestrator.generate_response_without_filler("Test"):
            responses.append((text, phase))

        # Should skip filler phase
        phases = [r[1] for r in responses]
        assert ResponsePhase.FILLER not in phases
        assert ResponsePhase.FULL in phases


class TestErrorHandling:
    """Test error handling and edge cases."""

    async def test_openai_api_error_returns_fallback(
        self, orchestrator: DualLLMOrchestrator, mock_openai_client: AsyncMock
    ) -> None:
        """Test OpenAI API errors return fallback message."""
        # Mock API error
        mock_openai_client.chat.completions.create.side_effect = Exception("API timeout")

        responses = []
        async for text, phase in orchestrator.generate_response("Test"):
            responses.append((text, phase))

        # Should still complete with filler + fallback
        assert any(r[1] == ResponsePhase.FILLER for r in responses)
        assert any(r[1] == ResponsePhase.COMPLETE for r in responses)

        # Fallback message should be present
        full_text = "".join(r[0] for r in responses if r[1] == ResponsePhase.FULL)
        assert "apologize" in full_text.lower() or "trouble" in full_text.lower()

    async def test_empty_conversation_history(
        self, orchestrator: DualLLMOrchestrator, mock_openai_client: AsyncMock
    ) -> None:
        """Test handling of empty conversation history."""
        # Mock response
        async def mock_stream() -> AsyncIterator[Mock]:
            mock_chunk = Mock()
            mock_chunk.choices = [Mock()]
            mock_chunk.choices[0].delta.content = "Response"
            yield mock_chunk

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        responses = []
        async for text, phase in orchestrator.generate_response(
            "Test", conversation_history=None
        ):
            responses.append((text, phase))

        # Should work without errors
        assert len(responses) > 0

    async def test_empty_full_response_chunks(
        self, orchestrator: DualLLMOrchestrator, mock_openai_client: AsyncMock
    ) -> None:
        """Test handling when full response has no chunks."""

        async def empty_stream() -> AsyncIterator[Mock]:
            # Yield nothing
            return
            yield  # Make this a generator

        mock_openai_client.chat.completions.create.return_value = empty_stream()

        responses = []
        async for text, phase in orchestrator.generate_response("Test"):
            responses.append((text, phase))

        # Should still have filler and completion
        phases = [r[1] for r in responses]
        assert ResponsePhase.FILLER in phases
        assert ResponsePhase.COMPLETE in phases


class TestPerformance:
    """Test performance characteristics and latency targets."""

    async def test_filler_latency_under_50ms(
        self, orchestrator: DualLLMOrchestrator
    ) -> None:
        """Test filler generation latency is under 50ms."""
        import time

        start = time.perf_counter()
        filler = await orchestrator._generate_filler("Test question")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"Filler latency {elapsed_ms:.2f}ms exceeds 50ms target"
        assert filler.latency_ms < 50

    async def test_first_audio_latency_under_2s(
        self, orchestrator: DualLLMOrchestrator, mock_openai_client: AsyncMock
    ) -> None:
        """Test first audio (filler) arrives under 2 seconds.

        NOTE: This is measuring time to FIRST audio (filler), not full response.
        Filler should arrive in < 50ms, well under the 2s target.
        """
        # Mock fast streaming response
        async def fast_stream() -> AsyncIterator[Mock]:
            mock_chunk = Mock()
            mock_chunk.choices = [Mock()]
            mock_chunk.choices[0].delta.content = "Fast response"
            yield mock_chunk

        mock_openai_client.chat.completions.create.return_value = fast_stream()

        import time

        start = time.perf_counter()

        # Get first response (should be filler)
        async for _text, phase in orchestrator.generate_response("Test"):
            if phase == ResponsePhase.FILLER:
                elapsed_ms = (time.perf_counter() - start) * 1000
                assert (
                    elapsed_ms < 2000
                ), f"First audio latency {elapsed_ms:.2f}ms exceeds 2s target"
                break
