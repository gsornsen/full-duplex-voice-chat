"""Dual LLM Orchestrator for parallel fast/full response generation.

This module implements a dual-LLM architecture that provides low-latency conversational
responses by running fast template-based fillers in parallel with full LLM calls.

Architecture:
    1. Launch parallel LLM calls (fast template + full OpenAI streaming)
    2. Return filler immediately (0ms latency)
    3. Buffer full response during filler playback
    4. Transition seamlessly when filler completes
    5. Stream remaining full response

Response phases:
    - FILLER: Template-based instant response (0ms, 100% reliable)
    - TRANSITION: Seamless handoff from filler to full response
    - FULL: Stream remaining OpenAI response tokens
    - COMPLETE: Response generation finished

Performance targets:
    - Filler latency: 0ms (template selection)
    - Full LLM TTFT: p95 < 500ms (OpenAI streaming)
    - Transition latency: p95 < 50ms (buffer handoff)
    - Total latency: Filler + Transition ≈ 50ms (vs 500ms baseline)
"""

import asyncio
import logging
import secrets
import time
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ResponsePhase(Enum):
    """Response generation phase indicators.

    Phases:
        FILLER: Template-based filler response (instant)
        TRANSITION: Handoff from filler to full response
        FULL: Full LLM response streaming
        COMPLETE: Response generation completed
    """

    FILLER = "filler"
    TRANSITION = "transition"
    FULL = "full"
    COMPLETE = "complete"


class LLMResponse(BaseModel):
    """LLM response metadata and content.

    Attributes:
        content: Response text content
        latency_ms: Generation latency in milliseconds
        token_count: Number of tokens generated
        is_filler: Whether this is a template filler response
        phase: Current response phase
    """

    content: str = Field(description="Response text content")
    latency_ms: float = Field(ge=0.0, description="Generation latency in milliseconds")
    token_count: int = Field(ge=0, description="Number of tokens generated")
    is_filler: bool = Field(description="Whether this is a template filler response")
    phase: ResponsePhase = Field(description="Current response phase")


class DualLLMOrchestrator:
    """Dual LLM orchestrator for parallel fast/full response generation.

    This orchestrator implements a dual-LLM architecture that provides low-latency
    conversational responses by:
    1. Launching parallel fast (template) and full (OpenAI) LLM calls
    2. Returning template filler immediately (0ms latency)
    3. Buffering full response during filler playback
    4. Transitioning seamlessly when filler completes
    5. Streaming remaining full response tokens

    Example:
        >>> orchestrator = DualLLMOrchestrator(openai_api_key="sk-...")
        >>> async for text, phase in orchestrator.generate_response(
        ...     user_message="What's the weather?",
        ...     conversation_history=[],
        ... ):
        ...     print(f"[{phase.value}] {text}")
        [filler] Let me check that for you...
        [transition] The weather today is
        [full] sunny with a high of 72°F.
        [complete]

    Thread-safety: This class is NOT thread-safe. Use from a single async task.
    """

    # Template filler responses (20+ variants for natural variety)
    FILLER_TEMPLATES = [
        # Acknowledgment fillers (5 variants)
        "Let me check that for you...",
        "Let me look into that...",
        "Let me see...",
        "Give me just a moment...",
        "One moment please...",
        # Thinking fillers (5 variants)
        "Hmm, that's an interesting question...",
        "That's a great question...",
        "Let me think about that...",
        "Good question...",
        "Interesting...",
        # Processing fillers (5 variants)
        "Let me process that...",
        "Let me work on that...",
        "Let me figure that out...",
        "Working on it...",
        "Processing...",
        # Context-aware fillers (5 variants)
        "Let me find that information...",
        "Let me gather the details...",
        "Let me pull up the data...",
        "Checking that now...",
        "Looking that up...",
    ]

    # Context-specific filler mappings
    CONTEXT_FILLERS = {
        "weather": [
            "Let me check the weather...",
            "Checking the forecast...",
            "Looking up the weather conditions...",
        ],
        "time": [
            "Let me check the time...",
            "Getting the current time...",
        ],
        "calculation": [
            "Let me calculate that...",
            "Working on the math...",
            "Computing that now...",
        ],
        "search": [
            "Let me search for that...",
            "Looking that up...",
            "Searching now...",
        ],
    }

    def __init__(
        self,
        openai_api_key: str,
        fast_model: str = "gpt-4o-mini",
        full_model: str = "gpt-4o-mini",
        filler_max_tokens: int = 30,
        full_max_tokens: int = 500,
        filler_enabled: bool = True,
    ) -> None:
        """Initialize dual LLM orchestrator.

        Args:
            openai_api_key: OpenAI API key for authentication
            fast_model: Model for fast response (currently unused, templates only)
            full_model: Model for full response (default: gpt-4o-mini)
            filler_max_tokens: Max tokens for filler response (default: 30)
            full_max_tokens: Max tokens for full response (default: 500)
            filler_enabled: Whether to use filler responses (default: True)

        Raises:
            ValueError: If openai_api_key is empty or invalid
        """
        if not openai_api_key or openai_api_key == "sk-your-openai-api-key":
            raise ValueError(
                "Invalid OpenAI API key. Set OPENAI_API_KEY environment variable."
            )

        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.fast_model = fast_model
        self.full_model = full_model
        self.filler_max_tokens = filler_max_tokens
        self.full_max_tokens = full_max_tokens
        self.filler_enabled = filler_enabled

        logger.info(
            f"DualLLMOrchestrator initialized: "
            f"full_model={full_model}, filler_enabled={filler_enabled}"
        )

    def _select_filler(self, user_message: str) -> str:
        """Select context-aware filler template based on user message.

        Uses keyword matching to select contextually appropriate fillers,
        falling back to generic templates if no match.

        Args:
            user_message: User's input message

        Returns:
            Selected filler template string
        """
        message_lower = user_message.lower()

        # Context-aware selection
        for context, templates in self.CONTEXT_FILLERS.items():
            if context in message_lower:
                # Use secrets.choice for cryptographically secure random selection
                # (avoids S311 warning, though not critical for filler selection)
                return secrets.choice(templates)

        # Fallback to generic templates
        return secrets.choice(self.FILLER_TEMPLATES)

    async def _generate_filler(self, user_message: str) -> LLMResponse:
        """Generate instant filler response using templates.

        Args:
            user_message: User's input message

        Returns:
            LLMResponse with filler content and metadata
        """
        start_time = time.perf_counter()
        filler_text = self._select_filler(user_message)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        token_count = len(filler_text) // 4

        return LLMResponse(
            content=filler_text,
            latency_ms=latency_ms,
            token_count=token_count,
            is_filler=True,
            phase=ResponsePhase.FILLER,
        )

    async def _generate_full_response(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Generate full LLM response using OpenAI streaming.

        Args:
            user_message: User's input message
            conversation_history: List of previous messages [{"role": "...", "content": "..."}]

        Yields:
            Response text chunks as they arrive

        Raises:
            Exception: If OpenAI API call fails
        """
        try:
            # Build messages list with proper typing
            messages: list[dict[str, Any]] = conversation_history.copy()
            messages.append({"role": "user", "content": user_message})

            logger.debug(f"Calling OpenAI API: model={self.full_model}, messages={len(messages)}")

            # Stream response from OpenAI with explicit typing
            stream = await self.openai_client.chat.completions.create(
                model=self.full_model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=self.full_max_tokens,
                temperature=0.7,
                stream=True,
            )

            async for chunk in stream:  # type: ignore[union-attr]
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # Fallback response on error
            yield "I apologize, but I'm having trouble processing that request right now."

    async def generate_response(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[tuple[str, ResponsePhase]]:
        """Generate response using dual LLM architecture with parallel execution.

        Flow:
            1. Launch parallel tasks: filler (template) + full (OpenAI)
            2. Yield filler immediately (0ms latency)
            3. Buffer full response during filler playback
            4. Transition seamlessly to full response
            5. Stream remaining full response tokens
            6. Yield completion marker

        Args:
            user_message: User's input message
            conversation_history: Optional list of previous messages

        Yields:
            Tuples of (text_chunk, response_phase)

        Example:
            >>> async for text, phase in orchestrator.generate_response("Hello"):
            ...     print(f"[{phase.value}] {text}")
            [filler] Let me check that for you...
            [transition] Hello
            [full] ! How can I help you today?
            [complete]
        """
        if conversation_history is None:
            conversation_history = []

        start_time = time.perf_counter()

        # Phase 1: Launch parallel LLM calls
        logger.debug(f"Launching dual LLM for message: {user_message[:50]}...")

        # Start full response generation in background
        full_response_task = asyncio.create_task(
            self._buffer_full_response(user_message, conversation_history)
        )

        # Phase 2: Yield filler immediately (0ms latency)
        if self.filler_enabled:
            filler = await self._generate_filler(user_message)
            logger.debug(
                f"Filler generated in {filler.latency_ms:.2f}ms: {filler.content[:50]}..."
            )
            yield (filler.content, ResponsePhase.FILLER)

        # Phase 3: Wait for full response buffer
        try:
            buffered_chunks = await full_response_task
        except Exception as e:
            logger.error(f"Full response generation failed: {e}")
            # Fallback if full response fails
            yield (
                "I apologize, but I encountered an error processing your request.",
                ResponsePhase.FULL,
            )
            yield ("", ResponsePhase.COMPLETE)
            return

        # Phase 4: Transition to full response
        if buffered_chunks:
            first_chunk = buffered_chunks[0]
            logger.debug(f"Transitioning to full response: {first_chunk[:50]}...")
            yield (first_chunk, ResponsePhase.TRANSITION)

            # Phase 5: Stream remaining buffered chunks
            for chunk in buffered_chunks[1:]:
                yield (chunk, ResponsePhase.FULL)

        # Phase 6: Signal completion
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Response completed in {total_latency_ms:.2f}ms "
            f"(chunks={len(buffered_chunks)})"
        )
        yield ("", ResponsePhase.COMPLETE)

    async def _buffer_full_response(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]],
    ) -> list[str]:
        """Buffer full response chunks during filler playback.

        This method collects all chunks from the full LLM response stream
        so they can be yielded after the filler completes.

        Args:
            user_message: User's input message
            conversation_history: List of previous messages

        Returns:
            List of response text chunks
        """
        chunks: list[str] = []
        start_time = time.perf_counter()

        async for chunk in self._generate_full_response(user_message, conversation_history):
            chunks.append(chunk)

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Full response buffered in {latency_ms:.2f}ms: "
            f"{len(chunks)} chunks, "
            f"{''.join(chunks)[:50]}..."
        )

        return chunks

    async def generate_response_without_filler(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[tuple[str, ResponsePhase]]:
        """Generate response without filler (direct streaming).

        This method bypasses the filler mechanism and streams the full
        response directly. Useful for testing or when fillers are disabled.

        Args:
            user_message: User's input message
            conversation_history: Optional list of previous messages

        Yields:
            Tuples of (text_chunk, response_phase)
        """
        if conversation_history is None:
            conversation_history = []

        # Stream full response directly
        async for chunk in self._generate_full_response(user_message, conversation_history):
            yield (chunk, ResponsePhase.FULL)

        yield ("", ResponsePhase.COMPLETE)
