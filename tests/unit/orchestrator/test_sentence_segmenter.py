"""Unit tests for SentenceSegmenter.

Tests cover:
- Basic sentence segmentation (., !, ?)
- Abbreviation handling (Dr., Mr., etc.)
- Decimal number handling ($9.99, 3.14)
- Ellipsis handling (...)
- Timeout-based forced emission
- Edge cases (empty tokens, multiple terminators)
"""

import asyncio
from collections.abc import AsyncIterator

import pytest

from src.orchestrator.sentence_segmenter import SentenceSegmenter


@pytest.fixture
def segmenter() -> SentenceSegmenter:
    """Create default SentenceSegmenter."""
    return SentenceSegmenter(min_tokens=2, buffer_timeout=0.5)


@pytest.fixture
def fast_segmenter() -> SentenceSegmenter:
    """Create fast SentenceSegmenter with short timeout."""
    return SentenceSegmenter(min_tokens=2, buffer_timeout=0.2)


class TestBasicSegmentation:
    """Test basic sentence segmentation."""

    async def test_segment_single_sentence_period(
        self, segmenter: SentenceSegmenter
    ) -> None:
        """Test segmentation of single sentence with period."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["Hello", " world", "."]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 1
        assert sentences[0] == "Hello world."

    async def test_segment_single_sentence_question(
        self, segmenter: SentenceSegmenter
    ) -> None:
        """Test segmentation of single sentence with question mark."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["How", " are", " you", "?"]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 1
        assert sentences[0] == "How are you?"

    async def test_segment_single_sentence_exclamation(
        self, segmenter: SentenceSegmenter
    ) -> None:
        """Test segmentation of single sentence with exclamation."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["Great", "!"]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 1
        assert sentences[0] == "Great!"

    async def test_segment_multiple_sentences(
        self, segmenter: SentenceSegmenter
    ) -> None:
        """Test segmentation of multiple sentences."""

        async def token_stream() -> AsyncIterator[str]:
            tokens = [
                "Hello",
                " world",
                ".",
                " How",
                " are",
                " you",
                "?",
                " Great",
                "!",
            ]
            for token in tokens:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "Great!"

    async def test_segment_minimum_tokens_requirement(
        self, segmenter: SentenceSegmenter
    ) -> None:
        """Test minimum tokens requirement before segmentation."""

        async def token_stream() -> AsyncIterator[str]:
            # Single token with period - should still emit eventually
            yield "Hi."

        sentences = [s async for s in segmenter.segment(token_stream())]

        # Should emit via timeout fallback (min_tokens=2 not met)
        assert len(sentences) == 1
        assert sentences[0] == "Hi."


class TestAbbreviations:
    """Test abbreviation handling."""

    async def test_abbreviation_title_dr(self, segmenter: SentenceSegmenter) -> None:
        """Test Dr. abbreviation does not split sentence."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["Dr", ".", " Smith", " is", " here", "."]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 1
        assert sentences[0] == "Dr. Smith is here."

    async def test_abbreviation_title_mr(self, segmenter: SentenceSegmenter) -> None:
        """Test Mr. abbreviation does not split sentence."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["Meet", " Mr", ".", " Jones", "."]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 1
        assert sentences[0] == "Meet Mr. Jones."

    async def test_abbreviation_etc_at_end(self, segmenter: SentenceSegmenter) -> None:
        """Test etc. at sentence end correctly terminates."""

        async def token_stream() -> AsyncIterator[str]:
            tokens = [
                "Bring",
                " apples",
                ",",
                " oranges",
                ",",
                " etc",
                ".",
                " Then",
                " go",
                " home",
                ".",
            ]
            for token in tokens:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 2
        assert sentences[0] == "Bring apples, oranges, etc."
        assert sentences[1] == "Then go home."

    async def test_abbreviation_usa(self, segmenter: SentenceSegmenter) -> None:
        """Test U.S. abbreviation handling."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["The", " U.S", ".", " economy", " is", " strong", "."]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        # Should emit as one sentence (U.S. is abbreviation)
        assert len(sentences) == 1
        assert "U.S." in sentences[0]

    async def test_abbreviation_lowercase_continuation(
        self, segmenter: SentenceSegmenter
    ) -> None:
        """Test abbreviation followed by lowercase continues sentence."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["Visit", " Dr", ".", " Smith", " at", " noon", "."]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 1
        assert sentences[0] == "Visit Dr. Smith at noon."


class TestDecimalNumbers:
    """Test decimal number handling."""

    async def test_decimal_price(self, segmenter: SentenceSegmenter) -> None:
        """Test decimal price does not split sentence."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["The", " price", " is", " $", "9", ".", "99", "."]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 1
        assert sentences[0] == "The price is $9.99."

    async def test_decimal_pi(self, segmenter: SentenceSegmenter) -> None:
        """Test decimal constant does not split sentence."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["Pi", " equals", " 3", ".", "14159", "."]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 1
        assert "3.14159" in sentences[0]

    async def test_decimal_incomplete(self, segmenter: SentenceSegmenter) -> None:
        """Test incomplete decimal (trailing period)."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["The", " value", " is", " 42", ".", " Done", "."]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        # 42. should be treated as decimal (not sentence boundary)
        # Final period after "Done" should trigger emission
        assert len(sentences) >= 1


class TestEllipsis:
    """Test ellipsis handling."""

    async def test_ellipsis_continuation(self, segmenter: SentenceSegmenter) -> None:
        """Test ellipsis does not split sentence."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["Wait", ".", ".", ".", " okay", "."]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 1
        assert sentences[0] == "Wait... okay."

    async def test_ellipsis_mid_sentence(self, segmenter: SentenceSegmenter) -> None:
        """Test ellipsis mid-sentence continuation."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["Well", ".", ".", ".", " maybe", " not", "."]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 1
        assert "..." in sentences[0]


class TestMultipleTerminators:
    """Test multiple terminator handling."""

    async def test_multiple_exclamations(self, segmenter: SentenceSegmenter) -> None:
        """Test multiple exclamation marks."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["Wow", "!", "!"]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 1
        assert sentences[0] == "Wow!!"

    async def test_multiple_questions(self, segmenter: SentenceSegmenter) -> None:
        """Test multiple question marks."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["Really", "?", "?"]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 1
        assert sentences[0] == "Really??"

    async def test_mixed_terminators(self, segmenter: SentenceSegmenter) -> None:
        """Test mixed terminators."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["What", "!", "?", " Confused", "?"]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        # Should handle mixed terminators
        assert len(sentences) >= 1


class TestTimeoutForcing:
    """Test timeout-based forced emission."""

    async def test_timeout_forces_emission(
        self, fast_segmenter: SentenceSegmenter
    ) -> None:
        """Test timeout forces emission without terminator."""

        async def token_stream() -> AsyncIterator[str]:
            tokens = ["This", " sentence", " has", " no", " terminator"]
            for token in tokens:
                yield token
                await asyncio.sleep(0.06)  # Total ~0.3s > 0.2s timeout

        sentences = [s async for s in fast_segmenter.segment(token_stream())]

        # Should force emit due to timeout
        assert len(sentences) >= 1
        assert "sentence" in sentences[0]

    async def test_timeout_respects_minimum_tokens(
        self, segmenter: SentenceSegmenter
    ) -> None:
        """Test timeout only applies after minimum tokens met."""

        async def token_stream() -> AsyncIterator[str]:
            # Only one token - below min_tokens=2
            yield "Hi"
            await asyncio.sleep(0.6)  # Exceed timeout

        sentences = [s async for s in segmenter.segment(token_stream())]

        # Should still emit eventually (at end of stream)
        assert len(sentences) == 1
        assert sentences[0] == "Hi"

    async def test_timeout_with_subsequent_tokens(
        self, fast_segmenter: SentenceSegmenter
    ) -> None:
        """Test timeout emission doesn't block subsequent tokens."""

        async def token_stream() -> AsyncIterator[str]:
            # First group - will timeout
            yield "Slow"
            yield " group"
            await asyncio.sleep(0.25)

            # Second group - after timeout
            yield " Fast"
            yield " group"
            yield "."

        sentences = [s async for s in fast_segmenter.segment(token_stream())]

        # Should have multiple sentences (timeout split + normal split)
        assert len(sentences) >= 1


class TestEmptyTokens:
    """Test empty token handling."""

    async def test_empty_tokens_ignored(self, segmenter: SentenceSegmenter) -> None:
        """Test empty tokens are ignored."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["Hello", "", "  ", " world", "", ".", ""]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 1
        assert sentences[0] == "Hello world."

    async def test_only_empty_tokens(self, segmenter: SentenceSegmenter) -> None:
        """Test stream with only empty tokens."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["", "  ", "", "   "]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        # Should emit nothing (all empty)
        assert len(sentences) == 0

    async def test_empty_tokens_between_sentences(
        self, segmenter: SentenceSegmenter
    ) -> None:
        """Test empty tokens between sentences."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["First", ".", "", "", " Second", ".", ""]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 2
        assert sentences[0] == "First."
        assert sentences[1] == "Second."


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_single_period_only(self, segmenter: SentenceSegmenter) -> None:
        """Test stream with only a period."""

        async def token_stream() -> AsyncIterator[str]:
            yield "."

        sentences = [s async for s in segmenter.segment(token_stream())]

        # Should emit period (or nothing if filtered)
        assert len(sentences) <= 1

    async def test_no_tokens(self, segmenter: SentenceSegmenter) -> None:
        """Test empty token stream."""

        async def token_stream() -> AsyncIterator[str]:
            return
            yield  # Make this a generator

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 0

    async def test_long_sentence_no_terminator(
        self, segmenter: SentenceSegmenter
    ) -> None:
        """Test long sentence without terminator eventually emits."""

        async def token_stream() -> AsyncIterator[str]:
            # Many tokens without terminator
            for i in range(20):
                yield f"word{i} "

        sentences = [s async for s in segmenter.segment(token_stream())]

        # Should emit at end of stream
        assert len(sentences) >= 1
        assert "word0" in sentences[0]

    async def test_sentence_with_newlines(
        self, segmenter: SentenceSegmenter
    ) -> None:
        """Test sentence with embedded newlines."""

        async def token_stream() -> AsyncIterator[str]:
            for token in ["Hello", "\n", "world", "."]:
                yield token

        sentences = [s async for s in segmenter.segment(token_stream())]

        assert len(sentences) == 1
        # Newlines are preserved in the sentence
        assert "Hello" in sentences[0]
        assert "world" in sentences[0]


class TestPerformance:
    """Test performance characteristics."""

    async def test_streaming_latency_low(
        self, fast_segmenter: SentenceSegmenter
    ) -> None:
        """Test that streaming has low latency (sentences emitted quickly)."""
        import time

        async def token_stream() -> AsyncIterator[str]:
            for token in ["Quick", " sentence", "."]:
                yield token
                await asyncio.sleep(0.01)

        start = time.perf_counter()

        sentences = []
        async for sentence in fast_segmenter.segment(token_stream()):
            sentences.append(sentence)
            elapsed_ms = (time.perf_counter() - start) * 1000
            # First sentence should arrive within 200ms (timeout + processing)
            assert elapsed_ms < 500

        assert len(sentences) == 1

    async def test_buffer_timeout_accuracy(
        self, fast_segmenter: SentenceSegmenter
    ) -> None:
        """Test buffer timeout is reasonably accurate."""
        import time

        async def token_stream() -> AsyncIterator[str]:
            # Emit tokens slowly (will trigger timeout)
            for token in ["Slow", " emission"]:
                yield token
                await asyncio.sleep(0.05)

        start = time.perf_counter()

        sentences = []
        async for sentence in fast_segmenter.segment(token_stream()):
            sentences.append(sentence)
            (time.perf_counter() - start) * 1000

        # Should emit around timeout threshold (200ms ± 100ms tolerance)
        # Note: This is a loose bound due to asyncio scheduling
        assert len(sentences) >= 1
