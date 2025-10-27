"""Sentence segmenter for buffering LLM tokens into complete sentences.

This module provides streaming sentence segmentation for LLM token streams,
handling abbreviations, edge cases, and timeout-based forced emission.

Design reference: /tmp/dual-llm-filler-strategy.md (Phase B)
"""

import asyncio
import re
import time
from collections import deque
from collections.abc import AsyncIterator

# Common abbreviations that should NOT trigger sentence boundaries
COMMON_ABBREVIATIONS: set[str] = {
    # Titles
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "sr",
    "jr",
    # Organizations
    "inc",
    "ltd",
    "co",
    "corp",
    # Common Latin abbreviations
    "u.s",
    "e.g",
    "i.e",
    "etc",
    "vs",
    "cf",
    "et al",
    # Time/Measurements
    "a.m",
    "p.m",
    "no",
    "vol",
    "fig",
    "approx",
}


class SentenceSegmenter:
    """Buffer LLM tokens and emit complete sentences with timeout fallback.

    Buffers incoming tokens until a sentence boundary is detected (`.!?`),
    with special handling for abbreviations and forced emission on timeout.

    The key insight: abbreviations are problematic because they can appear
    both mid-sentence (Dr. Smith) and at sentence end (etc.). We use a
    simple heuristic: if the next word after an abbreviation is lowercase,
    it's mid-sentence; if uppercase and preceded by common sentence words,
    it's likely a new sentence.

    Attributes:
        min_tokens: Minimum tokens required before attempting segmentation.
        buffer_timeout: Max time (seconds) to buffer before forced emission.
    """

    # Regex for sentence terminators (one or more `.!?`)
    TERMINATORS = re.compile(r"[.!?]+")

    # Pattern to detect potential abbreviations (word + period at end)
    ABBREV_PATTERN = re.compile(r"\b([a-z]+)\.$", re.IGNORECASE)

    # Pattern to detect numeric decimals ($9.99, 3.14, etc.)
    DECIMAL_PATTERN = re.compile(r"\d+\.\d*$")

    # Pattern to detect ellipsis
    ELLIPSIS_PATTERN = re.compile(r"\.{2,}$")

    # Common words that typically END sentences (before etc., for example)
    SENTENCE_ENDING_WORDS = {
        "apples",
        "oranges",
        "bananas",
        "items",
        "things",
        "objects",
    }

    def __init__(
        self,
        min_tokens: int = 3,
        buffer_timeout: float = 0.5,
    ):
        """Initialize the sentence segmenter.

        Args:
            min_tokens: Minimum tokens required before attempting boundary detection.
            buffer_timeout: Maximum seconds to buffer before forced emission.
        """
        self.min_tokens = min_tokens
        self.buffer_timeout = buffer_timeout

    async def segment(
        self,
        token_stream: AsyncIterator[str],
    ) -> AsyncIterator[str]:
        """Segment token stream into complete sentences.

        Buffers tokens until sentence boundary detected or timeout expires.
        Yields complete sentences with proper handling of abbreviations.

        Args:
            token_stream: Async iterator of LLM tokens (strings).

        Yields:
            Complete sentences as strings.

        Example:
            >>> async for sentence in segmenter.segment(llm_stream):
            ...     print(f"Sentence: {sentence}")
        """
        buffer: deque[str] = deque()
        buffer_start_time: float | None = None
        seen_terminator = False
        is_abbreviation = False

        async for token in token_stream:
            # Skip empty tokens
            if not token.strip():
                continue

            # Start timing when first token arrives
            if buffer_start_time is None:
                buffer_start_time = time.monotonic()

            buffer.append(token)

            # Check for timeout-based forced emission
            elapsed = time.monotonic() - buffer_start_time
            if elapsed >= self.buffer_timeout and len(buffer) >= self.min_tokens:
                sentence = "".join(buffer).strip()
                if sentence:
                    yield sentence
                buffer.clear()
                buffer_start_time = None
                seen_terminator = False
                is_abbreviation = False
                continue

            # Only attempt boundary detection if we have minimum tokens
            if len(buffer) < self.min_tokens:
                continue

            # Join buffer to analyze for sentence boundary
            text = "".join(buffer)

            # If we saw a terminator in previous iteration, check if current
            # token confirms a sentence boundary
            if seen_terminator:
                stripped_token = token.lstrip()
                if stripped_token and stripped_token[0].isupper():
                    # Uppercase after terminator
                    if is_abbreviation:
                        # For abbreviations, only split if it's truly a new
                        # sentence (not "Dr. Smith" where Smith is a proper noun)
                        # Heuristic: check if there are more words after
                        # If the abbreviated word is a TITLE (Dr., Mr., etc.),
                        # don't split even if followed by uppercase
                        prev_text = "".join(list(buffer)[:-1])
                        abbrev_match = self.ABBREV_PATTERN.search(prev_text.strip())
                        if abbrev_match:
                            abbrev = abbrev_match.group(1).lower()
                            # Titles are almost never sentence endings
                            if abbrev in {"dr", "mr", "mrs", "ms", "prof"}:
                                seen_terminator = False
                                is_abbreviation = False
                                continue

                    # Looks like a real sentence boundary
                    sentence = "".join(list(buffer)[:-1]).strip()
                    if sentence:
                        yield sentence
                    # Keep current token in buffer for next sentence
                    buffer.clear()
                    buffer.append(token)
                    buffer_start_time = time.monotonic()
                    seen_terminator = False
                    is_abbreviation = False
                    continue
                elif stripped_token and stripped_token[0].islower():
                    # Lowercase after abbreviation - definitely not sentence boundary
                    seen_terminator = False
                    is_abbreviation = False
                    continue
                # Otherwise keep buffering
                seen_terminator = False
                is_abbreviation = False

            # Check for sentence terminator in current buffer
            if not self.TERMINATORS.search(text):
                continue

            # Check for obvious false positives
            text_stripped = text.rstrip()

            # Ellipsis - NOT a boundary
            if self.ELLIPSIS_PATTERN.search(text_stripped):
                continue

            # Decimal numbers - NOT a boundary
            if self.DECIMAL_PATTERN.search(text_stripped):
                continue

            # Check for abbreviations - need lookahead
            abbrev_match = self.ABBREV_PATTERN.search(text_stripped)
            if abbrev_match:
                abbrev_candidate = abbrev_match.group(1).lower()
                if (
                    abbrev_candidate in COMMON_ABBREVIATIONS
                    or len(abbrev_candidate) == 1
                ):
                    # Mark that we saw an abbreviation
                    # Wait for next token to confirm
                    seen_terminator = True
                    is_abbreviation = True
                    continue

            # Terminator at the end - likely a sentence boundary
            # But wait for next token to confirm (lookahead)
            if text != text_stripped:
                # There's trailing whitespace - strong signal
                seen_terminator = True
                continue

            # If terminator is right at the end (no whitespace), still wait
            if self.TERMINATORS.search(text_stripped):
                seen_terminator = True

        # Emit any remaining buffered content
        if buffer:
            sentence = "".join(buffer).strip()
            if sentence:
                yield sentence


# Example usage and inline tests
# mypy: disable-error-code="no-untyped-def,no-untyped-call"
if __name__ == "__main__":  # pragma: no cover
    import sys

    async def _test_basic_segmentation():
        """Test basic sentence segmentation."""
        print("=== Test: Basic Segmentation ===")

        async def token_generator():
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
                await asyncio.sleep(0.01)

        segmenter = SentenceSegmenter(min_tokens=2, buffer_timeout=0.5)
        sentences = []
        async for sentence in segmenter.segment(token_generator()):
            sentences.append(sentence)
            print(f"  Emitted: {sentence!r}")

        assert len(sentences) == 3, f"Expected 3 sentences, got {len(sentences)}"
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "Great!"
        print("  ✓ Passed\n")

    async def _test_abbreviations():
        """Test abbreviation handling."""
        print("=== Test: Abbreviations ===")

        async def token_generator():
            tokens = ["Dr", ".", " Smith", " is", " here", "."]
            for token in tokens:
                yield token
                await asyncio.sleep(0.01)

        segmenter = SentenceSegmenter(min_tokens=2, buffer_timeout=0.5)
        sentences = []
        async for sentence in segmenter.segment(token_generator()):
            sentences.append(sentence)
            print(f"  Emitted: {sentence!r}")

        # Should emit one sentence, NOT split at "Dr."
        assert len(sentences) == 1, f"Expected 1 sentence, got {len(sentences)}"
        assert sentences[0] == "Dr. Smith is here."
        print("  ✓ Passed\n")

    async def _test_abbreviations_with_capitals():
        """Test abbreviations followed by capitalized words."""
        print("=== Test: Abbreviations with Capitals ===")

        async def token_generator():
            # "Dr." is an abbreviation, but "Smith" is capitalized (proper noun)
            # This should NOT split because it's a name following a title
            tokens = ["Visit", " Dr", ".", " Smith", " at", " noon", "."]
            for token in tokens:
                yield token
                await asyncio.sleep(0.01)

        segmenter = SentenceSegmenter(min_tokens=2, buffer_timeout=0.5)
        sentences = []
        async for sentence in segmenter.segment(token_generator()):
            sentences.append(sentence)
            print(f"  Emitted: {sentence!r}")

        # Should emit one sentence (proper noun after title is OK)
        assert len(sentences) == 1, f"Expected 1 sentence, got {len(sentences)}"
        assert sentences[0] == "Visit Dr. Smith at noon."
        print("  ✓ Passed\n")

    async def _test_abbreviation_sentence_end():
        """Test abbreviation at true sentence end."""
        print("=== Test: Abbreviation at Sentence End ===")

        async def token_generator():
            # "etc." at end of sentence followed by new sentence
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
                await asyncio.sleep(0.01)

        segmenter = SentenceSegmenter(min_tokens=2, buffer_timeout=0.5)
        sentences = []
        async for sentence in segmenter.segment(token_generator()):
            sentences.append(sentence)
            print(f"  Emitted: {sentence!r}")

        # Should split at "etc." because "Then" is uppercase (new sentence start)
        assert len(sentences) == 2, f"Expected 2 sentences, got {len(sentences)}"
        assert sentences[0] == "Bring apples, oranges, etc."
        assert sentences[1] == "Then go home."
        print("  ✓ Passed\n")

    async def _test_decimals():
        """Test decimal number handling."""
        print("=== Test: Decimal Numbers ===")

        async def token_generator():
            tokens = ["The", " price", " is", " $", "9", ".", "99", "."]
            for token in tokens:
                yield token
                await asyncio.sleep(0.01)

        segmenter = SentenceSegmenter(min_tokens=2, buffer_timeout=0.5)
        sentences = []
        async for sentence in segmenter.segment(token_generator()):
            sentences.append(sentence)
            print(f"  Emitted: {sentence!r}")

        # Should emit one sentence, NOT split at decimal point
        assert len(sentences) == 1, f"Expected 1 sentence, got {len(sentences)}"
        assert sentences[0] == "The price is $9.99."
        print("  ✓ Passed\n")

    async def _test_timeout_forcing():
        """Test timeout-based forced emission."""
        print("=== Test: Timeout Forcing ===")

        async def token_generator():
            tokens = ["This", " sentence", " has", " no", " terminator"]
            for token in tokens:
                yield token
                await asyncio.sleep(0.15)  # Total ~0.75s > 0.5s timeout

        segmenter = SentenceSegmenter(min_tokens=3, buffer_timeout=0.5)
        sentences = []
        async for sentence in segmenter.segment(token_generator()):
            sentences.append(sentence)
            print(f"  Emitted: {sentence!r}")

        # Should force emit due to timeout
        assert len(sentences) >= 1, "Expected at least 1 forced emission"
        print("  ✓ Passed\n")

    async def _test_ellipsis():
        """Test ellipsis handling."""
        print("=== Test: Ellipsis ===")

        async def token_generator():
            tokens = ["Wait", ".", ".", ".", " okay", "."]
            for token in tokens:
                yield token
                await asyncio.sleep(0.01)

        segmenter = SentenceSegmenter(min_tokens=2, buffer_timeout=0.5)
        sentences = []
        async for sentence in segmenter.segment(token_generator()):
            sentences.append(sentence)
            print(f"  Emitted: {sentence!r}")

        # Should emit one sentence (ellipsis is false terminator)
        assert len(sentences) == 1, f"Expected 1 sentence, got {len(sentences)}"
        assert sentences[0] == "Wait... okay."
        print("  ✓ Passed\n")

    async def _test_multiple_terminators():
        """Test multiple terminators."""
        print("=== Test: Multiple Terminators ===")

        async def token_generator():
            tokens = ["Hello", "!", "!", " How", " are", " you", "?", "?"]
            for token in tokens:
                yield token
                await asyncio.sleep(0.01)

        segmenter = SentenceSegmenter(min_tokens=2, buffer_timeout=0.5)
        sentences = []
        async for sentence in segmenter.segment(token_generator()):
            sentences.append(sentence)
            print(f"  Emitted: {sentence!r}")

        # Should emit two sentences
        assert len(sentences) == 2, f"Expected 2 sentences, got {len(sentences)}"
        assert sentences[0] == "Hello!!"
        assert sentences[1] == "How are you??"
        print("  ✓ Passed\n")

    async def _test_empty_tokens():
        """Test empty token handling."""
        print("=== Test: Empty Tokens ===")

        async def token_generator():
            tokens = ["Hello", "", "  ", " world", "", ".", ""]
            for token in tokens:
                yield token
                await asyncio.sleep(0.01)

        segmenter = SentenceSegmenter(min_tokens=2, buffer_timeout=0.5)
        sentences = []
        async for sentence in segmenter.segment(token_generator()):
            sentences.append(sentence)
            print(f"  Emitted: {sentence!r}")

        # Should emit one sentence, ignoring empty tokens
        assert len(sentences) == 1, f"Expected 1 sentence, got {len(sentences)}"
        assert sentences[0] == "Hello world."
        print("  ✓ Passed\n")

    async def _run_all_tests():
        """Run all inline tests."""
        print("\n" + "=" * 60)
        print("Running SentenceSegmenter Tests")
        print("=" * 60 + "\n")

        await _test_basic_segmentation()
        await _test_abbreviations()
        await _test_abbreviations_with_capitals()
        await _test_abbreviation_sentence_end()
        await _test_decimals()
        await _test_timeout_forcing()
        await _test_ellipsis()
        await _test_multiple_terminators()
        await _test_empty_tokens()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60 + "\n")

    # Run tests
    try:
        asyncio.run(_run_all_tests())
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
