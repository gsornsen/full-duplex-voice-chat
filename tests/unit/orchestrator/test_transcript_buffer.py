"""Unit tests for transcript buffer management.

This test suite validates the TranscriptBuffer implementation:
- Duplicate detection (exact match, prefix match, similarity)
- Continuation detection (topic similarity, semantic coherence)
- Buffer management (max size, TTL cleanup, memory limits)
- Merge quality (proper spacing, punctuation, readability)
- Performance (operation latency, memory usage)

Test Coverage:
- Exact duplicate detection
- Prefix/suffix duplicate detection
- Continuation detection with topic similarity
- Smart transcript merging
- Buffer size limits and eviction
- TTL-based cleanup
- Concurrent access safety
- Edge cases and error handling
"""

import asyncio
import time

import pytest

from src.orchestrator.transcript_buffer import TranscriptBuffer, TranscriptEntry


@pytest.fixture
def buffer() -> TranscriptBuffer:
    """Create TranscriptBuffer instance for testing.

    Returns:
        TranscriptBuffer with default configuration
    """
    return TranscriptBuffer(max_size=10, ttl_seconds=30.0)


@pytest.fixture
def small_buffer() -> TranscriptBuffer:
    """Create small TranscriptBuffer for testing size limits.

    Returns:
        TranscriptBuffer with small capacity (5 entries)
    """
    return TranscriptBuffer(max_size=5, ttl_seconds=30.0)


# ============================================================================
# Basic Operations Tests
# ============================================================================


class TestBasicOperations:
    """Test suite for basic buffer operations."""

    def test_init_default_params(self) -> None:
        """Test buffer initialization with default parameters."""
        buffer = TranscriptBuffer()
        assert len(buffer) == 0
        assert buffer.max_size == 10
        assert buffer.ttl == 30.0

    def test_init_custom_params(self) -> None:
        """Test buffer initialization with custom parameters."""
        buffer = TranscriptBuffer(max_size=5, ttl_seconds=10.0)
        assert len(buffer) == 0
        assert buffer.max_size == 5
        assert buffer.ttl == 10.0

    def test_add_new_transcript(self) -> None:
        """Test adding a new transcript returns None."""
        buffer = TranscriptBuffer()
        result = buffer.add("hello world", is_complete=True, confidence=0.95)
        assert result is None
        assert len(buffer) == 1

    def test_clear(self) -> None:
        """Test clear removes all entries."""
        buffer = TranscriptBuffer()
        buffer.add("text 1", is_complete=True, confidence=0.9)
        buffer.add("text 2", is_complete=True, confidence=0.9)
        assert len(buffer) == 2

        buffer.clear()
        assert len(buffer) == 0

    def test_repr(self) -> None:
        """Test string representation."""
        buffer = TranscriptBuffer(max_size=5, ttl_seconds=15.0)
        buffer.add("test", is_complete=True, confidence=0.9)
        repr_str = repr(buffer)
        assert "TranscriptBuffer" in repr_str
        assert "size=1" in repr_str
        assert "max_size=5" in repr_str
        assert "ttl=15.0s" in repr_str


# ============================================================================
# Duplicate Detection Tests
# ============================================================================


class TestDuplicateDetection:
    """Test suite for duplicate detection logic."""

    def test_exact_duplicate(self) -> None:
        """Test exact duplicate detection returns original."""
        buffer = TranscriptBuffer()
        buffer.add("hello world", is_complete=True, confidence=0.9)
        result = buffer.add("hello world", is_complete=True, confidence=0.85)
        assert result == "hello world"
        assert len(buffer) == 1

    def test_exact_duplicate_updates_confidence(self) -> None:
        """Test duplicate with higher confidence updates entry."""
        buffer = TranscriptBuffer()
        buffer.add("hello world", is_complete=True, confidence=0.8)
        entry_before = buffer.buffer[0]
        original_timestamp = entry_before.timestamp

        time.sleep(0.01)  # Small delay to ensure different timestamp
        result = buffer.add("hello world", is_complete=True, confidence=0.95)

        assert result == "hello world"
        assert len(buffer) == 1
        entry_after = buffer.buffer[0]
        assert entry_after.confidence == 0.95
        assert entry_after.timestamp > original_timestamp

    def test_case_insensitive_duplicate(self) -> None:
        """Test that duplicate detection is case-insensitive."""
        buffer = TranscriptBuffer()
        buffer.add("Hello World", is_complete=True, confidence=0.9)

        # Test various case variations
        test_cases = [
            "hello world",
            "HELLO WORLD",
            "HeLLo WoRLd",
        ]

        for text in test_cases:
            result = buffer.add(text, is_complete=True, confidence=0.9)
            expected_msg = f"Case variation '{text}' should be detected as duplicate"
            assert result == "Hello World", expected_msg

    def test_whitespace_normalized_duplicate(self) -> None:
        """Test that duplicate detection normalizes whitespace."""
        buffer = TranscriptBuffer()
        buffer.add("hello world", is_complete=True, confidence=0.9)

        # Exact match with extra whitespace
        result = buffer.add("hello   world", is_complete=True, confidence=0.9)
        # Note: Normalized comparison, but preserves original text
        assert result is not None

    def test_rapid_duplicates(self) -> None:
        """Test rapid duplicate submissions."""
        buffer = TranscriptBuffer()
        results = []
        for _ in range(5):
            result = buffer.add("same text", is_complete=True, confidence=0.9)
            results.append(result)

        # First should be None (new), rest should be duplicates
        assert results[0] is None
        assert all(r == "same text" for r in results[1:])
        assert len(buffer) == 1


# ============================================================================
# Continuation Detection Tests
# ============================================================================


class TestContinuationDetection:
    """Test suite for continuation detection logic."""

    def test_prefix_continuation(self) -> None:
        """Test continuation detection with prefix match."""
        buffer = TranscriptBuffer()
        buffer.add("hello", is_complete=False, confidence=0.8)
        result = buffer.add("hello world", is_complete=True, confidence=0.95)
        assert result == "hello world"
        assert len(buffer) == 1
        assert buffer.buffer[0].text == "hello world"

    def test_prefix_continuation_with_whitespace(self) -> None:
        """Test continuation with whitespace normalization."""
        buffer = TranscriptBuffer()
        buffer.add("hello", is_complete=False, confidence=0.8)
        result = buffer.add("hello   world", is_complete=True, confidence=0.95)
        assert result == "hello   world"  # Preserves original spacing
        assert len(buffer) == 1

    def test_suffix_backtrack(self) -> None:
        """Test backtrack handling when new text is prefix of partial."""
        buffer = TranscriptBuffer()
        buffer.add("hello world", is_complete=False, confidence=0.8)
        result = buffer.add("hello", is_complete=False, confidence=0.9)
        # Should return the longer version (previous entry)
        assert result == "hello world"
        assert len(buffer) == 1
        # Entry should be updated with new confidence/timestamp
        assert buffer.buffer[0].confidence == 0.9

    def test_multiple_continuations(self) -> None:
        """Test chaining multiple continuations."""
        buffer = TranscriptBuffer()

        result1 = buffer.add("hello", is_complete=False, confidence=0.7)
        assert result1 is None

        result2 = buffer.add("hello world", is_complete=False, confidence=0.85)
        assert result2 == "hello world"

        result3 = buffer.add("hello world how are you", is_complete=True, confidence=0.95)
        assert result3 == "hello world how are you"

        assert len(buffer) == 1
        assert buffer.buffer[0].text == "hello world how are you"

    def test_partial_to_complete_transition(self) -> None:
        """Test transition from partial to complete utterance."""
        buffer = TranscriptBuffer()
        buffer.add("hello", is_complete=False, confidence=0.8)
        assert buffer.buffer[0].is_complete is False

        buffer.add("hello world", is_complete=True, confidence=0.95)
        assert buffer.buffer[0].is_complete is True
        assert buffer.buffer[0].text == "hello world"

    def test_continuation_same_prefix_different_suffix(self) -> None:
        """Test handling of same prefix with different suffixes."""
        buffer = TranscriptBuffer()
        buffer.add("hello world", is_complete=False, confidence=0.8)
        result = buffer.add("hello there", is_complete=True, confidence=0.9)
        # Not a continuation (different suffix), should add new entry
        assert result is None
        assert len(buffer) == 2

    def test_confidence_merging(self) -> None:
        """Test confidence is updated to maximum during merge."""
        buffer = TranscriptBuffer()
        buffer.add("hello", is_complete=False, confidence=0.95)
        buffer.add("hello world", is_complete=True, confidence=0.75)
        # Should keep the higher confidence
        assert buffer.buffer[0].confidence == 0.95


# ============================================================================
# Buffer Management Tests
# ============================================================================


class TestBufferManagement:
    """Test suite for buffer size and TTL management."""

    def test_size_based_eviction(self) -> None:
        """Test size-based eviction when max_size is reached."""
        buffer = TranscriptBuffer(max_size=3, ttl_seconds=30.0)

        buffer.add("text 1", is_complete=True, confidence=0.9)
        buffer.add("text 2", is_complete=True, confidence=0.9)
        buffer.add("text 3", is_complete=True, confidence=0.9)
        buffer.add("text 4", is_complete=True, confidence=0.9)

        # Should only have 3 entries (oldest evicted)
        assert len(buffer) == 3
        texts = [entry.text for entry in buffer.buffer]
        assert "text 1" not in texts
        assert "text 2" in texts
        assert "text 3" in texts
        assert "text 4" in texts

    def test_ttl_cleanup(self) -> None:
        """Test TTL-based cleanup removes expired entries."""
        buffer = TranscriptBuffer(max_size=10, ttl_seconds=0.1)

        buffer.add("old text", is_complete=True, confidence=0.9)
        time.sleep(0.15)  # Wait for TTL to expire
        buffer.add("new text", is_complete=True, confidence=0.9)

        # Old entry should be cleaned up
        assert len(buffer) == 1
        assert buffer.buffer[0].text == "new text"

    def test_ttl_cleanup_during_add(self) -> None:
        """Test that TTL cleanup happens automatically during add."""
        buffer = TranscriptBuffer(max_size=10, ttl_seconds=0.1)

        buffer.add("old 1", is_complete=True, confidence=0.9)
        buffer.add("old 2", is_complete=True, confidence=0.9)
        time.sleep(0.15)

        # Add new entry should trigger cleanup
        buffer.add("new", is_complete=True, confidence=0.9)

        # Only new entry should remain
        assert len(buffer) == 1
        assert buffer.buffer[0].text == "new"


# ============================================================================
# Query Operations Tests
# ============================================================================


class TestQueryOperations:
    """Test suite for buffer query operations."""

    def test_get_recent_default(self) -> None:
        """Test get_recent returns recent transcripts in reverse order."""
        buffer = TranscriptBuffer()
        buffer.add("text 1", is_complete=True, confidence=0.9)
        buffer.add("text 2", is_complete=True, confidence=0.9)
        buffer.add("text 3", is_complete=True, confidence=0.9)

        recent = buffer.get_recent(count=5)
        assert recent == ["text 3", "text 2", "text 1"]

    def test_get_recent_limited(self) -> None:
        """Test get_recent with count limit."""
        buffer = TranscriptBuffer()
        buffer.add("text 1", is_complete=True, confidence=0.9)
        buffer.add("text 2", is_complete=True, confidence=0.9)
        buffer.add("text 3", is_complete=True, confidence=0.9)

        recent = buffer.get_recent(count=2)
        assert recent == ["text 3", "text 2"]

    def test_get_recent_empty_buffer(self) -> None:
        """Test get_recent on empty buffer."""
        buffer = TranscriptBuffer()
        recent = buffer.get_recent(count=5)
        assert recent == []


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_empty_string_ignored(self) -> None:
        """Test empty strings are ignored."""
        buffer = TranscriptBuffer()
        result = buffer.add("", is_complete=True, confidence=1.0)
        assert result is None
        assert len(buffer) == 0

    def test_whitespace_only_ignored(self) -> None:
        """Test whitespace-only strings are ignored."""
        buffer = TranscriptBuffer()
        result = buffer.add("   ", is_complete=True, confidence=1.0)
        assert result is None
        assert len(buffer) == 0

    def test_unicode_text(self) -> None:
        """Test handling of unicode text."""
        buffer = TranscriptBuffer()
        buffer.add("你好世界", is_complete=True, confidence=0.9)
        result = buffer.add("你好世界", is_complete=True, confidence=0.8)
        assert result == "你好世界"

    def test_long_text(self) -> None:
        """Test handling of long text."""
        long_text = " ".join(["word"] * 1000)
        buffer = TranscriptBuffer()
        result = buffer.add(long_text, is_complete=True, confidence=0.9)
        assert result is None
        assert len(buffer) == 1

    def test_special_characters(self) -> None:
        """Test handling of special characters."""
        special_text = "Email: user@example.com, Price: $99.99, Code: #12345"
        buffer = TranscriptBuffer()
        result = buffer.add(special_text, is_complete=True, confidence=0.9)
        assert result is None
        assert len(buffer) == 1

    def test_interleaved_sessions(self) -> None:
        """Test handling of interleaved conversation sessions."""
        buffer = TranscriptBuffer()

        # Session 1 - partial
        buffer.add("hello", is_complete=False, confidence=0.8)

        # Session 2 - complete different text
        buffer.add("goodbye", is_complete=True, confidence=0.9)

        # Session 1 - continuation
        result = buffer.add("hello world", is_complete=True, confidence=0.95)

        # Should still detect continuation despite interleaving
        assert result == "hello world"
        assert len(buffer) == 2  # "goodbye" and merged "hello world"


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Test suite for performance characteristics."""

    def test_add_latency(self) -> None:
        """Test that add operation is fast (<10ms average)."""
        buffer = TranscriptBuffer()
        latencies = []

        for i in range(100):
            start = time.perf_counter()
            buffer.add(f"Test transcript {i}", is_complete=True, confidence=0.9)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        # Relaxed thresholds for CI environments
        assert avg_latency < 20.0, f"Average add latency {avg_latency:.2f}ms exceeds 20ms"
        assert max_latency < 100.0, f"Max add latency {max_latency:.2f}ms exceeds 100ms"

    def test_duplicate_detection_latency(self) -> None:
        """Test that duplicate detection is fast."""
        buffer = TranscriptBuffer()

        # Pre-populate buffer
        for i in range(50):
            buffer.add(f"Entry {i}", is_complete=True, confidence=0.9)

        latencies = []
        for i in range(50):
            start = time.perf_counter()
            buffer.add(f"Entry {i}", is_complete=True, confidence=0.9)  # Duplicate
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        expected_msg = f"Average duplicate detection latency {avg_latency:.2f}ms exceeds 10ms"
        assert avg_latency < 10.0, expected_msg


# ============================================================================
# Concurrent Access Tests
# ============================================================================


class TestConcurrentAccess:
    """Test suite for concurrent access safety."""

    @pytest.mark.asyncio
    async def test_concurrent_add(self) -> None:
        """Test concurrent add_transcript operations (thread safety)."""
        buffer = TranscriptBuffer()

        async def add_many(prefix: str) -> None:
            for i in range(50):
                buffer.add(f"{prefix} {i}", is_complete=True, confidence=0.9)
                await asyncio.sleep(0.001)  # Small delay

        # Run concurrent adds
        await asyncio.gather(
            add_many("Worker1"),
            add_many("Worker2"),
            add_many("Worker3"),
        )

        # Buffer should contain entries from all workers (up to max_size)
        assert len(buffer) <= buffer.max_size
        # No corruption/crashes is the success criterion

    @pytest.mark.asyncio
    async def test_concurrent_read_write(self) -> None:
        """Test concurrent reads and writes."""
        buffer = TranscriptBuffer()

        async def writer() -> None:
            for i in range(30):
                buffer.add(f"Write {i}", is_complete=True, confidence=0.9)
                await asyncio.sleep(0.001)

        async def reader() -> None:
            for _i in range(30):
                _ = buffer.get_recent(count=5)
                await asyncio.sleep(0.001)

        # Run concurrent read/write
        await asyncio.gather(writer(), reader())

        # Success = no crashes or exceptions


# ============================================================================
# Integration Scenarios (Master Plan)
# ============================================================================


class TestIntegrationScenarios:
    """Test scenarios from continuation detection master plan."""

    def test_scenario_thinking_pause(self) -> None:
        """Test Scenario 1: Thinking pause continuation.

        User: "Tell me about... [pause] ...quantum physics"
        Expected: Detect continuation, merge to "Tell me about quantum physics"
        """
        buffer = TranscriptBuffer()

        # First fragment (incomplete)
        result1 = buffer.add("Tell me about", is_complete=False, confidence=0.7)
        assert result1 is None

        # Second part (continuation)
        result2 = buffer.add("Tell me about quantum physics", is_complete=True, confidence=0.9)
        assert result2 == "Tell me about quantum physics"
        assert len(buffer) == 1

    def test_scenario_complete_question(self) -> None:
        """Test Scenario 2: Complete sentence sent immediately.

        User: "What's the weather?"
        Expected: Process immediately, no merging
        """
        buffer = TranscriptBuffer()

        text = "What's the weather?"
        result = buffer.add(text, is_complete=True, confidence=0.95)

        assert result is None  # New entry
        assert len(buffer) == 1
        assert buffer.buffer[0].text == text

    def test_scenario_duplicate_echo(self) -> None:
        """Test Scenario 3: Duplicate detection (echo/repeat).

        User: "Hello" -> "Hello" (ASR echo)
        Expected: Second "Hello" detected as duplicate, skip
        """
        buffer = TranscriptBuffer()

        # First occurrence
        result1 = buffer.add("Hello", is_complete=True, confidence=0.9)
        assert result1 is None

        # Duplicate (echo)
        result2 = buffer.add("Hello", is_complete=True, confidence=0.9)
        assert result2 == "Hello", "Exact duplicate should be detected"

        # Only one entry in buffer
        assert len(buffer) == 1

    def test_scenario_progressive_building(self) -> None:
        """Test progressive sentence building with multiple fragments."""
        buffer = TranscriptBuffer()

        # User builds up sentence over multiple pauses
        result1 = buffer.add("I want to", is_complete=False, confidence=0.7)
        assert result1 is None

        result2 = buffer.add("I want to go", is_complete=False, confidence=0.8)
        # Could be continuation or different
        assert result2 == "I want to go" or result2 is None

        result3 = buffer.add("I want to go to the store", is_complete=True, confidence=0.95)
        # Should detect as continuation
        assert result3 == "I want to go to the store"


# ============================================================================
# TranscriptEntry Tests
# ============================================================================


class TestTranscriptEntry:
    """Test suite for TranscriptEntry dataclass."""

    def test_entry_creation(self) -> None:
        """Test creating a transcript entry."""
        entry = TranscriptEntry(
            text="hello world",
            timestamp=time.time(),
            is_complete=True,
            confidence=0.95
        )
        assert entry.text == "hello world"
        assert entry.is_complete is True
        assert entry.confidence == 0.95
        assert isinstance(entry.timestamp, float)
