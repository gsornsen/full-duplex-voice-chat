"""Transcript buffer for detecting and merging ASR continuations.

This module provides intelligent buffering and deduplication for ASR transcripts,
handling cases where ASR engines return partial results, duplicates, or
continuations of previous utterances.

Key features:
- Duplicate detection (exact and prefix matching)
- Intelligent continuation merging
- Automatic cleanup (TTL and size-based)
- Confidence-based merging decisions
- O(1) lookups for recent transcripts

Typical usage:
    buffer = TranscriptBuffer(max_size=10, ttl_seconds=30.0)

    # Add transcript, get merged text if duplicate/continuation
    merged = buffer.add("hello world", is_complete=True, confidence=0.95)
    if merged:
        # ASR returned continuation/duplicate, use merged text
        process_transcript(merged)
    else:
        # New unique transcript
        process_transcript("hello world")
"""

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class TranscriptEntry:
    """Single transcript entry with metadata.

    Attributes:
        text: The transcript text.
        timestamp: Unix timestamp when the entry was created.
        is_complete: Whether this is a complete utterance (vs partial).
        confidence: ASR confidence score (0.0-1.0).
    """
    text: str
    timestamp: float
    is_complete: bool
    confidence: float


class TranscriptBuffer:
    """Buffer for managing recent ASR transcripts with deduplication.

    This class maintains a fixed-size buffer of recent transcripts and provides
    intelligent merging of duplicates and continuations. Entries are automatically
    cleaned up based on TTL and buffer size.

    Attributes:
        buffer: Deque of recent transcript entries.
        ttl: Time-to-live for entries in seconds.
        max_size: Maximum number of entries to store.
    """

    def __init__(self, max_size: int = 10, ttl_seconds: float = 30.0):
        """Initialize transcript buffer.

        Args:
            max_size: Maximum number of entries to store. Oldest entries
                are automatically evicted when limit is reached.
            ttl_seconds: Time-to-live for entries in seconds. Entries older
                than TTL are removed during cleanup.
        """
        self.buffer: deque[TranscriptEntry] = deque(maxlen=max_size)
        self.ttl = ttl_seconds
        self.max_size = max_size

    def add(
        self,
        text: str,
        is_complete: bool = True,
        confidence: float = 1.0
    ) -> str | None:
        """Add transcript and return merged text if duplicate/continuation.

        This method performs the following operations:
        1. Clean expired entries (TTL-based)
        2. Check for exact duplicates
        3. Check for prefix/suffix continuations
        4. Merge if continuation detected
        5. Add new entry to buffer

        Args:
            text: The transcript text to add.
            is_complete: Whether this is a complete utterance.
            confidence: ASR confidence score (0.0-1.0).

        Returns:
            Merged text if this is a duplicate/continuation, None if new text.

        Examples:
            >>> buffer = TranscriptBuffer()
            >>> buffer.add("hello", is_complete=False, confidence=0.8)
            None
            >>> buffer.add("hello world", is_complete=True, confidence=0.95)
            'hello world'  # Continuation detected
        """
        # Clean old entries
        self._cleanup_expired()

        # Normalize input
        text = text.strip()
        if not text:
            return None

        current_time = time.time()

        # Check for duplicates and continuations
        for i, entry in enumerate(reversed(self.buffer)):
            # Only check recent entries (within TTL)
            if current_time - entry.timestamp > self.ttl:
                break

            # Exact duplicate - return original
            if entry.text == text:
                # Update timestamp and confidence if better
                if confidence > entry.confidence:
                    entry.confidence = confidence
                    entry.timestamp = current_time
                    entry.is_complete = is_complete
                return entry.text

            # Prefix match - continuation detected
            if text.startswith(entry.text):
                # Extract the continuation part
                continuation = text[len(entry.text):].strip()
                if continuation:
                    # Merge and update entry
                    merged_text = text
                    # Remove old entry and add updated one
                    actual_index = len(self.buffer) - 1 - i
                    del self.buffer[actual_index]
                    self.buffer.append(TranscriptEntry(
                        text=merged_text,
                        timestamp=current_time,
                        is_complete=is_complete,
                        confidence=max(confidence, entry.confidence)
                    ))
                    return merged_text
                else:
                    # Same text with whitespace difference
                    return entry.text

            # Suffix match - previous was partial
            if entry.text.startswith(text) and not entry.is_complete:
                # Current text is a prefix of previous partial - likely backtrack
                # Keep the longer (more complete) version
                if confidence >= entry.confidence:
                    # Update entry with higher confidence version
                    actual_index = len(self.buffer) - 1 - i
                    del self.buffer[actual_index]
                    self.buffer.append(TranscriptEntry(
                        text=entry.text,
                        timestamp=current_time,
                        is_complete=is_complete,
                        confidence=confidence
                    ))
                return entry.text

        # No duplicate/continuation found - add as new entry
        self.buffer.append(TranscriptEntry(
            text=text,
            timestamp=current_time,
            is_complete=is_complete,
            confidence=confidence
        ))
        return None

    def _cleanup_expired(self) -> None:
        """Remove entries older than TTL.

        This method removes entries from the buffer that are older than the
        configured TTL. Called automatically by add() to prevent unbounded growth.
        """
        current_time = time.time()
        # Remove from left (oldest) until we hit a non-expired entry
        while self.buffer and (current_time - self.buffer[0].timestamp) > self.ttl:
            self.buffer.popleft()

    def clear(self) -> None:
        """Clear all entries from the buffer."""
        self.buffer.clear()

    def get_recent(self, count: int = 5) -> list[str]:
        """Get N most recent transcript texts.

        Args:
            count: Number of recent transcripts to return.

        Returns:
            List of transcript texts, most recent first.
        """
        self._cleanup_expired()
        # Return most recent entries (from right of deque)
        recent = list(self.buffer)[-count:]
        recent.reverse()
        return [entry.text for entry in recent]

    def __len__(self) -> int:
        """Return number of entries in buffer."""
        return len(self.buffer)

    def __repr__(self) -> str:
        """Return string representation of buffer."""
        return (
            f"TranscriptBuffer(size={len(self.buffer)}, "
            f"max_size={self.max_size}, ttl={self.ttl}s)"
        )
