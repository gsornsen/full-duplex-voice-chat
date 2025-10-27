"""Heuristic-based speech continuation detection.

This module implements Phase 1 continuation detection using fast linguistic
and syntactic heuristics to classify utterance completeness. Designed for
<1ms latency with 75-80% target accuracy.

The detector analyzes transcribed text to determine if the user is likely to
continue speaking after a pause, enabling intelligent buffering of multi-segment
speech for more natural conversation flow.

Example:
    ```python
    detector = ContinuationDetector()

    # Incomplete utterance (trailing preposition)
    is_complete, confidence = detector.is_complete("I'm going to")
    # Returns: (False, 0.10) - low confidence (incomplete)

    # Complete utterance
    is_complete, confidence = detector.is_complete("I'm going home.")
    # Returns: (True, 0.95) - high confidence (complete)
    ```
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompletionResult:
    """Result of utterance completion analysis.

    Attributes:
        is_complete: True if utterance appears complete, False if likely to continue
        confidence: Confidence score 0.0-1.0 (higher = more confident)
        reason: Human-readable reason for the classification
    """

    is_complete: bool
    confidence: float
    reason: str


class ContinuationDetector:
    """Fast heuristic-based continuation detector.

    Analyzes speech transcripts to determine if the user is likely to continue
    speaking after a pause. Uses multiple linguistic and syntactic heuristics:

    1. Terminal punctuation (95% confidence complete)
    2. Common closing phrases (95% confidence complete)
    3. Trailing conjunctions/prepositions (90% confidence incomplete)
    4. Unbalanced delimiters (95% confidence incomplete)
    5. Question words + structure (90% confidence complete)
    6. Short fragments <3 words (85% confidence incomplete)

    Thread-safety: Thread-safe (stateless, read-only configuration).

    Performance: <1ms per classification (no ML dependencies).

    Confidence Scoring:
        - Complete sentences: confidence = likelihood it's complete
          (0.0 = definitely incomplete, 1.0 = definitely complete)
        - Incomplete sentences: confidence = 1.0 - likelihood
          (so 0.1 confidence means very likely incomplete)
    """

    # Trailing words that strongly indicate incomplete utterance
    TRAILING_INCOMPLETE = frozenset(
        [
            # Prepositions
            "to",
            "from",
            "in",
            "on",
            "at",
            "by",
            "with",
            "about",
            "for",
            "of",
            "into",
            "onto",
            "under",
            "over",
            "through",
            "between",
            "among",
            # Conjunctions
            "and",
            "but",
            "or",
            "so",
            "because",
            "since",
            "while",
            "although",
            "if",
            "unless",
            "until",
            "when",
            "where",
            "whereas",
            # Articles/Determiners
            "the",
            "a",
            "an",
            "this",
            "that",
            "these",
            "those",
            # Auxiliary verbs
            "is",
            "are",
            "was",
            "were",
            "will",
            "would",
            "can",
            "could",
            "should",
            "shall",
            "may",
            "might",
            "must",
            "have",
            "has",
            "had",
            # Modals and helping verbs
            "do",
            "does",
            "did",
            "being",
            "been",
            "am",
        ]
    )

    # Closing phrases that indicate complete thought
    CLOSING_PHRASES = frozenset(
        [
            "thanks",
            "thank you",
            "thanks a lot",
            "thank you very much",
            "bye",
            "goodbye",
            "see you",
            "talk to you later",
            "catch you later",
            "that's it",
            "that's all",
            "okay",
            "ok",
            "alright",
            "all right",
            "got it",
            "sounds good",
            "perfect",
            "great",
            "awesome",
            "done",
            "finished",
            "complete",
            "never mind",
            "nevermind",
            "please",
            "yes",
            "no",
            "stop",
            "hello",
            "hi",
            "hey",
        ]
    )

    # Filler words that suggest continuation
    FILLER_WORDS = frozenset(
        [
            "um",
            "uh",
            "er",
            "ah",
            "like",
            "you know",
            "actually",
            "well",
            "so",
            "basically",
            "literally",
            "just",
        ]
    )

    # Question words that typically start questions
    QUESTION_WORDS = frozenset(
        [
            "what",
            "when",
            "where",
            "who",
            "whom",
            "whose",
            "why",
            "which",
            "how",
            "can",
            "do",
            "does",
            "is",
            "are",
            "will",
        ]
    )

    def __init__(
        self,
        min_word_count: int = 3,
        enable_delimiter_check: bool = True,
        enable_question_detection: bool = True,
    ) -> None:
        """Initialize continuation detector.

        Args:
            min_word_count: Minimum words to consider utterance potentially complete
            enable_delimiter_check: Check for unbalanced quotes/parentheses
            enable_question_detection: Detect questions with terminal question marks
        """
        self._min_word_count = min_word_count
        self._enable_delimiter_check = enable_delimiter_check
        self._enable_question_detection = enable_question_detection

    def is_complete(self, text: str) -> tuple[bool, float]:
        """Determine if utterance is complete (primary interface).

        Args:
            text: Transcribed speech text

        Returns:
            Tuple of (is_complete, confidence) where:
            - is_complete: True if utterance appears complete
            - confidence: Score 0.0-1.0 (higher = more confident)
        """
        result = self.analyze(text)
        return result.is_complete, result.confidence

    def is_complete_sentence(self, text: str) -> tuple[bool, float]:
        """Determine if utterance is complete (alias for backward compatibility).

        Args:
            text: Transcribed speech text

        Returns:
            Tuple of (is_complete, confidence) where:
            - is_complete: True if utterance appears complete
            - confidence: Score 0.0-1.0 (higher = more confident)
        """
        return self.is_complete(text)

    def analyze(self, text: str) -> CompletionResult:
        """Analyze utterance completeness with detailed reasoning.

        Applies heuristics in priority order (highest confidence first):
        1. Empty/whitespace-only (incomplete, confidence=0.0)
        2. Terminal punctuation (complete, confidence=0.80-0.95)
        3. Closing phrases (complete, confidence=0.80-0.95)
        4. Unbalanced delimiters (incomplete, confidence=0.05-0.10)
        5. Trailing incomplete word (incomplete, confidence=0.10-0.30)
        6. Question structure (complete, confidence=0.90-0.95)
        7. Short fragment (incomplete, confidence=0.15-0.30)
        8. Filler words (incomplete, confidence=0.40-0.55)
        9. Default fallback (complete, confidence=0.60-0.65)

        Args:
            text: Transcribed speech text

        Returns:
            CompletionResult with classification, confidence, and reasoning
        """
        # Normalize text
        normalized = text.strip()

        # Check 1: Empty or whitespace-only
        if not normalized:
            return CompletionResult(
                is_complete=False, confidence=0.0, reason="Empty or whitespace-only text"
            )

        # Check 2: Terminal punctuation (highest priority for complete)
        punctuation_result = self._check_terminal_punctuation(normalized)
        if punctuation_result is not None:
            return punctuation_result

        # Check 3: Closing phrases (before short fragment check)
        closing_result = self._check_closing_phrase(normalized)
        if closing_result is not None:
            return closing_result

        # Check 4: Unbalanced delimiters (high priority for incomplete)
        if self._enable_delimiter_check:
            delimiter_result = self._check_unbalanced_delimiters(normalized)
            if delimiter_result is not None:
                return delimiter_result

        # Check 5: Trailing incomplete word (before question structure)
        trailing_result = self._check_trailing_incomplete(normalized)
        if trailing_result is not None:
            return trailing_result

        # Check 6: Question structure
        if self._enable_question_detection:
            question_result = self._check_question_structure(normalized)
            if question_result is not None:
                return question_result

        # Check 7: Short fragment
        short_result = self._check_short_fragment(normalized)
        if short_result is not None:
            return short_result

        # Check 8: Filler words
        filler_result = self._check_filler_words(normalized)
        if filler_result is not None:
            return filler_result

        # Check 9: Incomplete sentence structure
        structure_result = self._check_sentence_structure(normalized)
        if structure_result is not None:
            return structure_result

        # Default: Assume complete with medium confidence
        return CompletionResult(
            is_complete=True, confidence=0.60, reason="No strong indicators; default to complete"
        )

    def _check_unbalanced_delimiters(self, text: str) -> CompletionResult | None:
        """Check for unbalanced quotes, parentheses, or brackets.

        Args:
            text: Normalized text

        Returns:
            CompletionResult if unbalanced delimiters found, None otherwise
        """
        # Check double quotes
        if text.count('"') % 2 != 0:
            return CompletionResult(
                is_complete=False, confidence=0.05, reason='Unbalanced double quotes (")'
            )

        # Check single quotes (apostrophes are tricky, only count isolated ones)
        isolated_quotes = len(re.findall(r"\s'|\'\s", text))
        if isolated_quotes % 2 != 0:
            return CompletionResult(
                is_complete=False, confidence=0.10, reason="Unbalanced single quotes (')"
            )

        # Check parentheses
        if text.count("(") != text.count(")"):
            return CompletionResult(
                is_complete=False, confidence=0.05, reason="Unbalanced parentheses"
            )

        # Check square brackets
        if text.count("[") != text.count("]"):
            return CompletionResult(
                is_complete=False, confidence=0.05, reason="Unbalanced square brackets"
            )

        return None

    def _check_terminal_punctuation(self, text: str) -> CompletionResult | None:
        """Check for sentence-ending punctuation.

        Args:
            text: Normalized text

        Returns:
            CompletionResult if terminal punctuation found, None otherwise
        """
        if text.endswith("."):
            return CompletionResult(
                is_complete=True, confidence=0.95, reason="Ends with period (.)"
            )

        if text.endswith("!"):
            return CompletionResult(
                is_complete=True, confidence=0.95, reason="Ends with exclamation mark (!)"
            )

        if text.endswith("?"):
            return CompletionResult(
                is_complete=True, confidence=0.95, reason="Ends with question mark (?)"
            )

        # Check for comma at end (indicates continuation)
        if text.endswith(","):
            return CompletionResult(
                is_complete=False,
                confidence=0.20,
                reason="Ends with comma (mid-clause continuation)",
            )

        # Check for ellipsis (indicates continuation/uncertainty)
        if text.endswith("..."):
            return CompletionResult(
                is_complete=False,
                confidence=0.30,
                reason="Ends with ellipsis (continuation signal)",
            )

        return None

    def _check_closing_phrase(self, text: str) -> CompletionResult | None:
        """Check for common closing phrases.

        Args:
            text: Normalized text

        Returns:
            CompletionResult if closing phrase found, None otherwise
        """
        # Normalize to lowercase for matching
        lower_text = text.lower()

        for phrase in self.CLOSING_PHRASES:
            # Exact match
            if lower_text == phrase:
                return CompletionResult(
                    is_complete=True, confidence=0.95, reason=f'Closing phrase: "{phrase}"'
                )
            # Ends with multi-word phrase
            if len(phrase.split()) > 1 and lower_text.endswith(f" {phrase}"):
                return CompletionResult(
                    is_complete=True,
                    confidence=0.85,
                    reason=f'Ends with closing phrase: "{phrase}"',
                )

        # Check if single word is a closer
        words = lower_text.split()
        if words and words[-1] in self.CLOSING_PHRASES:
            return CompletionResult(
                is_complete=True, confidence=0.80, reason=f'Ends with closing word: "{words[-1]}"'
            )

        return None

    def _check_question_structure(self, text: str) -> CompletionResult | None:
        """Check for question structure (question word + content).

        Questions without terminal '?' are considered potentially complete
        if they have question structure (question word at start + sufficient words).

        Args:
            text: Normalized text

        Returns:
            CompletionResult if question structure detected, None otherwise
        """
        # Get first word
        words = text.split()
        if not words:
            return None

        first_word = words[0].lower().rstrip(",.!?;:")

        # Check if starts with question word
        if first_word in self.QUESTION_WORDS:
            # Question word found - check if sufficient content follows
            if len(words) >= 3:  # e.g., "What is that"
                return CompletionResult(
                    is_complete=True,
                    confidence=0.90,
                    reason=f'Question structure: starts with "{first_word}"',
                )

        return None

    def _check_short_fragment(self, text: str) -> CompletionResult | None:
        """Check if utterance is very short (likely incomplete or filler).

        Args:
            text: Normalized text

        Returns:
            CompletionResult if short fragment detected, None otherwise
        """
        words = text.split()
        word_count = len(words)

        if word_count < self._min_word_count:
            # Single word fragments get very low confidence
            if word_count == 1:
                return CompletionResult(
                    is_complete=False, confidence=0.15, reason=f"Single word fragment: '{text}'"
                )
            # Two word fragments get low confidence
            else:
                return CompletionResult(
                    is_complete=False,
                    confidence=0.25,
                    reason=f"Short fragment: only {word_count} word(s)",
                )

        return None

    def _check_trailing_incomplete(self, text: str) -> CompletionResult | None:
        """Check for trailing words that indicate incompleteness.

        Args:
            text: Normalized text

        Returns:
            CompletionResult if trailing incomplete word found, None otherwise
        """
        # Remove punctuation and get last word
        text_clean = re.sub(r"[^\w\s]", "", text)
        words = text_clean.split()

        if not words:
            return None

        last_word = words[-1].lower()

        if last_word in self.TRAILING_INCOMPLETE:
            # Determine confidence based on word type
            # Articles and conjunctions are stronger signals
            if last_word in {"a", "an", "the", "and", "but", "or"}:
                confidence = 0.10
            # Prepositions and auxiliaries are also strong signals
            elif last_word in {"to", "from", "in", "on", "is", "are", "will", "would"}:
                confidence = 0.20
            # Other incomplete words have moderate signal
            else:
                confidence = 0.30

            return CompletionResult(
                is_complete=False,
                confidence=confidence,
                reason=f'Trailing incomplete word: "{last_word}"',
            )

        return None

    def _check_filler_words(self, text: str) -> CompletionResult | None:
        """Check for filler words that suggest continuation.

        Args:
            text: Normalized text

        Returns:
            CompletionResult if filler words detected, None otherwise
        """
        lower_text = text.lower()
        words = lower_text.split()

        # Check for filler words at start or end
        if words:
            # Filler at start
            if words[0] in self.FILLER_WORDS:
                return CompletionResult(
                    is_complete=False,
                    confidence=0.40,
                    reason=f'Starts with filler word: "{words[0]}"',
                )

            # Filler at end (but not if it's a single word)
            if len(words) >= 2 and words[-1] in self.FILLER_WORDS:
                return CompletionResult(
                    is_complete=False,
                    confidence=0.45,
                    reason=f'Ends with filler word: "{words[-1]}"',
                )

            # Check for multi-word fillers
            for phrase in self.FILLER_WORDS:
                if len(phrase.split()) > 1:
                    if phrase in lower_text:
                        return CompletionResult(
                            is_complete=False,
                            confidence=0.50,
                            reason=f'Contains filler phrase: "{phrase}"',
                        )

        return None

    def _check_sentence_structure(self, text: str) -> CompletionResult | None:
        """Check for incomplete sentence structure.

        Detects patterns like subject-only sentences without verbs.

        Args:
            text: Normalized text

        Returns:
            CompletionResult if incomplete structure detected, None otherwise
        """
        words = text.split()
        if len(words) < 2:
            return None

        # Check for "I want to <verb>" pattern - if <verb> is missing, incomplete
        lower_text = text.lower()
        if lower_text.startswith("i want to") and len(words) == 3:
            return CompletionResult(
                is_complete=False,
                confidence=0.20,
                reason="Incomplete 'I want to' pattern (missing verb)",
            )

        # Check for "I want to go" - still likely incomplete (where?)
        if lower_text == "i want to go":
            return CompletionResult(
                is_complete=False,
                confidence=0.45,
                reason="'I want to go' typically continues with destination",
            )

        return None
