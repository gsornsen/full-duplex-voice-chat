"""Unit tests for ASR base protocol and data types.

Tests the abstract base class interface and data types without
requiring a concrete adapter implementation.
"""

import pytest

from asr.asr_base import (
    ASRAdapterBase,
    ASRCapabilities,
    ASRError,
    ASRLanguage,
    InvalidAudioError,
    ModelNotLoadedError,
    TranscriptionError,
    TranscriptionResult,
)


class TestTranscriptionResult:
    """Test TranscriptionResult dataclass."""

    def test_basic_result(self) -> None:
        """Test creating basic transcription result."""
        result = TranscriptionResult(
            text="Hello world",
            confidence=0.95,
        )
        assert result.text == "Hello world"
        assert result.confidence == 0.95
        assert result.language is None
        assert result.duration_ms == 0
        assert result.word_timestamps is None
        assert result.is_partial is False

    def test_full_result(self) -> None:
        """Test creating result with all fields."""
        timestamps = [("Hello", 0, 500), ("world", 500, 1000)]
        result = TranscriptionResult(
            text="Hello world",
            confidence=0.92,
            language="en",
            duration_ms=1000,
            word_timestamps=timestamps,
            is_partial=True,
        )
        assert result.text == "Hello world"
        assert result.confidence == 0.92
        assert result.language == "en"
        assert result.duration_ms == 1000
        assert result.word_timestamps == timestamps
        assert result.is_partial is True

    def test_confidence_validation_low(self) -> None:
        """Test confidence validation rejects values below 0.0."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            TranscriptionResult(text="test", confidence=-0.1)

    def test_confidence_validation_high(self) -> None:
        """Test confidence validation rejects values above 1.0."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            TranscriptionResult(text="test", confidence=1.1)

    def test_confidence_validation_boundary_low(self) -> None:
        """Test confidence accepts 0.0."""
        result = TranscriptionResult(text="test", confidence=0.0)
        assert result.confidence == 0.0

    def test_confidence_validation_boundary_high(self) -> None:
        """Test confidence accepts 1.0."""
        result = TranscriptionResult(text="test", confidence=1.0)
        assert result.confidence == 1.0

    def test_duration_validation(self) -> None:
        """Test duration validation rejects negative values."""
        with pytest.raises(ValueError, match="Duration must be non-negative"):
            TranscriptionResult(text="test", confidence=0.9, duration_ms=-100)

    def test_duration_zero(self) -> None:
        """Test duration accepts zero."""
        result = TranscriptionResult(text="test", confidence=0.9, duration_ms=0)
        assert result.duration_ms == 0

    def test_immutability(self) -> None:
        """Test that TranscriptionResult is immutable (frozen dataclass)."""
        result = TranscriptionResult(text="test", confidence=0.9)
        with pytest.raises(AttributeError):
            result.text = "modified"  # type: ignore[misc]


class TestASRLanguage:
    """Test ASRLanguage enum."""

    def test_language_codes(self) -> None:
        """Test that language codes are accessible."""
        assert ASRLanguage.EN.value == "en"
        assert ASRLanguage.ES.value == "es"
        assert ASRLanguage.FR.value == "fr"
        assert ASRLanguage.DE.value == "de"
        assert ASRLanguage.ZH.value == "zh"
        assert ASRLanguage.JA.value == "ja"
        assert ASRLanguage.KO.value == "ko"
        assert ASRLanguage.AUTO.value == "auto"

    def test_language_string_values(self) -> None:
        """Test that language enums can be used as strings."""
        # Since ASRLanguage inherits from str, the value IS the string
        assert ASRLanguage.EN == "en"  # type: ignore[comparison-overlap]
        assert ASRLanguage.AUTO == "auto"


class TestASRExceptions:
    """Test ASR exception hierarchy."""

    def test_base_exception(self) -> None:
        """Test that ASRError can be raised."""
        with pytest.raises(ASRError):
            raise ASRError("Base error")

    def test_model_not_loaded_error(self) -> None:
        """Test ModelNotLoadedError is subclass of ASRError."""
        assert issubclass(ModelNotLoadedError, ASRError)
        with pytest.raises(ModelNotLoadedError):
            raise ModelNotLoadedError("Model not loaded")

    def test_invalid_audio_error(self) -> None:
        """Test InvalidAudioError is subclass of ASRError."""
        assert issubclass(InvalidAudioError, ASRError)
        with pytest.raises(InvalidAudioError):
            raise InvalidAudioError("Invalid audio")

    def test_transcription_error(self) -> None:
        """Test TranscriptionError is subclass of ASRError."""
        assert issubclass(TranscriptionError, ASRError)
        with pytest.raises(TranscriptionError):
            raise TranscriptionError("Transcription failed")

    def test_catch_hierarchy(self) -> None:
        """Test that catching ASRError catches all subtypes."""
        try:
            raise ModelNotLoadedError("test")
        except ASRError:
            pass  # Should catch it

        try:
            raise InvalidAudioError("test")
        except ASRError:
            pass  # Should catch it

        try:
            raise TranscriptionError("test")
        except ASRError:
            pass  # Should catch it


class TestASRAdapterBase:
    """Test ASRAdapterBase abstract class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that ASRAdapterBase cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ASRAdapterBase()  # type: ignore[abstract]

    def test_subclass_must_implement_abstract_methods(self) -> None:
        """Test that subclass must implement all abstract methods."""

        # Missing all abstract methods
        class IncompleteAdapter(ASRAdapterBase):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAdapter()  # type: ignore[abstract]

    def test_valid_concrete_implementation(self) -> None:
        """Test that concrete implementation with all methods works."""

        class MockAdapter(ASRAdapterBase):
            async def initialize(self) -> None:
                pass

            async def transcribe(
                self, audio: bytes, sample_rate: int, language: str | None = None
            ) -> TranscriptionResult:
                return TranscriptionResult(text="mock", confidence=1.0)

            async def shutdown(self) -> None:
                pass

            def get_supported_sample_rates(self) -> list[int]:
                return [16000, 48000]

            def is_initialized(self) -> bool:
                return True

            def get_model_info(self) -> dict[str, str | int | float]:
                return {"name": "mock", "version": "1.0"}

        # Should not raise
        adapter = MockAdapter()
        assert adapter is not None


class TestASRCapabilities:
    """Test ASRCapabilities descriptor."""

    def test_default_capabilities(self) -> None:
        """Test default capabilities."""
        caps = ASRCapabilities()
        assert caps.supports_streaming is False
        assert caps.supports_timestamps is False
        assert caps.supports_language_detection is False
        assert caps.supported_languages == ["en"]
        assert caps.max_audio_duration_s == 30.0
        assert caps.requires_gpu is False
        assert caps.model_size_mb == 0

    def test_custom_capabilities(self) -> None:
        """Test custom capabilities."""
        caps = ASRCapabilities(
            supports_streaming=True,
            supports_timestamps=True,
            supports_language_detection=True,
            supported_languages=["en", "es", "fr"],
            max_audio_duration_s=60.0,
            requires_gpu=True,
            model_size_mb=500,
        )
        assert caps.supports_streaming is True
        assert caps.supports_timestamps is True
        assert caps.supports_language_detection is True
        assert caps.supported_languages == ["en", "es", "fr"]
        assert caps.max_audio_duration_s == 60.0
        assert caps.requires_gpu is True
        assert caps.model_size_mb == 500

    def test_to_dict(self) -> None:
        """Test capabilities serialization to dict."""
        caps = ASRCapabilities(
            supports_streaming=True,
            supported_languages=["en", "es"],
            model_size_mb=244,
        )
        caps_dict = caps.to_dict()

        assert isinstance(caps_dict, dict)
        assert caps_dict["supports_streaming"] is True
        assert caps_dict["supports_timestamps"] is False
        assert caps_dict["supported_languages"] == ["en", "es"]
        assert caps_dict["model_size_mb"] == 244

    def test_to_dict_all_fields(self) -> None:
        """Test that to_dict includes all fields."""
        caps = ASRCapabilities()
        caps_dict = caps.to_dict()

        expected_keys = {
            "supports_streaming",
            "supports_timestamps",
            "supports_language_detection",
            "supported_languages",
            "max_audio_duration_s",
            "requires_gpu",
            "model_size_mb",
        }
        assert set(caps_dict.keys()) == expected_keys
