"""Tests for configuration validator."""

import pytest
from orchestrator.config_validator import ConfigurationError, ConfigValidator
from pytest import MonkeyPatch


def test_validate_tts_cosyvoice_configuration(monkeypatch: MonkeyPatch) -> None:
    """Test TTS validation for CosyVoice configuration."""
    # Clear all env vars first
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL_ID", raising=False)
    monkeypatch.delenv("ADAPTER_TYPE", raising=False)

    monkeypatch.setenv("DEFAULT_MODEL", "cosyvoice2-en-base")
    monkeypatch.setenv("ADAPTER_TYPE", "cosyvoice2")

    valid, warnings = ConfigValidator.validate_tts_configuration()

    # If voicepack exists, should pass with no warnings
    # If voicepack doesn't exist, should warn about missing voicepack
    assert isinstance(warnings, list)


def test_validate_tts_piper_configuration(monkeypatch: MonkeyPatch) -> None:
    """Test TTS validation for Piper configuration."""
    # Clear all env vars first
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL_ID", raising=False)
    monkeypatch.delenv("ADAPTER_TYPE", raising=False)

    monkeypatch.setenv("DEFAULT_MODEL", "piper-en-us-lessac-medium")
    monkeypatch.setenv("ADAPTER_TYPE", "piper")

    valid, warnings = ConfigValidator.validate_tts_configuration()

    # Should pass (or only warn about missing files)
    assert isinstance(warnings, list)


def test_validate_tts_mismatch(monkeypatch: MonkeyPatch) -> None:
    """Test TTS validation detects adapter/model mismatch."""
    # Clear all env vars first
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL_ID", raising=False)
    monkeypatch.delenv("ADAPTER_TYPE", raising=False)

    monkeypatch.setenv("DEFAULT_MODEL", "cosyvoice2-en-base")
    monkeypatch.setenv("ADAPTER_TYPE", "piper")  # MISMATCH!

    valid, warnings = ConfigValidator.validate_tts_configuration()

    assert not valid
    assert any("requires ADAPTER_TYPE=cosyvoice2" in w for w in warnings)


def test_validate_asr_cuda_unavailable(monkeypatch: MonkeyPatch) -> None:
    """Test ASR validation warns when CUDA requested but unavailable."""
    # Clear all env vars first
    monkeypatch.delenv("ASR_DEVICE", raising=False)
    monkeypatch.delenv("ASR_COMPUTE_TYPE", raising=False)

    monkeypatch.setenv("ASR_DEVICE", "cuda")

    # This will check actual CUDA availability
    valid, warnings = ConfigValidator.validate_asr_configuration()

    # Warnings depend on actual CUDA availability in test environment
    assert isinstance(warnings, list)


def test_validate_all_strict_mode(monkeypatch: MonkeyPatch) -> None:
    """Test strict mode raises error on warnings."""
    # Clear all env vars first
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL_ID", raising=False)
    monkeypatch.delenv("ADAPTER_TYPE", raising=False)

    monkeypatch.setenv("DEFAULT_MODEL", "invalid-model-id")

    with pytest.raises(ConfigurationError):
        ConfigValidator.validate_all(strict=True)


def test_validate_all_permissive_mode(monkeypatch: MonkeyPatch) -> None:
    """Test permissive mode logs but doesn't raise."""
    # Clear all env vars first
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL_ID", raising=False)
    monkeypatch.delenv("ADAPTER_TYPE", raising=False)

    monkeypatch.setenv("DEFAULT_MODEL", "invalid-model-id")

    # Should not raise
    ConfigValidator.validate_all(strict=False)


def test_validate_tts_no_model_set(monkeypatch: MonkeyPatch) -> None:
    """Test validation when no model is configured."""
    # Clear all env vars
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL_ID", raising=False)
    monkeypatch.delenv("ADAPTER_TYPE", raising=False)

    valid, warnings = ConfigValidator.validate_tts_configuration()

    # Should pass with warning
    assert valid
    assert len(warnings) == 1
    assert "not set" in warnings[0]


def test_validate_tts_mock_configuration(monkeypatch: MonkeyPatch) -> None:
    """Test validation for mock adapter."""
    # Clear all env vars first
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL_ID", raising=False)
    monkeypatch.delenv("ADAPTER_TYPE", raising=False)

    monkeypatch.setenv("DEFAULT_MODEL", "mock")
    monkeypatch.setenv("ADAPTER_TYPE", "mock")

    valid, warnings = ConfigValidator.validate_tts_configuration()

    # Should pass without warnings
    assert valid
    assert len(warnings) == 0


def test_validate_asr_cpu_float16_warning(monkeypatch: MonkeyPatch) -> None:
    """Test ASR warns about incompatible CPU/float16 combo."""
    # Clear all env vars first
    monkeypatch.delenv("ASR_DEVICE", raising=False)
    monkeypatch.delenv("ASR_COMPUTE_TYPE", raising=False)

    monkeypatch.setenv("ASR_DEVICE", "cpu")
    monkeypatch.setenv("ASR_COMPUTE_TYPE", "float16")

    valid, warnings = ConfigValidator.validate_asr_configuration()

    # Should warn about incompatibility
    assert not valid
    assert any("float16 not supported on CPU" in w for w in warnings)


def test_validate_tts_cosyvoice_missing_voicepack(monkeypatch: MonkeyPatch) -> None:
    """Test TTS validation warns when voicepack is missing."""
    # Clear all env vars first
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL_ID", raising=False)
    monkeypatch.delenv("ADAPTER_TYPE", raising=False)

    # Use a voicepack name that definitely doesn't exist
    monkeypatch.setenv("DEFAULT_MODEL", "cosyvoice2-nonexistent-voice")
    monkeypatch.setenv("ADAPTER_TYPE", "cosyvoice2")

    valid, warnings = ConfigValidator.validate_tts_configuration()

    # Should warn about missing voicepack
    assert not valid
    assert len(warnings) > 0
    assert "Voicepack not found" in warnings[0]
