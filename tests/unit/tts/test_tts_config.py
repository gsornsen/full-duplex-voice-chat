"""Unit tests for TTS worker configuration.

Tests configuration loading, validation, and defaults for TTSWorkerConfig.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError
from tts.config import (
    AudioConfig,
    LoggingConfig,
    MetricsConfig,
    ModelManagerConfig,
    RedisConfig,
    TTSWorkerConfig,
    WorkerCapabilitiesConfig,
    WorkerConfig,
)


def test_worker_capabilities_defaults() -> None:
    """Test worker capabilities configuration defaults."""
    config = WorkerCapabilitiesConfig()
    assert config.streaming is True
    assert config.zero_shot is False
    assert config.lora is False
    assert config.cpu_ok is False
    assert config.languages == ["en"]
    assert config.emotive_zero_prompt is False
    assert config.max_concurrent_sessions == 3


def test_worker_capabilities_language_validation() -> None:
    """Test worker capabilities language validation."""
    # Valid languages
    config = WorkerCapabilitiesConfig(languages=["en", "zh"])
    assert config.languages == ["en", "zh"]

    # Empty list should fail (Pydantic min_length validation)
    with pytest.raises(ValidationError, match="at least 1 item"):
        WorkerCapabilitiesConfig(languages=[])

    # Empty string should fail (custom validator)
    with pytest.raises(ValidationError, match="Invalid language code"):
        WorkerCapabilitiesConfig(languages=[""])


def test_worker_config_defaults() -> None:
    """Test worker configuration defaults."""
    config = WorkerConfig(name="test-worker")
    assert config.name == "test-worker"
    assert config.grpc_host == "0.0.0.0"  # noqa: S104
    assert config.grpc_port == 7001
    assert isinstance(config.capabilities, WorkerCapabilitiesConfig)


def test_worker_config_port_validation() -> None:
    """Test worker gRPC port validation."""
    # Valid port
    config = WorkerConfig(name="test", grpc_port=9000)
    assert config.grpc_port == 9000

    # Invalid port (too low)
    with pytest.raises(ValidationError):
        WorkerConfig(name="test", grpc_port=80)

    # Invalid port (too high)
    with pytest.raises(ValidationError):
        WorkerConfig(name="test", grpc_port=70000)


def test_worker_config_name_validation() -> None:
    """Test worker name validation."""
    # Valid name
    config = WorkerConfig(name="tts-worker-1")
    assert config.name == "tts-worker-1"

    # Empty name should fail
    with pytest.raises(ValidationError):
        WorkerConfig(name="")

    # Name too long should fail
    with pytest.raises(ValidationError):
        WorkerConfig(name="a" * 101)


def test_model_manager_config_defaults() -> None:
    """Test model manager configuration defaults."""
    config = ModelManagerConfig(default_model_id="mock-440hz")
    assert config.default_model_id == "mock-440hz"
    assert config.preload_model_ids == []
    assert config.ttl_ms == 600000
    assert config.min_residency_ms == 120000
    assert config.evict_check_interval_ms == 30000
    assert config.resident_cap == 3
    assert config.max_parallel_loads == 1
    assert config.warmup_enabled is True
    assert config.warmup_text == "This is a warmup test."


def test_model_manager_config_validation() -> None:
    """Test model manager configuration validation."""
    # Empty default_model_id should fail
    with pytest.raises(ValidationError):
        ModelManagerConfig(default_model_id="")

    # Negative TTL should fail
    with pytest.raises(ValidationError):
        ModelManagerConfig(default_model_id="test", ttl_ms=0)

    # Negative min_residency should fail
    with pytest.raises(ValidationError):
        ModelManagerConfig(default_model_id="test", min_residency_ms=0)


def test_audio_config_defaults() -> None:
    """Test audio configuration defaults."""
    config = AudioConfig()
    assert config.output_sample_rate == 48000
    assert config.frame_duration_ms == 20
    assert config.loudness_target_lufs == -16.0
    assert config.normalization_enabled is True


def test_audio_config_sample_rate_validation() -> None:
    """Test audio sample rate validation."""
    # Valid sample rate (M2 only supports 48000)
    config = AudioConfig(output_sample_rate=48000)
    assert config.output_sample_rate == 48000

    # Invalid sample rate should fail
    with pytest.raises(ValidationError, match="M2 only supports 48000 Hz"):
        AudioConfig(output_sample_rate=44100)


def test_audio_config_frame_duration_validation() -> None:
    """Test audio frame duration validation."""
    # Valid frame duration (M2 only supports 20ms)
    config = AudioConfig(frame_duration_ms=20)
    assert config.frame_duration_ms == 20

    # Invalid frame duration should fail
    with pytest.raises(ValidationError, match="M2 only supports 20ms"):
        AudioConfig(frame_duration_ms=30)


def test_redis_config_defaults() -> None:
    """Test Redis configuration defaults."""
    config = RedisConfig()
    assert config.url == "redis://localhost:6379"
    assert config.registration_ttl_seconds == 30
    assert config.heartbeat_interval_seconds == 10


def test_redis_config_url_validation() -> None:
    """Test Redis URL validation."""
    # Valid URLs
    config = RedisConfig(url="redis://localhost:6379")
    assert config.url == "redis://localhost:6379"

    config = RedisConfig(url="rediss://secure.redis.host:6380")
    assert config.url == "rediss://secure.redis.host:6380"

    # Invalid URL (wrong protocol)
    with pytest.raises(ValidationError, match="must start with redis://"):
        RedisConfig(url="http://localhost:6379")

    # Empty URL
    with pytest.raises(ValidationError, match="cannot be empty"):
        RedisConfig(url="")


def test_metrics_config_defaults() -> None:
    """Test metrics configuration defaults."""
    config = MetricsConfig()
    assert config.enabled is True
    assert config.prometheus_port == 9090
    assert config.track_latency is True
    assert config.track_rtf is True
    assert config.track_queue_depth is True


def test_logging_config_defaults() -> None:
    """Test logging configuration defaults."""
    config = LoggingConfig()
    assert config.level == "INFO"
    assert config.format == "json"
    assert config.include_session_id is True


def test_logging_config_level_validation() -> None:
    """Test logging level validation."""
    # Valid levels (case insensitive)
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        config = LoggingConfig(level=level)
        assert config.level == level

    # Case insensitive
    config = LoggingConfig(level="debug")
    assert config.level == "DEBUG"

    # Invalid level
    with pytest.raises(ValidationError, match="Invalid log level"):
        LoggingConfig(level="TRACE")


def test_logging_config_format_validation() -> None:
    """Test logging format validation."""
    # Valid formats
    config = LoggingConfig(format="json")
    assert config.format == "json"

    config = LoggingConfig(format="text")
    assert config.format == "text"

    # Case insensitive
    config = LoggingConfig(format="JSON")
    assert config.format == "json"

    # Invalid format
    with pytest.raises(ValidationError, match="Invalid log format"):
        LoggingConfig(format="xml")


def test_tts_worker_config_composition() -> None:
    """Test that TTSWorkerConfig properly composes all sub-configs."""
    config = TTSWorkerConfig(
        worker=WorkerConfig(name="test-worker"),
        model_manager=ModelManagerConfig(default_model_id="mock-440hz"),
        audio=AudioConfig(),
        redis=RedisConfig(),
    )

    assert config.worker.name == "test-worker"
    assert config.model_manager.default_model_id == "mock-440hz"
    assert config.audio.output_sample_rate == 48000
    assert config.redis.url == "redis://localhost:6379"
    assert isinstance(config.metrics, MetricsConfig)
    assert isinstance(config.logging, LoggingConfig)


def test_tts_worker_config_from_yaml(tmp_path: Path) -> None:
    """Test loading TTS worker configuration from YAML file."""
    # Create temporary config file
    config_file = tmp_path / "test_worker.yaml"
    config_file.write_text(
        """
worker:
  name: "test-worker"
  grpc_port: 9002
  capabilities:
    streaming: true
    languages: ["en", "zh"]

model_manager:
  default_model_id: "test-model"
  ttl_ms: 300000

audio:
  output_sample_rate: 48000
  frame_duration_ms: 20

redis:
  url: "redis://testhost:6379"

logging:
  level: "DEBUG"
"""
    )

    # Load config
    config = TTSWorkerConfig.from_yaml(config_file)

    # Verify loaded values
    assert config.worker.name == "test-worker"
    assert config.worker.grpc_port == 9002
    assert config.worker.capabilities.languages == ["en", "zh"]
    assert config.model_manager.default_model_id == "test-model"
    assert config.model_manager.ttl_ms == 300000
    assert config.redis.url == "redis://testhost:6379"
    assert config.logging.level == "DEBUG"


def test_tts_worker_config_from_yaml_missing_file() -> None:
    """Test loading configuration from non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        TTSWorkerConfig.from_yaml(Path("/nonexistent/worker.yaml"))


def test_tts_worker_config_from_yaml_invalid_values() -> None:
    """Test that invalid YAML values raise validation errors."""
    import tempfile

    # Create temp file with invalid values
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            """
worker:
  name: "test"
  grpc_port: 70000  # Invalid port

model_manager:
  default_model_id: "test"

audio:
  output_sample_rate: 48000
  frame_duration_ms: 20

redis:
  url: "redis://localhost:6379"
"""
        )
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValidationError):
            TTSWorkerConfig.from_yaml(temp_path)
    finally:
        temp_path.unlink()


def test_tts_worker_config_loads_actual_config() -> None:
    """Test loading the actual worker.yaml configuration file."""
    config_path = Path("configs/worker.yaml")

    # Skip if config doesn't exist
    if not config_path.exists():
        pytest.skip("configs/worker.yaml not found")

    # Should load without errors
    config = TTSWorkerConfig.from_yaml(config_path)

    # Verify basic structure
    assert config.worker.name
    assert config.worker.grpc_port >= 1024
    assert config.model_manager.default_model_id
    assert config.audio.output_sample_rate == 48000
    assert config.audio.frame_duration_ms == 20
