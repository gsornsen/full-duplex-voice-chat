"""Unit tests for orchestrator configuration.

Tests configuration loading, validation, and defaults.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from orchestrator.config import (
    LiveKitConfig,
    OrchestratorConfig,
    RedisConfig,
    RoutingConfig,
    TransportConfig,
    VADConfig,
    WebSocketConfig,
)


def test_websocket_config_defaults() -> None:
    """Test WebSocket configuration defaults."""
    config = WebSocketConfig()
    assert config.enabled is True
    assert config.host == "0.0.0.0"  # noqa: S104
    assert config.port == 8080
    assert config.max_connections == 100
    assert config.frame_queue_size == 50


def test_websocket_config_validation() -> None:
    """Test WebSocket configuration validation."""
    # Valid port
    config = WebSocketConfig(port=9000)
    assert config.port == 9000

    # Invalid port (too low)
    with pytest.raises(ValueError):
        WebSocketConfig(port=80)

    # Invalid port (too high)
    with pytest.raises(ValueError):
        WebSocketConfig(port=70000)


def test_livekit_config_defaults() -> None:
    """Test LiveKit configuration defaults."""
    config = LiveKitConfig()
    assert config.enabled is False
    assert config.url == "http://localhost:7880"
    assert config.api_key == "devkey"
    assert config.api_secret == "secret"
    assert config.room_prefix == "tts-session"


def test_vad_config_defaults() -> None:
    """Test VAD configuration defaults."""
    config = VADConfig()
    assert config.enabled is True
    assert config.aggressiveness == 2
    assert config.sample_rate == 16000
    assert config.frame_duration_ms == 20


def test_vad_config_sample_rate_validation() -> None:
    """Test VAD sample rate validation."""
    # Valid rates
    for rate in [8000, 16000, 32000, 48000]:
        config = VADConfig(sample_rate=rate)
        assert config.sample_rate == rate

    # Invalid rate
    with pytest.raises(ValueError, match="VAD sample_rate must be one of"):
        VADConfig(sample_rate=44100)


def test_vad_config_frame_duration_validation() -> None:
    """Test VAD frame duration validation."""
    # Valid durations
    for duration in [10, 20, 30]:
        config = VADConfig(frame_duration_ms=duration)
        assert config.frame_duration_ms == duration

    # Invalid duration
    with pytest.raises(ValueError, match="VAD frame_duration_ms must be one of"):
        VADConfig(frame_duration_ms=25)


def test_redis_config_defaults() -> None:
    """Test Redis configuration defaults."""
    config = RedisConfig()
    assert config.url == "redis://localhost:6379"
    assert config.db == 0
    assert config.worker_key_prefix == "worker:"
    assert config.worker_ttl_seconds == 30
    assert config.connection_pool_size == 10


def test_routing_config_defaults() -> None:
    """Test routing configuration defaults."""
    config = RoutingConfig()
    assert config.static_worker_addr == "grpc://localhost:7001"
    assert config.prefer_resident_models is True
    assert config.load_balance_strategy == "queue_depth"


def test_orchestrator_config_defaults() -> None:
    """Test root orchestrator configuration defaults."""
    config = OrchestratorConfig()

    # Check nested configs
    assert isinstance(config.transport, TransportConfig)
    assert isinstance(config.redis, RedisConfig)
    assert isinstance(config.routing, RoutingConfig)
    assert isinstance(config.vad, VADConfig)

    # Check operational settings
    assert config.log_level == "INFO"
    assert config.graceful_shutdown_timeout_s == 10


@patch("os.getenv")
def test_orchestrator_config_from_yaml(mock_getenv: Mock, tmp_path: Path) -> None:
    """Test loading configuration from YAML file."""
    # Mock os.getenv to return None (no environment overrides)
    mock_getenv.return_value = None

    # Create temporary config file
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """transport:
  websocket:
    enabled: true
    port: 9000
  livekit:
    enabled: false

redis:
  url: "redis://testhost:6379"

routing:
  static_worker_addr: "grpc://worker:7001"

vad:
  aggressiveness: 3

log_level: "DEBUG"
"""
    )

    # Load config
    config = OrchestratorConfig.from_yaml(config_file)

    # Verify loaded values
    assert config.transport.websocket.port == 9000
    assert config.transport.livekit.enabled is False
    assert config.redis.url == "redis://testhost:6379"
    assert config.routing.static_worker_addr == "grpc://worker:7001"
    assert config.vad.aggressiveness == 3
    assert config.log_level == "DEBUG"


@patch("os.getenv")
def test_orchestrator_config_from_yaml_with_env_overrides(mock_getenv: Mock, tmp_path: Path) -> None: # noqa: E501
    """Test loading configuration from YAML file with environment variable overrides."""
    # Mock os.getenv to return override URL for REDIS_URL
    def getenv_side_effect(key: str) -> str | None:
        if key == "REDIS_URL":
            return "redis://patched_host:6379"
        return None

    mock_getenv.side_effect = getenv_side_effect

    # Create temporary config file
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """transport:
  websocket:
    enabled: true
    port: 9000
  livekit:
    enabled: false

redis:
  url: "redis://testhost:6379"

routing:
  static_worker_addr: "grpc://worker:7001"

vad:
  aggressiveness: 3

log_level: "DEBUG"
"""
    )

    # Load config
    config = OrchestratorConfig.from_yaml(config_file)

    # Verify loaded values with environment variable overrides
    assert config.transport.websocket.port == 9000
    assert config.transport.livekit.enabled is False
    assert config.redis.url == "redis://patched_host:6379"  # Should be overridden
    assert config.routing.static_worker_addr == "grpc://worker:7001"
    assert config.vad.aggressiveness == 3
    assert config.log_level == "DEBUG"


def test_orchestrator_config_from_yaml_missing_file() -> None:
    """Test loading configuration from non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        OrchestratorConfig.from_yaml(Path("/nonexistent/config.yaml"))


def test_orchestrator_config_from_yaml_with_defaults_missing() -> None:
    """Test loading config with defaults when file doesn't exist."""
    config = OrchestratorConfig.from_yaml_with_defaults(
        Path("/nonexistent/config.yaml")
    )

    # Should return defaults
    assert config.transport.websocket.port == 8080
    assert config.log_level == "INFO"


def test_orchestrator_config_from_yaml_with_defaults_exists(tmp_path: Path) -> None:
    """Test loading config with defaults when file exists."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """transport:
  websocket:
    port: 9000
"""
    )

    config = OrchestratorConfig.from_yaml_with_defaults(config_file)

    # Should load from file
    assert config.transport.websocket.port == 9000


def test_transport_config_composition() -> None:
    """Test that TransportConfig properly composes WebSocket and LiveKit configs."""
    config = TransportConfig(
        websocket=WebSocketConfig(port=9000, enabled=True),
        livekit=LiveKitConfig(enabled=True, url="http://custom:7880"),
    )

    assert config.websocket.port == 9000
    assert config.websocket.enabled is True
    assert config.livekit.enabled is True
    assert config.livekit.url == "http://custom:7880"
