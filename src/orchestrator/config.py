"""Configuration schema for orchestrator.

Defines Pydantic models for loading and validating orchestrator configuration
from YAML files and environment variables.
"""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class WebSocketConfig(BaseModel):
    """WebSocket transport configuration."""

    enabled: bool = Field(default=True, description="Enable WebSocket transport")
    host: str = Field(default="0.0.0.0", description="Bind host address")  # noqa: S104
    port: int = Field(default=8080, ge=1024, le=65535, description="Bind port")
    max_connections: int = Field(
        default=100, ge=1, description="Maximum concurrent connections"
    )
    frame_queue_size: int = Field(
        default=50, ge=10, description="Audio frame buffer size per session"
    )


class LiveKitConfig(BaseModel):
    """LiveKit/WebRTC transport configuration."""

    enabled: bool = Field(
        default=False, description="Enable LiveKit transport (optional for M2)"
    )
    url: str = Field(
        default="http://localhost:7880",
        description="LiveKit server URL",
    )
    api_key: str = Field(default="devkey", description="LiveKit API key")
    api_secret: str = Field(default="secret", description="LiveKit API secret")
    room_prefix: str = Field(
        default="tts-session",
        description="Prefix for auto-generated room names",
    )


class TransportConfig(BaseModel):
    """Transport layer configuration."""

    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    livekit: LiveKitConfig = Field(default_factory=LiveKitConfig)


class RedisConfig(BaseModel):
    """Redis configuration for service discovery."""

    url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL",
    )
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    worker_key_prefix: str = Field(
        default="worker:",
        description="Key prefix for worker registrations",
    )
    worker_ttl_seconds: int = Field(
        default=30,
        ge=5,
        description="Worker registration TTL (heartbeat interval)",
    )
    connection_pool_size: int = Field(
        default=10,
        ge=1,
        description="Redis connection pool size",
    )


class RoutingConfig(BaseModel):
    """Worker routing configuration."""

    static_worker_addr: str | None = Field(
        default="grpc://localhost:7001",
        description="Static worker address for M2 (single worker routing)",
    )
    # Future (M9+): dynamic routing with load balancing
    prefer_resident_models: bool = Field(
        default=True,
        description="Prefer workers with model already loaded (M9+)",
    )
    load_balance_strategy: str = Field(
        default="queue_depth",
        description="Load balancing strategy: queue_depth, latency, round_robin (M9+)",
    )


class VADConfig(BaseModel):
    """Voice Activity Detection configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable VAD processing (stub for M2, active in M3+)",
    )
    aggressiveness: int = Field(
        default=2,
        ge=0,
        le=3,
        description="VAD aggressiveness (0=least, 3=most aggressive)",
    )
    sample_rate: int = Field(
        default=16000,
        description="VAD input sample rate (webrtcvad requires 16kHz)",
    )
    frame_duration_ms: int = Field(
        default=20,
        description="VAD frame duration in milliseconds",
    )

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        """Validate that sample rate is supported by webrtcvad."""
        valid_rates = [8000, 16000, 32000, 48000]
        if v not in valid_rates:
            raise ValueError(
                f"VAD sample_rate must be one of {valid_rates}, got {v}"
            )
        return v

    @field_validator("frame_duration_ms")
    @classmethod
    def validate_frame_duration(cls, v: int) -> int:
        """Validate that frame duration is supported by webrtcvad."""
        valid_durations = [10, 20, 30]
        if v not in valid_durations:
            raise ValueError(
                f"VAD frame_duration_ms must be one of {valid_durations}, got {v}"
            )
        return v


class OrchestratorConfig(BaseModel):
    """Root orchestrator configuration."""

    transport: TransportConfig = Field(default_factory=TransportConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    vad: VADConfig = Field(default_factory=VADConfig)

    # Operational settings
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    graceful_shutdown_timeout_s: int = Field(
        default=10,
        ge=1,
        description="Graceful shutdown timeout in seconds",
    )

    @classmethod
    def from_yaml(cls, path: Path) -> "OrchestratorConfig":
        """Load configuration from YAML file with environment variable overrides.

        Args:
            path: Path to YAML configuration file

        Returns:
            Loaded configuration

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If YAML is invalid or validation fails
        """
        import os

        import yaml

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Apply environment variable overrides
        if redis_url := os.getenv("REDIS_URL"):
            # Override Redis URL from environment if set
            if "redis" not in data:
                data["redis"] = {}
            data["redis"]["url"] = redis_url

        # Apply LiveKit environment variable overrides
        if livekit_url := os.getenv("LIVEKIT_URL"):
            if "transport" not in data:
                data["transport"] = {}
            if "livekit" not in data["transport"]:
                data["transport"]["livekit"] = {}
            data["transport"]["livekit"]["url"] = livekit_url
            data["transport"]["livekit"]["enabled"] = True

        if livekit_api_key := os.getenv("LIVEKIT_API_KEY"):
            if "transport" not in data:
                data["transport"] = {}
            if "livekit" not in data["transport"]:
                data["transport"]["livekit"] = {}
            data["transport"]["livekit"]["api_key"] = livekit_api_key

        if livekit_api_secret := os.getenv("LIVEKIT_API_SECRET"):
            if "transport" not in data:
                data["transport"] = {}
            if "livekit" not in data["transport"]:
                data["transport"]["livekit"] = {}
            data["transport"]["livekit"]["api_secret"] = livekit_api_secret

        return cls.model_validate(data)

    @classmethod
    def from_yaml_with_defaults(cls, path: Path | None = None) -> "OrchestratorConfig":
        """Load configuration from YAML or use defaults if file doesn't exist.

        Args:
            path: Optional path to YAML configuration file

        Returns:
            Loaded configuration or defaults
        """
        if path is not None and path.exists():
            return cls.from_yaml(path)

        # Return defaults
        return cls()
