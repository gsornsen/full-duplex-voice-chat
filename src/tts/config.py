"""TTS Worker Configuration Models.

Pydantic models for validating worker.yaml configuration.
This ensures all config errors are caught at load time rather than runtime.
"""

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator


class WorkerCapabilitiesConfig(BaseModel):
    """Worker capability flags configuration.

    Attributes:
        streaming: Supports streaming synthesis
        zero_shot: Supports zero-shot voice cloning (M7+)
        lora: Supports LoRA adapters (M8+)
        cpu_ok: Can run on CPU (non-GPU)
        languages: Supported language codes (e.g., ["en", "zh"])
        emotive_zero_prompt: Supports emotion without reference audio (M6+)
        max_concurrent_sessions: Maximum concurrent synthesis sessions
    """

    streaming: bool = Field(
        default=True,
        description="Supports streaming synthesis",
    )
    zero_shot: bool = Field(
        default=False,
        description="Supports zero-shot voice cloning (M7+)",
    )
    lora: bool = Field(
        default=False,
        description="Supports LoRA adapters (M8+)",
    )
    cpu_ok: bool = Field(
        default=False,
        description="Can run on CPU (non-GPU)",
    )
    languages: list[str] = Field(
        default_factory=lambda: ["en"],
        description="Supported language codes",
        min_length=1,
    )
    emotive_zero_prompt: bool = Field(
        default=False,
        description="Supports emotion without reference audio (M6+)",
    )
    max_concurrent_sessions: int = Field(
        default=3,
        description="Maximum concurrent synthesis sessions",
        ge=1,
    )

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, v: list[str]) -> list[str]:
        """Ensure language codes are non-empty and valid."""
        if not v:
            raise ValueError("At least one language must be specified")
        for lang in v:
            if not lang or not lang.strip():
                raise ValueError(f"Invalid language code: '{lang}'")
        return v


class WorkerConfig(BaseModel):
    """Worker identity and network configuration.

    Attributes:
        name: Unique worker identifier for Redis registration
        grpc_host: gRPC server bind address
        grpc_port: gRPC server port
        capabilities: Worker capability flags
    """

    name: str = Field(
        ...,
        description="Unique worker identifier for Redis registration",
        min_length=1,
        max_length=100,
    )
    grpc_host: str = Field(
        default="0.0.0.0",  # noqa: S104
        description="gRPC server bind address",
    )
    grpc_port: int = Field(
        default=7001,
        description="gRPC server port",
        ge=1024,
        le=65535,
    )
    capabilities: WorkerCapabilitiesConfig = Field(
        default_factory=WorkerCapabilitiesConfig,
        description="Worker capability flags",
    )


class ModelManagerConfig(BaseModel):
    """Model lifecycle management configuration (M4+ features).

    Attributes:
        default_model_id: Model to load at startup (required)
        preload_model_ids: Additional models to preload at startup (M4+)
        ttl_ms: Idle model TTL in milliseconds (M4+ feature)
        min_residency_ms: Minimum model residency time (M4+ feature)
        evict_check_interval_ms: Eviction check interval (M4+ feature)
        resident_cap: Maximum resident models (M4+ feature)
        max_parallel_loads: Maximum concurrent model loads (M4+ feature)
        warmup_enabled: Enable model warmup at load time
        warmup_text: Text to use for warmup synthesis
    """

    default_model_id: str = Field(
        ...,
        description="Model to load at startup (required)",
        min_length=1,
    )
    preload_model_ids: list[str] = Field(
        default_factory=list,
        description="Additional models to preload at startup (M4+)",
    )
    ttl_ms: int = Field(
        default=600000,
        description="Idle model TTL in milliseconds (M4+ feature)",
        gt=0,
    )
    min_residency_ms: int = Field(
        default=120000,
        description="Minimum model residency time (M4+ feature)",
        gt=0,
    )
    evict_check_interval_ms: int = Field(
        default=30000,
        description="Eviction check interval (M4+ feature)",
        gt=0,
    )
    resident_cap: int = Field(
        default=3,
        description="Maximum resident models (M4+ feature)",
        ge=1,
    )
    max_parallel_loads: int = Field(
        default=1,
        description="Maximum concurrent model loads (M4+ feature)",
        ge=1,
    )
    warmup_enabled: bool = Field(
        default=True,
        description="Enable model warmup at load time",
    )
    warmup_text: str = Field(
        default="This is a warmup test.",
        description="Text to use for warmup synthesis",
        min_length=1,
    )

    @field_validator("ttl_ms")
    @classmethod
    def validate_ttl(cls, v: int) -> int:
        """Ensure TTL is a positive value."""
        if v <= 0:
            raise ValueError("TTL must be greater than 0")
        return v

    @field_validator("min_residency_ms")
    @classmethod
    def validate_min_residency(cls, v: int) -> int:
        """Ensure min_residency is a positive value."""
        if v <= 0:
            raise ValueError("Minimum residency must be greater than 0")
        return v


class AudioConfig(BaseModel):
    """Audio processing configuration.

    Attributes:
        output_sample_rate: Output sample rate in Hz (fixed at 48000 for M2)
        frame_duration_ms: Frame duration in milliseconds (fixed at 20 for M2)
        loudness_target_lufs: Target LUFS for normalization
        normalization_enabled: Apply loudness normalization
    """

    output_sample_rate: int = Field(
        default=48000,
        description="Output sample rate (fixed in M2)",
        ge=8000,
    )
    frame_duration_ms: int = Field(
        default=20,
        description="Frame duration in milliseconds (fixed in M2)",
        ge=10,
    )
    loudness_target_lufs: float = Field(
        default=-16.0,
        description="Target LUFS for normalization",
        ge=-23.0,
        le=-12.0,
    )
    normalization_enabled: bool = Field(
        default=True,
        description="Apply loudness normalization",
    )

    @field_validator("output_sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        """Ensure sample rate is valid for M2."""
        if v != 48000:
            raise ValueError("M2 only supports 48000 Hz sample rate")
        return v

    @field_validator("frame_duration_ms")
    @classmethod
    def validate_frame_duration(cls, v: int) -> int:
        """Ensure frame duration is valid for M2."""
        if v != 20:
            raise ValueError("M2 only supports 20ms frame duration")
        return v


class RedisConfig(BaseModel):
    """Redis connection configuration.

    Attributes:
        url: Redis connection URL
        registration_ttl_seconds: Worker registration TTL in seconds
        heartbeat_interval_seconds: Heartbeat interval in seconds
    """

    url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL",
    )
    registration_ttl_seconds: int = Field(
        default=30,
        description="Worker registration TTL in seconds",
        ge=5,
        le=3600,
    )
    heartbeat_interval_seconds: int = Field(
        default=10,
        description="Heartbeat interval in seconds",
        ge=1,
        le=300,
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure Redis URL is non-empty."""
        if not v or not v.strip():
            raise ValueError("Redis URL cannot be empty")
        if not v.startswith("redis://") and not v.startswith("rediss://"):
            raise ValueError("Redis URL must start with redis:// or rediss://")
        return v


class MetricsConfig(BaseModel):
    """Metrics and monitoring configuration.

    Attributes:
        enabled: Enable Prometheus metrics
        prometheus_port: Metrics exposition port
        track_latency: Track synthesis latency metrics
        track_rtf: Track real-time factor metrics
        track_queue_depth: Track queue depth metrics
    """

    enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics",
    )
    prometheus_port: int = Field(
        default=9090,
        description="Metrics exposition port",
        ge=1024,
        le=65535,
    )
    track_latency: bool = Field(
        default=True,
        description="Track synthesis latency metrics",
    )
    track_rtf: bool = Field(
        default=True,
        description="Track real-time factor metrics",
    )
    track_queue_depth: bool = Field(
        default=True,
        description="Track queue depth metrics",
    )


class LoggingConfig(BaseModel):
    """Logging configuration.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format (json or text)
        include_session_id: Include session ID in log messages
    """

    level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR)",
    )
    format: str = Field(
        default="json",
        description="Log format (json or text)",
    )
    include_session_id: bool = Field(
        default=True,
        description="Include session ID in log messages",
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Ensure log format is valid."""
        valid_formats = {"json", "text"}
        v_lower = v.lower()
        if v_lower not in valid_formats:
            raise ValueError(f"Invalid log format: {v}. Must be one of {valid_formats}")
        return v_lower


class TTSWorkerConfig(BaseModel):
    """Complete TTS Worker configuration.

    This model validates the entire worker.yaml file structure and ensures
    all configuration errors are caught at load time rather than runtime.

    Example:
        >>> config = TTSWorkerConfig.from_yaml("configs/worker.yaml")
        >>> print(config.model_manager.default_model_id)
        "cosyvoice2-en-base"
        >>> print(config.worker.grpc_port)
        7002
    """

    worker: WorkerConfig
    model_manager: ModelManagerConfig
    audio: AudioConfig
    redis: RedisConfig
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TTSWorkerConfig":
        """Load and validate configuration from YAML file.

        Args:
            path: Path to worker.yaml configuration file

        Returns:
            Validated TTSWorkerConfig instance

        Raises:
            ValidationError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML syntax is invalid

        Example:
            >>> config = TTSWorkerConfig.from_yaml("configs/worker.yaml")
            >>> print(config.worker.name)
            "tts-worker-0"
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls(**data)


# Legacy function for backward compatibility
def load_config(config_path: Path) -> dict[str, Any]:
    """Load TTS worker configuration from YAML file.

    DEPRECATED: Use TTSWorkerConfig.from_yaml() instead for type-safe validation.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Configuration dictionary
    """
    with config_path.open("r") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config
