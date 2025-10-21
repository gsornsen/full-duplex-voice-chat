"""TTS Worker CLI entry point.

This module is invoked when running `python -m src.tts` or `python -m src.tts.worker`.
It provides command-line argument parsing for the TTS worker server.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from src.tts.config import TTSWorkerConfig
from src.tts.worker import start_worker


def setup_logging(level: str = "INFO", format_type: str = "text") -> None:
    """Configure logging for the worker.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Log format type (json or text)
    """
    if format_type == "json":
        # TODO: Implement JSON logging format in future milestone
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="TTS Worker - gRPC server for text-to-speech synthesis"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worker.yaml"),
        help="Path to worker configuration YAML file (default: configs/worker.yaml)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=os.getenv("ADAPTER_TYPE", "mock"),
        choices=["mock", "sesame", "cosyvoice2", "xtts", "piper"],
        help="TTS adapter to use (default: mock)",
    )
    parser.add_argument(
        "--default-model",
        type=str,
        default=os.getenv("DEFAULT_MODEL_ID", os.getenv("DEFAULT_MODEL", None)),
        help="Override default model ID from config (precedence: CLI > ENV > config)",
    )
    parser.add_argument(
        "--preload",
        type=str,
        nargs="+",
        help="Override preload model IDs from config (space-separated list)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Override gRPC server port from config",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override log level from config",
    )

    return parser.parse_args()


async def main() -> None:
    """Main entry point for TTS worker.

    Loads configuration, sets up logging, and starts the gRPC server.
    Runs until interrupted with Ctrl+C or SIGTERM.

    Configuration precedence for default_model_id:
        1. CLI flag (--default-model)
        2. Environment variable (DEFAULT_MODEL_ID or DEFAULT_MODEL)
        3. YAML config (model_manager.default_model_id)
    """
    args = parse_args()

    # Load configuration
    try:
        config = TTSWorkerConfig.from_yaml(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply configuration overrides with precedence: CLI > ENV > config
    model_source = "config"

    # Default model ID: CLI > ENV > config
    if args.default_model:
        config.model_manager.default_model_id = args.default_model
        model_source = "cli"
    elif env_model := os.getenv("DEFAULT_MODEL_ID") or os.getenv("DEFAULT_MODEL"):
        config.model_manager.default_model_id = env_model
        model_source = "env"

    # Apply other CLI overrides
    if args.preload:
        config.model_manager.preload_model_ids = args.preload
    if args.port:
        config.worker.grpc_port = args.port
    if args.log_level:
        config.logging.level = args.log_level

    # Setup logging
    setup_logging(level=config.logging.level, format_type=config.logging.format)
    logger = logging.getLogger(__name__)

    logger.info(
        "Starting TTS worker",
        extra={
            "worker_name": config.worker.name,
            "grpc_port": config.worker.grpc_port,
            "adapter": args.adapter,
            "default_model": config.model_manager.default_model_id,
            "model_source": model_source,  # Track where config came from
            "preload_models": config.model_manager.preload_model_ids,
        },
    )

    # Create legacy config dict for worker.py
    # TODO: Update worker.py to accept TTSWorkerConfig directly in future milestone
    legacy_config = {
        "port": config.worker.grpc_port,
        "name": config.worker.name,
        "adapter": args.adapter,
        "model_manager": {
            "default_model_id": config.model_manager.default_model_id,
            "default_model_source": model_source,  # Pass source to worker.py
            "preload_model_ids": config.model_manager.preload_model_ids,
            "ttl_ms": config.model_manager.ttl_ms,
            "min_residency_ms": config.model_manager.min_residency_ms,
            "evict_check_interval_ms": config.model_manager.evict_check_interval_ms,
            "resident_cap": config.model_manager.resident_cap,
            "max_parallel_loads": config.model_manager.max_parallel_loads,
            "warmup_enabled": config.model_manager.warmup_enabled,
            "warmup_text": config.model_manager.warmup_text,
        },
        "audio": {
            "output_sample_rate": config.audio.output_sample_rate,
            "frame_duration_ms": config.audio.frame_duration_ms,
            "loudness_target_lufs": config.audio.loudness_target_lufs,
            "normalization_enabled": config.audio.normalization_enabled,
        },
        "redis": {
            "url": config.redis.url,
            "registration_ttl_seconds": config.redis.registration_ttl_seconds,
            "heartbeat_interval_seconds": config.redis.heartbeat_interval_seconds,
        },
        "capabilities": {
            "streaming": config.worker.capabilities.streaming,
            "zero_shot": config.worker.capabilities.zero_shot,
            "lora": config.worker.capabilities.lora,
            "cpu_ok": config.worker.capabilities.cpu_ok,
            "languages": config.worker.capabilities.languages,
            "emotive_zero_prompt": config.worker.capabilities.emotive_zero_prompt,
            "max_concurrent_sessions": config.worker.capabilities.max_concurrent_sessions,
        },
    }

    # Start worker
    try:
        await start_worker(legacy_config)
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.exception("Worker failed with error", extra={"error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
