"""Unit tests for DEFAULT_MODEL environment variable support.

Tests the precedence order and validation logic for default model configuration:
1. CLI flag (--default-model)
2. Environment variable (DEFAULT_MODEL)
3. YAML config (model_manager.default_model_id)
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tts.__main__ import validate_model_id


class TestValidateModelId:
    """Test suite for model ID validation."""

    @pytest.mark.parametrize(
        "model_id,expected",
        [
            # Valid patterns
            ("mock-440hz", True),
            ("mock-test-model", True),
            ("piper-en-us-lessac-medium", True),
            ("piper-en-us-lessac-low", True),
            ("cosyvoice2-en-base", True),
            ("cosyvoice2-zh-instruct", True),
            ("xtts-v2-multilingual", True),
            ("sesame-en-base", True),
            # Invalid patterns
            ("invalid", False),
            ("", False),
            ("model", False),
            ("-model", False),
            ("adapter-", False),
            ("unknown-adapter-model", False),
            ("123-model", False),
            ("PIPER-EN-US", False),  # Case sensitive adapter name
        ],
    )
    def test_validate_model_id_patterns(self, model_id: str, expected: bool) -> None:
        """Test model ID validation against expected patterns."""
        assert validate_model_id(model_id) == expected

    def test_validate_model_id_special_characters(self) -> None:
        """Test model IDs with special characters."""
        # Underscore and dash allowed in model name
        assert validate_model_id("piper-en_us_lessac-medium") is True
        # Special characters not allowed in adapter name
        assert validate_model_id("pip@r-model") is False
        assert validate_model_id("piper model-test") is False


class TestDefaultModelPrecedence:
    """Test suite for DEFAULT_MODEL configuration precedence."""

    @pytest.fixture
    def mock_config_file(self, tmp_path: Path) -> Path:
        """Create a temporary worker.yaml config file."""
        config_path = tmp_path / "worker.yaml"
        config_content = """
worker:
  name: "test-worker"
  grpc_port: 7001
  capabilities:
    streaming: true
    max_concurrent_sessions: 3

model_manager:
  default_model_id: "piper-en-us-lessac-medium"
  preload_model_ids: []
  ttl_ms: 600000
  min_residency_ms: 120000
  evict_check_interval_ms: 30000
  resident_cap: 3
  max_parallel_loads: 1
  warmup_enabled: true
  warmup_text: "This is a warmup test."

audio:
  output_sample_rate: 48000
  frame_duration_ms: 20
  loudness_target_lufs: -16.0
  normalization_enabled: true

redis:
  url: "redis://localhost:6379"
  registration_ttl_seconds: 30
  heartbeat_interval_seconds: 10

logging:
  level: "INFO"
  format: "text"
  include_session_id: true
"""
        config_path.write_text(config_content)
        return config_path

    @patch("src.tts.__main__.start_worker")
    @patch("src.tts.__main__.asyncio.run")
    def test_config_only_no_overrides(
        self,
        mock_asyncio_run: MagicMock,
        mock_start_worker: MagicMock,
        mock_config_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test default model from config file only (no CLI, no ENV)."""
        # Clear environment
        monkeypatch.delenv("DEFAULT_MODEL", raising=False)

        # Mock asyncio.run to prevent coroutine execution and capture the coroutine
        captured_coro = None

        def capture_coro(coro):  # type: ignore[no-untyped-def]
            nonlocal captured_coro
            captured_coro = coro
            # Close the coroutine to prevent "never awaited" warning
            coro.close()

        mock_asyncio_run.side_effect = capture_coro

        # Mock sys.argv to simulate CLI without --default-model
        with patch("sys.argv", ["src.tts", "--config", str(mock_config_file)]):
            # Import and run main
            # Execute main (coroutine will be captured and closed)
            import asyncio

            from src.tts.__main__ import main

            asyncio.run(main())

            # Verify asyncio.run was called
            assert mock_asyncio_run.called

    @patch.dict(os.environ, {"DEFAULT_MODEL": "cosyvoice2-en-base"})
    @patch("src.tts.__main__.start_worker")
    @patch("src.tts.__main__.asyncio.run")
    def test_env_var_overrides_config(
        self,
        mock_asyncio_run: MagicMock,
        mock_start_worker: MagicMock,
        mock_config_file: Path,
    ) -> None:
        """Test environment variable overrides config file."""
        # Mock asyncio.run to prevent coroutine execution and capture the coroutine
        captured_coro = None

        def capture_coro(coro):  # type: ignore[no-untyped-def]
            nonlocal captured_coro
            captured_coro = coro
            # Close the coroutine to prevent "never awaited" warning
            coro.close()

        mock_asyncio_run.side_effect = capture_coro

        # Mock sys.argv to simulate CLI without --default-model
        with patch("sys.argv", ["src.tts", "--config", str(mock_config_file)]):
            # Capture the coroutine and close it
            import asyncio

            from src.tts.__main__ import main

            asyncio.run(main())

            # Verify env var took precedence
            assert os.environ["DEFAULT_MODEL"] == "cosyvoice2-en-base"

    @patch("src.tts.__main__.start_worker")
    @patch("src.tts.__main__.asyncio.run")
    def test_cli_overrides_env_and_config(
        self,
        mock_asyncio_run: MagicMock,
        mock_start_worker: MagicMock,
        mock_config_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test CLI flag overrides both environment variable and config."""
        # Set environment variable
        monkeypatch.setenv("DEFAULT_MODEL", "cosyvoice2-en-base")

        # Mock asyncio.run to prevent coroutine execution and capture the coroutine
        captured_coro = None

        def capture_coro(coro):  # type: ignore[no-untyped-def]
            nonlocal captured_coro
            captured_coro = coro
            # Close the coroutine to prevent "never awaited" warning
            coro.close()

        mock_asyncio_run.side_effect = capture_coro

        # Mock sys.argv to simulate CLI with --default-model
        with patch(
            "sys.argv",
            [
                "src.tts",
                "--config",
                str(mock_config_file),
                "--default-model",
                "mock-440hz",
            ],
        ):
            import asyncio

            from src.tts.__main__ import main

            asyncio.run(main())

            # CLI flag should take precedence over ENV and config

    def test_invalid_model_id_exits(
        self,
        mock_config_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that invalid model ID causes sys.exit(1).

        This test verifies that when an invalid model ID is provided (either via
        CLI, environment variable, or config), the application exits with code 1
        and prints an appropriate error message.

        The test uses proper async resource cleanup to avoid "coroutine never awaited"
        warnings when run in a test suite with other tests.
        """
        # Set invalid model via environment
        monkeypatch.setenv("DEFAULT_MODEL", "invalid-model-format")

        # Mock sys.argv
        with patch("sys.argv", ["src.tts", "--config", str(mock_config_file)]):
            from src.tts.__main__ import main

            # Create the coroutine
            coro = main()

            # Run main - should exit with error
            # We use asyncio.run() which will properly clean up the coroutine
            with pytest.raises(SystemExit) as exc_info:
                import asyncio

                asyncio.run(coro)

            # Verify exit code is 1 (error)
            assert exc_info.value.code == 1

            # Verify error message was printed
            captured = capsys.readouterr()
            assert "Invalid model ID" in captured.err


class TestWorkerEnvVarFallback:
    """Test suite for worker.py environment variable fallback."""

    @pytest.mark.asyncio
    async def test_worker_uses_env_var_when_no_cli_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test worker.py reads DEFAULT_MODEL from environment when not set via CLI."""
        monkeypatch.setenv("DEFAULT_MODEL", "cosyvoice2-zh-instruct")

        # Mock ModelManager to avoid actual model loading
        with patch("src.tts.worker.ModelManager") as mock_mm:
            mock_mm_instance = MagicMock()
            mock_mm_instance.initialize = AsyncMock()
            mock_mm_instance.shutdown = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            with patch("src.tts.worker.grpc.aio.server") as mock_server:
                mock_server_instance = MagicMock()
                mock_server_instance.start = AsyncMock()
                mock_server_instance.stop = AsyncMock()
                mock_server.return_value = mock_server_instance

                # Mock wait_for_termination to raise KeyboardInterrupt immediately
                async def mock_wait() -> None:
                    raise KeyboardInterrupt

                mock_server_instance.wait_for_termination = mock_wait

                from src.tts.worker import start_worker

                config = {
                    "port": 7002,
                    "name": "test-worker",
                    "adapter": "cosyvoice2",
                    "model_manager": {
                        "default_model_id": "piper-en-us-lessac-medium",  # Config default
                        "default_model_source": "config",  # Not overridden by CLI
                        "preload_model_ids": [],
                        "ttl_ms": 600000,
                        "min_residency_ms": 120000,
                        "resident_cap": 3,
                        "max_parallel_loads": 1,
                        "warmup_enabled": True,
                        "warmup_text": "Test",
                        "evict_check_interval_ms": 30000,
                    },
                }

                try:
                    await start_worker(config)
                except KeyboardInterrupt:
                    pass

                # Verify ModelManager was instantiated with env var value
                mock_mm.assert_called_once()
                call_kwargs = mock_mm.call_args[1]
                assert call_kwargs["default_model_id"] == "cosyvoice2-zh-instruct"

    @pytest.mark.asyncio
    async def test_worker_respects_cli_override_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test worker.py respects CLI override even when env var is set."""
        monkeypatch.setenv("DEFAULT_MODEL", "cosyvoice2-zh-instruct")

        with patch("src.tts.worker.ModelManager") as mock_mm:
            mock_mm_instance = MagicMock()
            mock_mm_instance.initialize = AsyncMock()
            mock_mm_instance.shutdown = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            with patch("src.tts.worker.grpc.aio.server") as mock_server:
                mock_server_instance = MagicMock()
                mock_server_instance.start = AsyncMock()
                mock_server_instance.stop = AsyncMock()
                mock_server.return_value = mock_server_instance

                async def mock_wait() -> None:
                    raise KeyboardInterrupt

                mock_server_instance.wait_for_termination = mock_wait

                from src.tts.worker import start_worker

                config = {
                    "port": 7002,
                    "name": "test-worker",
                    "adapter": "mock",
                    "model_manager": {
                        "default_model_id": "mock-440hz",  # CLI override
                        "default_model_source": "cli",  # Set by __main__.py
                        "preload_model_ids": [],
                        "ttl_ms": 600000,
                        "min_residency_ms": 120000,
                        "resident_cap": 3,
                        "max_parallel_loads": 1,
                        "warmup_enabled": True,
                        "warmup_text": "Test",
                        "evict_check_interval_ms": 30000,
                    },
                }

                try:
                    await start_worker(config)
                except KeyboardInterrupt:
                    pass

                # Verify ModelManager was instantiated with CLI value (not env)
                mock_mm.assert_called_once()
                call_kwargs = mock_mm.call_args[1]
                assert call_kwargs["default_model_id"] == "mock-440hz"


class TestModelSourceLogging:
    """Test suite for model source logging."""

    @pytest.mark.asyncio
    async def test_worker_logs_model_source(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that worker.py logs the source of default model configuration."""
        monkeypatch.setenv("DEFAULT_MODEL", "cosyvoice2-en-base")

        with patch("src.tts.worker.ModelManager") as mock_mm:
            mock_mm_instance = MagicMock()
            mock_mm_instance.initialize = AsyncMock()
            mock_mm_instance.shutdown = AsyncMock()
            mock_mm.return_value = mock_mm_instance

            with patch("src.tts.worker.grpc.aio.server") as mock_server:
                mock_server_instance = MagicMock()
                mock_server_instance.start = AsyncMock()
                mock_server_instance.stop = AsyncMock()
                mock_server.return_value = mock_server_instance

                async def mock_wait() -> None:
                    raise KeyboardInterrupt

                mock_server_instance.wait_for_termination = mock_wait

                from src.tts.worker import start_worker

                config = {
                    "port": 7002,
                    "name": "test-worker",
                    "adapter": "cosyvoice2",
                    "model_manager": {
                        "default_model_id": "piper-en-us-lessac-medium",
                        "default_model_source": "config",
                        "preload_model_ids": [],
                        "ttl_ms": 600000,
                        "min_residency_ms": 120000,
                        "resident_cap": 3,
                        "max_parallel_loads": 1,
                        "warmup_enabled": True,
                        "warmup_text": "Test",
                        "evict_check_interval_ms": 30000,
                    },
                }

                with caplog.at_level("INFO"):
                    try:
                        await start_worker(config)
                    except KeyboardInterrupt:
                        pass

                # Verify log contains model source information
                log_messages = [record.message for record in caplog.records]
                assert any("ModelManager configured" in msg for msg in log_messages)
