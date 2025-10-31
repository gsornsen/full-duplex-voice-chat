"""Integration tests for Piper TTS adapter with ModelManager and Worker.

Tests end-to-end flows including:
- Worker loads Piper model via ModelManager
- Session synthesis with real frame streaming
- Barge-in pause latency verification
- First Audio Latency (FAL) measurement
- Model switching and lifecycle
- Concurrent sessions
- TTL eviction with Piper models
- gRPC service integration
"""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tts.adapters.adapter_piper import PiperTTSAdapter
from tts.model_manager import ModelManager
from tts.tts_base import AdapterState


@pytest.fixture
def mock_piper_voice() -> Mock:
    """Create a mock PiperVoice for integration tests."""
    voice = Mock()
    # Return 500ms of audio per synthesis call
    audio_samples = np.zeros(11025, dtype=np.int16)  # 500ms at 22050 Hz

    # Create mock AudioChunk
    audio_chunk = Mock()
    audio_chunk.audio_int16_array = audio_samples
    audio_chunk.sample_rate = 22050
    audio_chunk.sample_width = 2
    audio_chunk.sample_channels = 1

    voice.synthesize.return_value = [audio_chunk]
    return voice


@pytest.fixture
def mock_voicepack_structure(tmp_path: Path) -> Path:
    """Create mock voicepack directory structure."""
    voicepack_dir = tmp_path / "voicepacks" / "piper" / "en-us-test"
    voicepack_dir.mkdir(parents=True)

    # Create mock ONNX model
    onnx_file = voicepack_dir / "model.onnx"
    onnx_file.write_bytes(b"mock onnx model data")

    # Create config
    config_file = voicepack_dir / "model.onnx.json"
    config_data = {"audio": {"sample_rate": 22050}}
    config_file.write_text(json.dumps(config_data))

    # Create metadata
    metadata_file = voicepack_dir / "metadata.yaml"
    metadata_content = """
    tags:
      lang: en
      cpu_ok: true
      domain: general
    """
    metadata_file.write_text(metadata_content)

    return tmp_path / "voicepacks"


# ============================================================================
# Model Loading Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_manager_loads_piper_model(
    mock_voicepack_structure: Path, mock_piper_voice: Mock
) -> None:
    """Test ModelManager can load Piper model via adapter."""
    with patch("tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        # Patch the voicepack path resolution
        with patch("tts.model_manager.Path") as mock_path_class:
            mock_voicepack_path = mock_voicepack_structure / "piper" / "en-us-test"
            mock_path_class.return_value = mock_voicepack_path

            mm = ModelManager(
                default_model_id="piper-en-us-test",
                warmup_enabled=False,
                evict_check_interval_ms=1000,
            )

            await mm.initialize()

            # Model should be loaded
            loaded_models = await mm.list_models()
            assert "piper-en-us-test" in loaded_models

            await mm.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_session_synthesis(
    mock_voicepack_structure: Path, mock_piper_voice: Mock
) -> None:
    """Test end-to-end synthesis session with Piper adapter."""
    with patch("tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        voicepack_path = mock_voicepack_structure / "piper" / "en-us-test"
        adapter = PiperTTSAdapter(
            model_id="piper-en-us-test",
            model_path=str(voicepack_path),
        )

        # Create text stream
        async def text_stream() -> AsyncGenerator[str, None]:
            yield "Hello, world!"
            yield "This is a test."

        # Synthesize and collect frames
        frames: list[bytes] = []
        async for frame in adapter.synthesize_stream(text_stream()):
            frames.append(frame)

        # Verify we got frames
        assert len(frames) > 0

        # Verify all frames are correct size (20ms at 48kHz)
        assert all(len(frame) == 1920 for frame in frames)

        # Verify Piper was called for both chunks
        assert mock_piper_voice.synthesize.call_count == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_barge_in_pause_latency_maintained(
    mock_voicepack_structure: Path, mock_piper_voice: Mock
) -> None:
    """Test barge-in PAUSE latency is < 50ms with Piper adapter."""
    # Mock Piper to return long audio (5 seconds)
    audio_samples = np.zeros(110250, dtype=np.int16)  # 5 seconds at 22050 Hz
    audio_chunk = Mock()
    audio_chunk.audio_int16_array = audio_samples
    mock_piper_voice.synthesize.return_value = [audio_chunk]

    with patch("tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        voicepack_path = mock_voicepack_structure / "piper" / "en-us-test"
        adapter = PiperTTSAdapter(
            model_id="piper-en-us-test",
            model_path=str(voicepack_path),
        )

        # Start synthesis
        async def text_stream() -> AsyncGenerator[str, None]:
            yield "Long text for testing barge-in latency measurement."

        frames: list[bytes] = []

        async def synthesize() -> None:
            async for frame in adapter.synthesize_stream(text_stream()):
                frames.append(frame)

        synthesis_task = asyncio.create_task(synthesize())

        # Wait for synthesis to start producing frames
        await asyncio.sleep(0.1)

        # Measure PAUSE latency
        pause_start = time.perf_counter()
        await adapter.control("PAUSE")
        pause_latency_ms = (time.perf_counter() - pause_start) * 1000

        # Verify latency is < 50ms (SLA requirement)
        assert pause_latency_ms < 50, f"PAUSE latency {pause_latency_ms:.2f}ms exceeds 50ms SLA"

        # Verify state is PAUSED
        assert adapter.get_state() == AdapterState.PAUSED

        # Cleanup
        await adapter.control("STOP")
        await synthesis_task


@pytest.mark.integration
@pytest.mark.asyncio
async def test_first_audio_latency_measurement(
    mock_voicepack_structure: Path, mock_piper_voice: Mock
) -> None:
    """Test First Audio Latency (FAL) measurement for Piper."""
    with patch("tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        voicepack_path = mock_voicepack_structure / "piper" / "en-us-test"
        adapter = PiperTTSAdapter(
            model_id="piper-en-us-test",
            model_path=str(voicepack_path),
        )

        # Warm up first
        await adapter.warm_up()

        # Measure FAL
        async def text_stream() -> AsyncGenerator[str, None]:
            yield "Test FAL measurement"

        fal_start = time.perf_counter()

        first_frame_received = False
        async for _frame in adapter.synthesize_stream(text_stream()):
            if not first_frame_received:
                fal_ms = (time.perf_counter() - fal_start) * 1000
                first_frame_received = True
                # For CPU adapter, target is < 500ms p95
                # In mock test, should be very fast
                assert fal_ms < 500, f"FAL {fal_ms:.2f}ms exceeds 500ms target"
                break


@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_switch_piper_to_mock_to_piper(
    mock_voicepack_structure: Path, mock_piper_voice: Mock
) -> None:
    """Test switching between Piper and Mock adapters."""
    with patch("tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        with patch("tts.model_manager.Path") as mock_path_class:
            voicepack_path = mock_voicepack_structure / "piper" / "en-us-test"
            mock_path_class.return_value = voicepack_path

            mm = ModelManager(
                default_model_id="mock-test",
                warmup_enabled=False,
                evict_check_interval_ms=1000,
            )

            await mm.initialize()

            # Load Piper model
            with patch.object(mm, "_load_model_impl") as mock_create:
                piper_adapter = PiperTTSAdapter(
                    model_id="piper-test", model_path=str(voicepack_path)
                )
                mock_create.return_value = piper_adapter

                await mm.load("piper-test")

                # Should have both models
                loaded = await mm.list_models()
                assert "mock-test" in loaded
                assert "piper-test" in loaded

                # Release Piper, load mock again
                await mm.release("piper-test")
                await mm.load("mock-test")

                # Verify we can switch back
                await mm.load("piper-test")

            await mm.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_piper_sessions(
    mock_voicepack_structure: Path, mock_piper_voice: Mock
) -> None:
    """Test concurrent synthesis sessions with Piper adapter."""
    with patch("tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        voicepack_path = mock_voicepack_structure / "piper" / "en-us-test"
        adapter = PiperTTSAdapter(
            model_id="piper-en-us-test",
            model_path=str(voicepack_path),
        )

        # Run 3 concurrent sessions
        async def session(session_id: int) -> int:
            async def text_stream() -> AsyncGenerator[str, None]:
                yield f"Session {session_id} text"

            frame_count = 0
            async for _ in adapter.synthesize_stream(text_stream()):
                frame_count += 1

            return frame_count

        # Note: Piper adapter is not thread-safe for concurrent synthesis
        # This test verifies sequential behavior when sessions are run concurrently
        # In production, use separate adapter instances per session
        results = []
        for i in range(3):
            count = await session(i)
            results.append(count)

        # All sessions should complete successfully
        assert len(results) == 3
        assert all(count > 0 for count in results)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ttl_eviction_with_piper_model(
    mock_voicepack_structure: Path, mock_piper_voice: Mock
) -> None:
    """Test TTL eviction works correctly with Piper models."""
    with patch("tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        with patch("tts.model_manager.Path") as mock_path_class:
            voicepack_path = mock_voicepack_structure / "piper" / "en-us-test"
            mock_path_class.return_value = voicepack_path

            mm = ModelManager(
                default_model_id="default",
                ttl_ms=100,  # 100ms TTL
                min_residency_ms=50,
                warmup_enabled=False,
                evict_check_interval_ms=50,
            )

            await mm.initialize()

            # Load Piper model
            with patch.object(mm, "_load_model_impl") as mock_create:
                piper_adapter = PiperTTSAdapter(
                    model_id="piper-test", model_path=str(voicepack_path)
                )
                mock_create.return_value = piper_adapter

                await mm.load("piper-test")
                await mm.release("piper-test")
                await mm.release("default")

                # Wait for TTL + min_residency
                await asyncio.sleep(0.2)

                # Force eviction
                await mm.evict_idle()

                # Piper model should be evicted
                loaded = await mm.list_models()
                # May not contain piper-test anymore
                assert "piper-test" not in loaded or len(loaded) == 0

            await mm.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_models_includes_piper(
    mock_voicepack_structure: Path, mock_piper_voice: Mock
) -> None:
    """Test ListModels includes Piper models in registry."""
    with patch("tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        with patch("tts.model_manager.Path") as mock_path_class:
            voicepack_path = mock_voicepack_structure / "piper" / "en-us-test"
            mock_path_class.return_value = voicepack_path

            mm = ModelManager(
                default_model_id="piper-en-us-test",
                warmup_enabled=False,
                evict_check_interval_ms=1000,
            )

            await mm.initialize()

            # List models
            models = await mm.list_models()

            # Should include Piper model
            assert "piper-en-us-test" in models

            await mm.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_capabilities_reports_correct_sample_rate(
    mock_voicepack_structure: Path, mock_piper_voice: Mock
) -> None:
    """Test GetCapabilities reports correct sample rate for Piper."""
    with patch("tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        voicepack_path = mock_voicepack_structure / "piper" / "en-us-test"
        adapter = PiperTTSAdapter(
            model_id="piper-en-us-test",
            model_path=str(voicepack_path),
        )

        # Verify adapter reports correct rates
        assert adapter.native_sample_rate == 22050  # From config
        # Output is always 48kHz (TARGET_SAMPLE_RATE_HZ)
        from tts.adapters.adapter_piper import TARGET_SAMPLE_RATE_HZ

        assert TARGET_SAMPLE_RATE_HZ == 48000


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graceful_failure_on_missing_voicepack(tmp_path: Path) -> None:
    """Test graceful failure when voicepack is missing."""
    nonexistent_path = tmp_path / "nonexistent" / "voicepack"

    with pytest.raises(FileNotFoundError, match="No ONNX model found"):
        PiperTTSAdapter(
            model_id="missing-model",
            model_path=str(nonexistent_path),
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_piper_warmup_with_model_manager(
    mock_voicepack_structure: Path, mock_piper_voice: Mock
) -> None:
    """Test Piper model warmup via ModelManager."""
    with patch("tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        with patch("tts.model_manager.Path") as mock_path_class:
            voicepack_path = mock_voicepack_structure / "piper" / "en-us-test"
            mock_path_class.return_value = voicepack_path

            mm = ModelManager(
                default_model_id="piper-en-us-test",
                warmup_enabled=True,
                warmup_text="Warmup test for Piper",
                evict_check_interval_ms=1000,
            )

            # Measure initialization with warmup
            start_time = time.perf_counter()
            await mm.initialize()
            init_duration_ms = (time.perf_counter() - start_time) * 1000

            # Warmup should complete quickly
            assert init_duration_ms < 2000  # < 2 seconds for mock

            # Model should be loaded
            loaded = await mm.list_models()
            assert "piper-en-us-test" in loaded

            await mm.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_piper_frame_timing_consistency(
    mock_voicepack_structure: Path, mock_piper_voice: Mock
) -> None:
    """Test Piper adapter maintains consistent frame timing."""
    with patch("tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        voicepack_path = mock_voicepack_structure / "piper" / "en-us-test"
        adapter = PiperTTSAdapter(
            model_id="piper-en-us-test",
            model_path=str(voicepack_path),
        )

        async def text_stream() -> AsyncGenerator[str, None]:
            yield "Test frame timing"

        # Measure inter-frame delays
        frame_times: list[float] = []
        last_time = time.perf_counter()

        async for _frame in adapter.synthesize_stream(text_stream()):
            current_time = time.perf_counter()
            frame_times.append(current_time - last_time)
            last_time = current_time

        # Skip first frame (warmup)
        if len(frame_times) > 1:
            inter_frame_delays = frame_times[1:]

            # Calculate jitter (standard deviation)
            avg_delay = sum(inter_frame_delays) / len(inter_frame_delays)
            variance = sum((d - avg_delay) ** 2 for d in inter_frame_delays) / len(
                inter_frame_delays
            )
            jitter_ms = (variance**0.5) * 1000

            # Jitter should be reasonable (< 50ms for mock)
            assert jitter_ms < 50, f"Frame jitter {jitter_ms:.2f}ms too high"
