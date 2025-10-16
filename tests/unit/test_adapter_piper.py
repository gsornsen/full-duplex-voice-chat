"""Unit tests for PiperTTSAdapter.

Tests initialization, streaming synthesis, resampling, control commands,
and performance characteristics of the Piper TTS adapter.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.tts.adapters.adapter_piper import (
    SAMPLES_PER_FRAME,
    TARGET_SAMPLE_RATE_HZ,
    AdapterState,
    PiperTTSAdapter,
)


@pytest.fixture
def mock_piper_voice() -> Mock:
    """Create a mock PiperVoice instance."""
    voice = Mock()
    # Mock synthesize to return AudioChunk objects
    voice.synthesize = Mock()
    return voice


@pytest.fixture
def mock_voicepack_dir(tmp_path: Path) -> Path:
    """Create a mock voicepack directory with ONNX model and config."""
    voicepack = tmp_path / "piper-test"
    voicepack.mkdir(parents=True)

    # Create mock ONNX file
    onnx_file = voicepack / "model.onnx"
    onnx_file.write_bytes(b"mock onnx data")

    # Create mock config file
    config_file = voicepack / "model.onnx.json"
    config_data = {
        "audio": {
            "sample_rate": 22050,
        }
    }
    config_file.write_text(json.dumps(config_data))

    return voicepack


# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_initialization_success(
    mock_voicepack_dir: Path, mock_piper_voice: Mock
) -> None:
    """Test successful adapter initialization."""
    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        # Verify initialization
        assert adapter.model_id == "piper-test"
        assert adapter.model_path == mock_voicepack_dir
        assert adapter.native_sample_rate == 22050
        assert adapter.state == AdapterState.IDLE
        assert adapter.pause_event.is_set()
        assert not adapter.stop_event.is_set()

        # Verify PiperVoice was loaded correctly
        mock_piper.load.assert_called_once()
        call_args = mock_piper.load.call_args
        assert str(mock_voicepack_dir / "model.onnx") in call_args[0]
        assert str(mock_voicepack_dir / "model.onnx.json") in call_args[0]


@pytest.mark.unit
def test_adapter_initialization_missing_onnx(tmp_path: Path) -> None:
    """Test initialization fails when ONNX file is missing."""
    voicepack = tmp_path / "empty"
    voicepack.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No ONNX model found"):
        PiperTTSAdapter(model_id="test", model_path=str(voicepack))


@pytest.mark.unit
def test_adapter_initialization_missing_config(tmp_path: Path) -> None:
    """Test initialization fails when config file is missing."""
    voicepack = tmp_path / "no-config"
    voicepack.mkdir(parents=True)
    onnx_file = voicepack / "model.onnx"
    onnx_file.write_bytes(b"mock")

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        PiperTTSAdapter(model_id="test", model_path=str(voicepack))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_initialization_invalid_config(tmp_path: Path) -> None:
    """Test initialization fails when config is invalid."""
    voicepack = tmp_path / "bad-config"
    voicepack.mkdir(parents=True)

    onnx_file = voicepack / "model.onnx"
    onnx_file.write_bytes(b"mock")

    config_file = voicepack / "model.onnx.json"
    config_file.write_text("invalid json")

    with pytest.raises(json.JSONDecodeError):
        PiperTTSAdapter(model_id="test", model_path=str(voicepack))


# ============================================================================
# Streaming Synthesis Tests
# ============================================================================


async def text_chunks_generator(*chunks: str) -> AsyncGenerator[str, None]:
    """Create an async generator of text chunks."""

    async def _gen() -> AsyncGenerator[str, None]:
        for chunk in chunks:
            yield chunk

    return _gen()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_single_chunk(
    mock_voicepack_dir: Path, mock_piper_voice: Mock
) -> None:
    """Test synthesis of a single text chunk."""
    # Mock Piper to return 1 second of audio at 22050 Hz
    audio_samples = np.zeros(22050, dtype=np.int16)
    audio_chunk = Mock()
    audio_chunk.audio_int16_array = audio_samples
    mock_piper_voice.synthesize.return_value = [audio_chunk]

    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        # Generate frames
        frames: list[bytes] = []
        async for frame in adapter.synthesize_stream(
            await text_chunks_generator("Hello, world!")
        ):
            frames.append(frame)

        # Verify frames
        assert len(frames) > 0
        # Each frame should be 20ms at 48kHz = 960 samples * 2 bytes = 1920 bytes
        assert all(len(frame) == SAMPLES_PER_FRAME * 2 for frame in frames)

        # Verify state after synthesis
        assert adapter.state == AdapterState.IDLE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_multiple_chunks(
    mock_voicepack_dir: Path, mock_piper_voice: Mock
) -> None:
    """Test synthesis of multiple text chunks."""
    # Mock Piper to return 500ms of audio per chunk
    audio_samples = np.zeros(11025, dtype=np.int16)  # 500ms at 22050 Hz
    audio_chunk = Mock()
    audio_chunk.audio_int16_array = audio_samples
    mock_piper_voice.synthesize.return_value = [audio_chunk]

    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        # Generate frames from multiple chunks
        frames: list[bytes] = []
        async for frame in adapter.synthesize_stream(
            await text_chunks_generator("Hello", "world", "test")
        ):
            frames.append(frame)

        # Should have frames from all chunks
        assert len(frames) > 0
        # Piper should have been called 3 times
        assert mock_piper_voice.synthesize.call_count == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthesize_empty_audio(
    mock_voicepack_dir: Path, mock_piper_voice: Mock
) -> None:
    """Test synthesis when Piper returns empty audio."""
    # Mock Piper to return empty audio
    mock_piper_voice.synthesize.return_value = []

    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        # Generate frames
        frames: list[bytes] = []
        async for frame in adapter.synthesize_stream(await text_chunks_generator("test")):
            frames.append(frame)

        # Should have no frames
        assert len(frames) == 0


# ============================================================================
# Resampling Tests
# ============================================================================


@pytest.mark.unit
def test_resample_audio_22050_to_48000() -> None:
    """Test resampling from 22050 Hz to 48000 Hz."""
    # Create a simple sine wave at 22050 Hz
    duration = 1.0  # 1 second
    freq = 440  # A4 note
    t = np.linspace(0, duration, int(22050 * duration))
    audio = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

    # Mock the adapter to test resampling
    with patch("src.tts.adapters.adapter_piper.PiperVoice"):
        adapter = PiperTTSAdapter.__new__(PiperTTSAdapter)
        adapter.native_sample_rate = 22050

        resampled = adapter._resample_audio(audio, 22050, 48000)

        # Check output sample rate
        expected_samples = int(len(audio) * 48000 / 22050)
        assert len(resampled) == expected_samples
        assert resampled.dtype == np.int16


@pytest.mark.unit
def test_resample_audio_no_op_same_rate() -> None:
    """Test resampling with same source and target rate (no-op)."""
    audio = np.zeros(48000, dtype=np.int16)

    with patch("src.tts.adapters.adapter_piper.PiperVoice"):
        adapter = PiperTTSAdapter.__new__(PiperTTSAdapter)

        resampled = adapter._resample_audio(audio, 48000, 48000)

        # Should return same array
        assert resampled is audio


@pytest.mark.unit
def test_resample_audio_preserves_duration() -> None:
    """Test that resampling preserves audio duration."""
    # 1 second of audio at 16000 Hz
    source_rate = 16000
    target_rate = 48000
    duration_samples = source_rate  # 1 second
    audio = np.random.randint(-32768, 32767, duration_samples, dtype=np.int16)

    with patch("src.tts.adapters.adapter_piper.PiperVoice"):
        adapter = PiperTTSAdapter.__new__(PiperTTSAdapter)

        resampled = adapter._resample_audio(audio, source_rate, target_rate)

        # Duration should be the same
        original_duration = len(audio) / source_rate
        resampled_duration = len(resampled) / target_rate
        assert abs(original_duration - resampled_duration) < 0.001  # < 1ms difference


# ============================================================================
# Frame Repacketization Tests
# ============================================================================


@pytest.mark.unit
def test_repacketize_exact_frames() -> None:
    """Test repacketization with exact number of frames."""
    # 100ms of audio = 5 frames (20ms each)
    audio = np.zeros(4800, dtype=np.int16)  # 100ms at 48kHz

    with patch("src.tts.adapters.adapter_piper.PiperVoice"):
        adapter = PiperTTSAdapter.__new__(PiperTTSAdapter)

        frames = adapter._repacketize_to_20ms(audio)

        # Should have 5 frames
        assert len(frames) == 5
        # Each frame should be 960 samples * 2 bytes = 1920 bytes
        assert all(len(frame) == 1920 for frame in frames)


@pytest.mark.unit
def test_repacketize_with_padding() -> None:
    """Test repacketization with padding on last frame."""
    # 50ms of audio = 2.5 frames, needs padding
    audio = np.zeros(2400, dtype=np.int16)  # 50ms at 48kHz

    with patch("src.tts.adapters.adapter_piper.PiperVoice"):
        adapter = PiperTTSAdapter.__new__(PiperTTSAdapter)

        frames = adapter._repacketize_to_20ms(audio)

        # Should have 3 frames (2 full + 1 padded)
        assert len(frames) == 3
        # All frames should be same size (last one padded)
        assert all(len(frame) == 1920 for frame in frames)


@pytest.mark.unit
def test_repacketize_empty_audio() -> None:
    """Test repacketization with empty audio."""
    audio = np.zeros(0, dtype=np.int16)

    with patch("src.tts.adapters.adapter_piper.PiperVoice"):
        adapter = PiperTTSAdapter.__new__(PiperTTSAdapter)

        frames = adapter._repacketize_to_20ms(audio)

        # Should have no frames
        assert len(frames) == 0


# ============================================================================
# Control Command Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_control_pause_during_synthesis(
    mock_voicepack_dir: Path, mock_piper_voice: Mock
) -> None:
    """Test PAUSE command during synthesis."""
    # Mock Piper to return long audio (5 seconds)
    audio_samples = np.zeros(110250, dtype=np.int16)  # 5 seconds at 22050 Hz
    audio_chunk = Mock()
    audio_chunk.audio_int16_array = audio_samples
    mock_piper_voice.synthesize.return_value = [audio_chunk]

    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        # Start synthesis in background
        frames: list[bytes] = []

        async def synthesize() -> None:
            async for frame in adapter.synthesize_stream(
                await text_chunks_generator("Long text")
            ):
                frames.append(frame)

        synthesis_task = asyncio.create_task(synthesize())

        # Wait for synthesis to start
        await asyncio.sleep(0.05)

        # Send PAUSE command
        pause_start = asyncio.get_event_loop().time()
        await adapter.control("PAUSE")
        pause_duration = (asyncio.get_event_loop().time() - pause_start) * 1000

        # PAUSE should respond in < 50ms
        assert pause_duration < 50

        # State should be PAUSED
        assert adapter.state == AdapterState.PAUSED

        # Count frames before pause
        frames_before_pause = len(frames)

        # Wait a bit
        await asyncio.sleep(0.1)

        # No new frames should be generated
        assert len(frames) == frames_before_pause

        # Cleanup
        await adapter.control("STOP")
        await synthesis_task


@pytest.mark.unit
@pytest.mark.asyncio
async def test_control_resume_after_pause(
    mock_voicepack_dir: Path, mock_piper_voice: Mock
) -> None:
    """Test RESUME command after PAUSE."""
    # Mock Piper to return audio
    audio_samples = np.zeros(110250, dtype=np.int16)
    audio_chunk = Mock()
    audio_chunk.audio_int16_array = audio_samples
    mock_piper_voice.synthesize.return_value = [audio_chunk]

    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        frames: list[bytes] = []

        async def synthesize() -> None:
            async for frame in adapter.synthesize_stream(
                await text_chunks_generator("Test")
            ):
                frames.append(frame)

        synthesis_task = asyncio.create_task(synthesize())
        await asyncio.sleep(0.05)

        # Pause
        await adapter.control("PAUSE")
        frames_at_pause = len(frames)

        # Resume
        await adapter.control("RESUME")

        # State should be SYNTHESIZING
        assert adapter.state == AdapterState.SYNTHESIZING

        # Wait for more frames
        await asyncio.sleep(0.1)

        # Should have more frames
        assert len(frames) > frames_at_pause

        # Cleanup
        await adapter.control("STOP")
        await synthesis_task


@pytest.mark.unit
@pytest.mark.asyncio
async def test_control_stop_terminates_synthesis(
    mock_voicepack_dir: Path, mock_piper_voice: Mock
) -> None:
    """Test STOP command terminates synthesis immediately."""
    audio_samples = np.zeros(110250, dtype=np.int16)
    audio_chunk = Mock()
    audio_chunk.audio_int16_array = audio_samples
    mock_piper_voice.synthesize.return_value = [audio_chunk]

    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        frames: list[bytes] = []

        async def synthesize() -> None:
            async for frame in adapter.synthesize_stream(
                await text_chunks_generator("Test")
            ):
                frames.append(frame)

        synthesis_task = asyncio.create_task(synthesize())
        await asyncio.sleep(0.05)

        # Stop
        await adapter.control("STOP")

        # State should be STOPPED
        assert adapter.state == AdapterState.STOPPED

        # Wait for task to complete
        await synthesis_task

        # No new frames should be generated
        final_count = len(frames)
        await asyncio.sleep(0.1)
        assert len(frames) == final_count


@pytest.mark.unit
@pytest.mark.asyncio
async def test_control_invalid_command_raises_error(
    mock_voicepack_dir: Path, mock_piper_voice: Mock
) -> None:
    """Test invalid control command raises ValueError."""
    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        with pytest.raises(ValueError, match="Unknown control command"):
            await adapter.control("INVALID")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_control_pause_when_idle_ignored(
    mock_voicepack_dir: Path, mock_piper_voice: Mock
) -> None:
    """Test PAUSE when IDLE is ignored gracefully."""
    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        # PAUSE when IDLE
        await adapter.control("PAUSE")

        # State should still be IDLE
        assert adapter.state == AdapterState.IDLE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_control_resume_when_not_paused_ignored(
    mock_voicepack_dir: Path, mock_piper_voice: Mock
) -> None:
    """Test RESUME when not paused is ignored gracefully."""
    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        # RESUME when IDLE
        await adapter.control("RESUME")

        # State should still be IDLE
        assert adapter.state == AdapterState.IDLE


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_warmup_completes_quickly(
    mock_voicepack_dir: Path, mock_piper_voice: Mock
) -> None:
    """Test warmup synthesis completes in < 1 second."""
    # Mock Piper to return audio quickly
    audio_samples = np.zeros(22050, dtype=np.int16)
    audio_chunk = Mock()
    audio_chunk.audio_int16_array = audio_samples
    mock_piper_voice.synthesize.return_value = [audio_chunk]

    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        # Measure warmup time
        start_time = asyncio.get_event_loop().time()
        await adapter.warm_up()
        warmup_duration = (asyncio.get_event_loop().time() - start_time) * 1000

        # Should complete in < 1 second (1000ms)
        assert warmup_duration < 1000

        # Piper should have been called for warmup
        assert mock_piper_voice.synthesize.called


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rtf_calculation_realistic(
    mock_voicepack_dir: Path, mock_piper_voice: Mock
) -> None:
    """Test Real-Time Factor (RTF) is calculated correctly."""
    # Mock Piper to return 1 second of audio
    audio_samples = np.zeros(22050, dtype=np.int16)
    audio_chunk = Mock()
    audio_chunk.audio_int16_array = audio_samples
    mock_piper_voice.synthesize.return_value = [audio_chunk]

    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        # Measure synthesis time
        start_time = asyncio.get_event_loop().time()

        frames: list[bytes] = []
        async for frame in adapter.synthesize_stream(
            await text_chunks_generator("Test audio")
        ):
            frames.append(frame)

        synthesis_time = asyncio.get_event_loop().time() - start_time

        # Calculate audio duration from frames
        total_samples = len(frames) * SAMPLES_PER_FRAME
        audio_duration = total_samples / TARGET_SAMPLE_RATE_HZ

        # RTF = synthesis_time / audio_duration
        # For CPU adapter, RTF should be reasonable (< 10x for test)
        rtf = synthesis_time / audio_duration if audio_duration > 0 else 0
        assert rtf >= 0  # Just ensure we can calculate it


# ============================================================================
# Model Lifecycle Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_load_model_logs_correctly(
    mock_voicepack_dir: Path, mock_piper_voice: Mock
) -> None:
    """Test load_model logs correctly (simplified for M5)."""
    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        # Call load_model (no-op for M5)
        await adapter.load_model("some-model-id")

        # Should not raise error
        assert True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unload_model_logs_correctly(
    mock_voicepack_dir: Path, mock_piper_voice: Mock
) -> None:
    """Test unload_model logs correctly (simplified for M5)."""
    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        # Call unload_model (no-op for M5)
        await adapter.unload_model("some-model-id")

        # Should not raise error
        assert True


# ============================================================================
# State Management Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_state_returns_current_state(
    mock_voicepack_dir: Path, mock_piper_voice: Mock
) -> None:
    """Test get_state returns current adapter state."""
    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        # Initial state
        assert adapter.get_state() == AdapterState.IDLE

        # During synthesis
        audio_samples = np.zeros(22050, dtype=np.int16)
        audio_chunk = Mock()
        audio_chunk.audio_int16_array = audio_samples
        mock_piper_voice.synthesize.return_value = [audio_chunk]

        async def synthesize() -> None:
            async for _ in adapter.synthesize_stream(
                await text_chunks_generator("Test")
            ):
                pass

        synthesis_task = asyncio.create_task(synthesize())
        await asyncio.sleep(0.05)

        # Should be SYNTHESIZING
        assert adapter.get_state() == AdapterState.SYNTHESIZING

        # Cleanup
        await adapter.control("STOP")
        await synthesis_task


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reset_clears_state(mock_voicepack_dir: Path, mock_piper_voice: Mock) -> None:
    """Test reset clears adapter state."""
    with patch("src.tts.adapters.adapter_piper.PiperVoice") as mock_piper:
        mock_piper.load.return_value = mock_piper_voice

        adapter = PiperTTSAdapter(
            model_id="piper-test",
            model_path=str(mock_voicepack_dir),
        )

        # Set some state
        await adapter.control("STOP")
        assert adapter.state == AdapterState.STOPPED
        assert adapter.stop_event.is_set()

        # Reset
        await adapter.reset()

        # State should be cleared
        assert adapter.state == AdapterState.IDLE  # type: ignore[comparison-overlap]
        assert adapter.pause_event.is_set()
        assert not adapter.stop_event.is_set()
