"""Unit tests for CosyVoice TTS adapter.

This test suite validates the CosyVoiceAdapter implementation against the TTSAdapter
protocol, ensuring proper state management, control command handling, synthesis
streaming, and GPU-specific functionality.

Test Coverage:
- Initialization (with/without CUDA, with/without CosyVoice2)
- State machine transitions
- Control commands (PAUSE/RESUME/STOP)
- Synthesis streaming (text → frames)
- Resampling (24kHz → 48kHz)
- Frame repacketization (20ms frames)
- Pause/resume during synthesis
- Stop during synthesis
- Warmup with GPU telemetry
- Error handling
- Edge cases
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.tts.adapters.adapter_cosyvoice import CosyVoiceAdapter
from src.tts.tts_base import AdapterState


@pytest.fixture
def mock_model_path(tmp_path: Path) -> Path:
    """Create temporary voicepack directory with dummy model files.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Path to temporary voicepack directory
    """
    voicepack_dir = tmp_path / "cosyvoice-model"
    voicepack_dir.mkdir()

    # Create dummy model files to skip download
    for filename in ["cosyvoice.yaml", "llm.pt", "flow.pt", "hift.pt"]:
        (voicepack_dir / filename).touch()

    return voicepack_dir


@pytest.fixture
def mock_cosyvoice_class() -> Mock:
    """Create a mock CosyVoice2 class.

    Returns:
        Mock class that can be instantiated
    """
    mock_model_instance = Mock()
    mock_model_instance.inference_zero_shot.return_value = [
        {"tts_speech": torch.zeros(1, 24000)}  # 1 second at 24kHz
    ]

    mock_class = Mock(return_value=mock_model_instance)
    return mock_class


@pytest.fixture
def adapter(mock_model_path: Path, mock_cosyvoice_class: Mock) -> CosyVoiceAdapter:
    """Create CosyVoiceAdapter instance with mocked dependencies.

    This fixture patches torch.cuda.is_available() to enable GPU tests
    and mocks the CosyVoice2 class to avoid requiring actual model files.

    Args:
        mock_model_path: Path to temporary voicepack directory
        mock_cosyvoice_class: Mock CosyVoice2 class

    Returns:
        CosyVoiceAdapter instance ready for testing
    """
    with patch("src.tts.adapters.adapter_cosyvoice.torch.cuda.is_available", return_value=True):
        with patch(
            "src.tts.adapters.adapter_cosyvoice.torch.cuda.get_device_name",
            return_value="Mock GPU",
        ):
            with patch("src.tts.adapters.adapter_cosyvoice.CosyVoice2", mock_cosyvoice_class):
                adapter = CosyVoiceAdapter("test-model", mock_model_path)
                return adapter


# ============================================================================
# Initialization Tests
# ============================================================================


def test_init_with_cuda_available(mock_model_path: Path, mock_cosyvoice_class: Mock) -> None:
    """Test adapter initialization when CUDA is available."""
    with patch("src.tts.adapters.adapter_cosyvoice.torch.cuda.is_available", return_value=True):
        with patch(
            "src.tts.adapters.adapter_cosyvoice.torch.cuda.get_device_name",
            return_value="NVIDIA RTX 4090",
        ):
            with patch("src.tts.adapters.adapter_cosyvoice.CosyVoice2", mock_cosyvoice_class):
                adapter = CosyVoiceAdapter("test-model", mock_model_path)

                assert adapter.model_id == "test-model"
                assert adapter.model_path == mock_model_path
                assert adapter.state == AdapterState.IDLE
                assert adapter.pause_event.is_set()  # Start unpaused
                assert not adapter.stop_event.is_set()
                assert adapter.device == torch.device("cuda")
                assert adapter.native_sample_rate == 24000  # CosyVoice 2 native sample rate


def test_init_without_cuda_falls_back_to_cpu(
    mock_model_path: Path, mock_cosyvoice_class: Mock
) -> None:
    """Test adapter initialization falls back to CPU when CUDA is not available."""
    with patch("src.tts.adapters.adapter_cosyvoice.torch.cuda.is_available", return_value=False):
        with patch("src.tts.adapters.adapter_cosyvoice.CosyVoice2", mock_cosyvoice_class):
            adapter = CosyVoiceAdapter("test-model", mock_model_path)

            # Should fall back to CPU, not raise error
            assert adapter.device == torch.device("cpu")
            assert adapter.model is not None


def test_init_without_cosyvoice_raises_import_error(mock_model_path: Path) -> None:
    """Test adapter initialization fails when CosyVoice2 is not installed."""
    with patch("src.tts.adapters.adapter_cosyvoice.torch.cuda.is_available", return_value=True):
        with patch(
            "src.tts.adapters.adapter_cosyvoice.torch.cuda.get_device_name",
            return_value="Mock GPU",
        ):
            with patch("src.tts.adapters.adapter_cosyvoice.CosyVoice2", None):
                with pytest.raises(ImportError, match="CosyVoice2 package not installed"):
                    CosyVoiceAdapter("test-model", mock_model_path)


def test_init_with_string_path(tmp_path: Path, mock_cosyvoice_class: Mock) -> None:
    """Test adapter initialization accepts string paths."""
    model_path = tmp_path / "model"
    model_path.mkdir()

    # Create dummy model files to skip download
    for filename in ["cosyvoice.yaml", "llm.pt", "flow.pt", "hift.pt"]:
        (model_path / filename).touch()

    with patch("src.tts.adapters.adapter_cosyvoice.torch.cuda.is_available", return_value=True):
        with patch(
            "src.tts.adapters.adapter_cosyvoice.torch.cuda.get_device_name",
            return_value="Mock GPU",
        ):
            with patch("src.tts.adapters.adapter_cosyvoice.CosyVoice2", mock_cosyvoice_class):
                adapter = CosyVoiceAdapter("test-model", str(model_path))
                assert adapter.model_path == model_path


def test_init_calls_cosyvoice_constructor_with_correct_params(
    mock_model_path: Path, mock_cosyvoice_class: Mock
) -> None:
    """Test adapter initialization calls CosyVoice2 constructor with correct parameters."""
    with patch("src.tts.adapters.adapter_cosyvoice.torch.cuda.is_available", return_value=True):
        with patch(
            "src.tts.adapters.adapter_cosyvoice.torch.cuda.get_device_name",
            return_value="Mock GPU",
        ):
            with patch(
                "src.tts.adapters.adapter_cosyvoice.CosyVoice2",
                mock_cosyvoice_class,
            ) as mock_cls:
                CosyVoiceAdapter("test-model", mock_model_path)

                # Verify CosyVoice2 was instantiated with correct args
                mock_cls.assert_called_once_with(
                    str(mock_model_path),
                    load_jit=False,
                    load_trt=False,
                    load_vllm=False,
                    fp16=True,
                )


# ============================================================================
# State Machine Tests
# ============================================================================


def test_get_state_returns_current_state(adapter: CosyVoiceAdapter) -> None:
    """Test get_state() returns the current adapter state."""
    assert adapter.get_state() == AdapterState.IDLE
    adapter.state = AdapterState.SYNTHESIZING
    assert adapter.get_state() == AdapterState.SYNTHESIZING


@pytest.mark.asyncio
async def test_reset_clears_state(adapter: CosyVoiceAdapter) -> None:
    """Test reset() returns adapter to initial state."""
    # Simulate state changes
    adapter.state = AdapterState.STOPPED
    adapter.pause_event.clear()
    adapter.stop_event.set()

    await adapter.reset()

    assert adapter.state == AdapterState.IDLE
    assert adapter.pause_event.is_set()
    assert not adapter.stop_event.is_set()


# ============================================================================
# Control Command Tests
# ============================================================================


@pytest.mark.asyncio
async def test_control_pause_from_synthesizing(adapter: CosyVoiceAdapter) -> None:
    """Test PAUSE command transitions from SYNTHESIZING to PAUSED."""
    adapter.state = AdapterState.SYNTHESIZING
    adapter.pause_event.set()

    await adapter.control("PAUSE")

    assert adapter.state == AdapterState.PAUSED
    assert not adapter.pause_event.is_set()


@pytest.mark.asyncio
async def test_control_pause_ignored_when_idle(adapter: CosyVoiceAdapter) -> None:
    """Test PAUSE command is ignored when adapter is IDLE."""
    adapter.state = AdapterState.IDLE
    await adapter.control("PAUSE")
    assert adapter.state == AdapterState.IDLE


@pytest.mark.asyncio
async def test_control_resume_from_paused(adapter: CosyVoiceAdapter) -> None:
    """Test RESUME command transitions from PAUSED to SYNTHESIZING."""
    adapter.state = AdapterState.PAUSED
    adapter.pause_event.clear()

    await adapter.control("RESUME")

    assert adapter.state == AdapterState.SYNTHESIZING
    assert adapter.pause_event.is_set()


@pytest.mark.asyncio
async def test_control_resume_ignored_when_not_paused(adapter: CosyVoiceAdapter) -> None:
    """Test RESUME command is ignored when adapter is not PAUSED."""
    adapter.state = AdapterState.IDLE
    await adapter.control("RESUME")
    assert adapter.state == AdapterState.IDLE


@pytest.mark.asyncio
async def test_control_stop_transitions_to_stopped(adapter: CosyVoiceAdapter) -> None:
    """Test STOP command transitions to STOPPED from any state."""
    adapter.state = AdapterState.SYNTHESIZING

    await adapter.control("STOP")

    assert adapter.state == AdapterState.STOPPED
    assert adapter.stop_event.is_set()
    assert adapter.pause_event.is_set()  # Unblock if paused


@pytest.mark.asyncio
async def test_control_unknown_command_raises_error(adapter: CosyVoiceAdapter) -> None:
    """Test unknown control command raises ValueError."""
    with pytest.raises(ValueError, match="Unknown control command: INVALID"):
        await adapter.control("INVALID")


# ============================================================================
# Synthesis Streaming Tests
# ============================================================================


@pytest.mark.asyncio
async def test_synthesize_stream_yields_frames(adapter: CosyVoiceAdapter) -> None:
    """Test synthesize_stream yields 20ms frames at 48kHz."""
    # Mock synthesis to return known audio (1 second at 24kHz)
    mock_audio = np.zeros(24000, dtype=np.int16)

    with patch.object(adapter, "_synthesize_cosyvoice", return_value=mock_audio):

        async def text_gen() -> AsyncIterator[str]:
            yield "Test text"

        frames = []
        async for frame in adapter.synthesize_stream(text_gen()):
            frames.append(frame)

        # Verify we got frames
        assert len(frames) > 0

        # Each frame should be 960 samples * 2 bytes = 1920 bytes (20ms @ 48kHz)
        for frame in frames:
            assert len(frame) == 1920


@pytest.mark.asyncio
async def test_synthesize_stream_sets_state_to_synthesizing(adapter: CosyVoiceAdapter) -> None:
    """Test synthesize_stream sets state to SYNTHESIZING."""
    mock_audio = np.zeros(1000, dtype=np.int16)

    with patch.object(adapter, "_synthesize_cosyvoice", return_value=mock_audio):

        async def text_gen() -> AsyncIterator[str]:
            yield "Test"

        async for _ in adapter.synthesize_stream(text_gen()):
            # During synthesis, state should be SYNTHESIZING
            assert adapter.state == AdapterState.SYNTHESIZING
            break  # Only check first frame


@pytest.mark.asyncio
async def test_synthesize_stream_resets_to_idle_after_completion(adapter: CosyVoiceAdapter) -> None:
    """Test synthesize_stream resets state to IDLE after completion."""
    mock_audio = np.zeros(1000, dtype=np.int16)

    with patch.object(adapter, "_synthesize_cosyvoice", return_value=mock_audio):

        async def text_gen() -> AsyncIterator[str]:
            yield "Test"

        _ = [frame async for frame in adapter.synthesize_stream(text_gen())]

    # After completion, state should be IDLE
    assert adapter.state == AdapterState.IDLE


@pytest.mark.asyncio
async def test_synthesize_stream_respects_stop_command(adapter: CosyVoiceAdapter) -> None:
    """Test synthesize_stream terminates on STOP command."""
    mock_audio = np.zeros(24000, dtype=np.int16)  # 1 second of audio

    with patch.object(adapter, "_synthesize_cosyvoice", return_value=mock_audio):

        async def text_gen() -> AsyncIterator[str]:
            yield "Test text"

        frames = []
        async for frame in adapter.synthesize_stream(text_gen()):
            frames.append(frame)
            if len(frames) == 2:
                # Stop after 2 frames
                await adapter.control("STOP")

        # Should have stopped early (less frames than full audio)
        # 1 second at 48kHz (after resampling) = 48000 samples / 960 per frame = 50 frames
        assert len(frames) < 50
        assert adapter.state == AdapterState.STOPPED


@pytest.mark.asyncio
async def test_synthesize_stream_respects_pause_resume(adapter: CosyVoiceAdapter) -> None:
    """Test synthesize_stream pauses and resumes on commands."""
    mock_audio = np.zeros(24000, dtype=np.int16)

    with patch.object(adapter, "_synthesize_cosyvoice", return_value=mock_audio):

        async def text_gen() -> AsyncIterator[str]:
            yield "Test text"

        frame_count = 0
        paused = False

        async def pause_after_frames() -> None:
            nonlocal paused
            await asyncio.sleep(0.05)  # Wait for a few frames
            await adapter.control("PAUSE")
            paused = True
            await asyncio.sleep(0.05)
            await adapter.control("RESUME")

        pause_task = asyncio.create_task(pause_after_frames())

        async for _ in adapter.synthesize_stream(text_gen()):
            frame_count += 1
            if frame_count > 10:
                break

        await pause_task
        assert paused  # Verify pause was triggered


@pytest.mark.asyncio
async def test_synthesize_stream_processes_multiple_chunks(adapter: CosyVoiceAdapter) -> None:
    """Test synthesize_stream processes multiple text chunks."""
    mock_audio = np.zeros(5000, dtype=np.int16)  # Small audio chunk

    with patch.object(adapter, "_synthesize_cosyvoice", return_value=mock_audio) as mock_synth:

        async def text_gen() -> AsyncIterator[str]:
            yield "Chunk 1"
            yield "Chunk 2"
            yield "Chunk 3"

        _ = [frame async for frame in adapter.synthesize_stream(text_gen())]

        # Verify _synthesize_cosyvoice was called 3 times
        assert mock_synth.call_count == 3


@pytest.mark.asyncio
async def test_synthesize_stream_handles_empty_text(adapter: CosyVoiceAdapter) -> None:
    """Test synthesize_stream handles empty text gracefully."""
    mock_audio = np.zeros(0, dtype=np.int16)

    with patch.object(adapter, "_synthesize_cosyvoice", return_value=mock_audio):

        async def text_gen() -> AsyncIterator[str]:
            yield ""

        frames = [frame async for frame in adapter.synthesize_stream(text_gen())]

        # Should not crash, may return no frames
        assert isinstance(frames, list)


# ============================================================================
# Resampling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_synthesize_stream_resamples_audio(adapter: CosyVoiceAdapter) -> None:
    """Test synthesize_stream resamples audio from 24kHz to 48kHz."""
    # Mock audio at native sample rate (24kHz)
    mock_audio = np.zeros(24000, dtype=np.int16)

    with patch("src.tts.adapters.adapter_cosyvoice.resample_audio") as mock_resample:
        mock_resample.return_value = np.zeros(48000, dtype=np.int16)  # Resampled to 48kHz

        with patch.object(adapter, "_synthesize_cosyvoice", return_value=mock_audio):

            async def text_gen() -> AsyncIterator[str]:
                yield "Test"

            _ = [frame async for frame in adapter.synthesize_stream(text_gen())]

            # Verify resample_audio was called with correct parameters
            mock_resample.assert_called_once()
            args = mock_resample.call_args[0]
            assert args[1] == 24000  # From native sample rate
            assert args[2] == 48000  # To target sample rate


@pytest.mark.asyncio
async def test_synthesize_stream_skips_resampling_if_same_rate(adapter: CosyVoiceAdapter) -> None:
    """Test synthesize_stream skips resampling if native rate matches target."""
    # Set native sample rate to 48000 (same as target)
    adapter.native_sample_rate = 48000
    mock_audio = np.zeros(48000, dtype=np.int16)

    with patch("src.tts.adapters.adapter_cosyvoice.resample_audio") as mock_resample:
        with patch.object(adapter, "_synthesize_cosyvoice", return_value=mock_audio):

            async def text_gen() -> AsyncIterator[str]:
                yield "Test"

            _ = [frame async for frame in adapter.synthesize_stream(text_gen())]

            # Verify resample_audio was NOT called
            mock_resample.assert_not_called()


# ============================================================================
# Warmup Tests
# ============================================================================


@pytest.mark.asyncio
async def test_warm_up_synthesizes_test_text(adapter: CosyVoiceAdapter) -> None:
    """Test warm_up() synthesizes warmup text."""
    mock_audio = np.zeros(24000, dtype=np.int16)

    with patch.object(adapter, "_synthesize_cosyvoice", return_value=mock_audio) as mock_synth:
        await adapter.warm_up()

        # Verify synthesis was called with warmup text
        mock_synth.assert_called_once()
        args = mock_synth.call_args[0]
        assert "Testing warmup synthesis" in args[0]


@pytest.mark.asyncio
async def test_warm_up_logs_gpu_memory_telemetry(adapter: CosyVoiceAdapter) -> None:
    """Test warm_up() logs GPU memory usage telemetry."""
    mock_audio = np.zeros(24000, dtype=np.int16)

    with patch.object(adapter, "_synthesize_cosyvoice", return_value=mock_audio):
        with patch("src.tts.adapters.adapter_cosyvoice.torch.cuda.is_available", return_value=True):
            with patch(
                "src.tts.adapters.adapter_cosyvoice.torch.cuda.memory_allocated",
                return_value=1024 * 1024 * 100,
            ):  # 100 MB
                with patch(
                    "src.tts.adapters.adapter_cosyvoice.torch.cuda.memory_reserved",
                    return_value=1024 * 1024 * 200,
                ):  # 200 MB
                    await adapter.warm_up()

                    # Test passes if no exceptions raised (telemetry logged successfully)


@pytest.mark.asyncio
async def test_warm_up_handles_no_cuda(adapter: CosyVoiceAdapter) -> None:
    """Test warm_up() handles case when CUDA is not available."""
    mock_audio = np.zeros(24000, dtype=np.int16)

    with patch.object(adapter, "_synthesize_cosyvoice", return_value=mock_audio):
        with patch(
            "src.tts.adapters.adapter_cosyvoice.torch.cuda.is_available",
            return_value=False,
        ):
            # Should not crash
            await adapter.warm_up()


# ============================================================================
# Model Lifecycle Tests
# ============================================================================


@pytest.mark.asyncio
async def test_load_model_logs_info(adapter: CosyVoiceAdapter) -> None:
    """Test load_model() logs information (model already loaded at init)."""
    # For M6, this is a no-op (model loaded at init)
    await adapter.load_model("test-model")
    # Test passes if no exceptions raised


@pytest.mark.asyncio
async def test_unload_model_logs_info(adapter: CosyVoiceAdapter) -> None:
    """Test unload_model() logs information (model lifecycle managed by instance)."""
    # For M6, this is a no-op
    await adapter.unload_model("test-model")
    # Test passes if no exceptions raised


# ============================================================================
# _synthesize_cosyvoice Tests
# ============================================================================


def test_synthesize_cosyvoice_returns_audio(adapter: CosyVoiceAdapter) -> None:
    """Test _synthesize_cosyvoice() returns audio as int16 numpy array."""
    # Mock the model's inference method
    mock_audio_tensor = torch.zeros(1, 24000, dtype=torch.float32)
    adapter.model.inference_zero_shot.return_value = [{"tts_speech": mock_audio_tensor}]

    audio = adapter._synthesize_cosyvoice("Test text")

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.int16
    assert len(audio) == 24000  # 1 second at 24kHz


def test_synthesize_cosyvoice_calls_inference_with_correct_params(
    adapter: CosyVoiceAdapter,
) -> None:
    """Test _synthesize_cosyvoice() calls inference with correct parameters."""
    mock_audio_tensor = torch.zeros(1, 24000, dtype=torch.float32)
    adapter.model.inference_zero_shot.return_value = [{"tts_speech": mock_audio_tensor}]

    _ = adapter._synthesize_cosyvoice("Test text")

    # Verify inference_zero_shot was called with correct args
    adapter.model.inference_zero_shot.assert_called_once()
    call_kwargs = adapter.model.inference_zero_shot.call_args[1]
    assert call_kwargs["tts_text"] == "Test text"
    assert call_kwargs["prompt_text"] == ""  # Empty for default voice
    assert call_kwargs["stream"] is False  # Batch mode for M6
    assert call_kwargs["speed"] == 1.0

    # Verify prompt_speech_16k is torch.Tensor with correct shape
    # Updated for 2-second silence prompt with batch dimension (1, 32000)
    prompt_speech = call_kwargs["prompt_speech_16k"]
    assert isinstance(prompt_speech, torch.Tensor), (
        f"prompt_speech_16k must be torch.Tensor, got {type(prompt_speech)}"
    )
    assert prompt_speech.shape == (1, 32000), (
        f"Expected shape (1, 32000) for 2-second silence with batch dim, got {prompt_speech.shape}"
    )
    assert prompt_speech.dtype == torch.float32, f"Expected float32, got {prompt_speech.dtype}"


def test_synthesize_cosyvoice_handles_empty_audio(adapter: CosyVoiceAdapter) -> None:
    """Test _synthesize_cosyvoice() handles empty audio generation."""
    # Mock empty audio response
    adapter.model.inference_zero_shot.return_value = []

    audio = adapter._synthesize_cosyvoice("Test text")

    assert isinstance(audio, np.ndarray)
    assert len(audio) == 0


def test_synthesize_cosyvoice_raises_error_when_model_none(
    mock_model_path: Path, mock_cosyvoice_class: Mock
) -> None:
    """Test _synthesize_cosyvoice() raises error when model is None."""
    # Create adapter with mocked model
    with patch("src.tts.adapters.adapter_cosyvoice.torch.cuda.is_available", return_value=True):
        with patch(
            "src.tts.adapters.adapter_cosyvoice.torch.cuda.get_device_name",
            return_value="Mock GPU",
        ):
            with patch("src.tts.adapters.adapter_cosyvoice.CosyVoice2", mock_cosyvoice_class):
                adapter = CosyVoiceAdapter("test-model", mock_model_path)
                adapter.model = None  # Explicitly set to None

                with pytest.raises(RuntimeError, match="Model not loaded"):
                    adapter._synthesize_cosyvoice("Test text")


def test_synthesize_cosyvoice_handles_multiple_chunks(adapter: CosyVoiceAdapter) -> None:
    """Test _synthesize_cosyvoice() concatenates multiple audio chunks."""
    # Mock multiple audio chunks
    chunk1 = torch.zeros(1, 12000, dtype=torch.float32)
    chunk2 = torch.zeros(1, 12000, dtype=torch.float32)
    adapter.model.inference_zero_shot.return_value = [
        {"tts_speech": chunk1},
        {"tts_speech": chunk2},
    ]

    audio = adapter._synthesize_cosyvoice("Test text")

    assert isinstance(audio, np.ndarray)
    assert len(audio) == 24000  # 12000 + 12000


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.asyncio
async def test_synthesize_stream_with_no_text_chunks(adapter: CosyVoiceAdapter) -> None:
    """Test synthesize_stream handles empty text chunk iterator."""

    async def empty_gen() -> AsyncIterator[str]:
        if False:
            yield ""

    frames = [frame async for frame in adapter.synthesize_stream(empty_gen())]

    assert frames == []
    assert adapter.state == AdapterState.IDLE


@pytest.mark.asyncio
async def test_multiple_stop_commands_idempotent(adapter: CosyVoiceAdapter) -> None:
    """Test multiple STOP commands are idempotent."""
    adapter.state = AdapterState.SYNTHESIZING

    await adapter.control("STOP")
    assert adapter.state == AdapterState.STOPPED

    await adapter.control("STOP")
    assert adapter.state == AdapterState.STOPPED  # Still stopped


@pytest.mark.asyncio
async def test_concurrent_control_commands_thread_safe(adapter: CosyVoiceAdapter) -> None:
    """Test concurrent control commands are thread-safe with lock protection."""
    adapter.state = AdapterState.SYNTHESIZING

    # Issue multiple control commands concurrently
    tasks = [
        adapter.control("PAUSE"),
        adapter.control("RESUME"),
        adapter.control("STOP"),
    ]

    await asyncio.gather(*tasks)

    # Final state should be STOPPED (last command wins due to async lock)
    assert adapter.state == AdapterState.STOPPED
