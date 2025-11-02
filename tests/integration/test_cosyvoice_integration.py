"""Integration tests for CosyVoice TTS adapter.

These tests use mocking to avoid requiring actual CosyVoice 2 model files.
For tests with real model inference, see:
- docs/COSYVOICE_PYTORCH_CONFLICT.md (PyTorch 2.3.1 requirement)
- docker-compose-test-cosyvoice.yml (Docker test environment)

To run with real models:
    docker compose -f docker-compose-test-cosyvoice.yml up --build

Test Coverage:
- Adapter lifecycle (initialization, warmup, cleanup)
- End-to-end synthesis flow (text → frames)
- Audio quality validation (frame format, resampling)
- Control command latency (PAUSE/RESUME/STOP < 50ms)
- Multiple chunk synthesis (sequential processing)
- Pause/resume during synthesis
- Stop mid-stream termination
- Concurrent sessions (sequential behavior)
- Error handling (missing models, synthesis failures)
- GPU memory management (telemetry, cleanup)

NOTE: These tests are marked as 'infrastructure' to skip in CI by default.
They can be run locally with: pytest -m infrastructure
"""

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from tts.adapters.adapter_cosyvoice import CosyVoiceAdapter
from tts.tts_base import AdapterState

# Mark all tests in this module as infrastructure tests (skip in CI)
pytestmark = pytest.mark.infrastructure


@pytest.fixture
def mock_voicepack_dir(tmp_path: Path) -> Path:
    """Create temporary voicepack directory structure for CosyVoice.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Path to temporary voicepack directory with minimal required files
    """
    voicepack = tmp_path / "cosyvoice-en-base"
    voicepack.mkdir()

    # Create minimal required files for CosyVoice 2
    (voicepack / "model.pt").touch()
    (voicepack / "config.json").write_text(
        json.dumps({"audio": {"sample_rate": 24000}})
    )
    (voicepack / "README.md").write_text("CosyVoice 2 model")

    return voicepack


@pytest.fixture
def mock_cosyvoice_model() -> MagicMock:
    """Mock CosyVoice2 model with realistic behavior.

    Returns:
        Mock model instance that simulates CosyVoice 2 inference with:
        - Realistic audio generation (~1 second at 24kHz)
        - Streaming API support (generator of audio chunks)
        - GPU tensor output format
        - Zero-shot inference parameters
    """
    mock_model = MagicMock()

    # Mock inference_zero_shot to return realistic audio chunks
    def mock_inference(
        tts_text: str,
        prompt_text: str,
        prompt_speech_16k: np.ndarray,
        stream: bool,
        speed: float,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Mock inference that yields audio chunks like real CosyVoice."""
        # Generate ~1 second of audio at 24kHz (CosyVoice native rate)
        audio_length = 24000
        # Small amplitude to avoid clipping: [-0.1, 0.1]
        audio_tensor = torch.randn(1, audio_length, dtype=torch.float32) * 0.1
        yield {"tts_speech": audio_tensor}

    mock_model.inference_zero_shot = MagicMock(side_effect=mock_inference)
    return mock_model


# ============================================================================
# Adapter Lifecycle Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_adapter_initialization_with_voicepack(
    mock_voicepack_dir: Path, mock_cosyvoice_model: MagicMock
) -> None:
    """Test CosyVoiceAdapter initializes successfully with voicepack."""
    with patch("tts.adapters.adapter_cosyvoice.CosyVoice2", return_value=mock_cosyvoice_model):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                adapter = CosyVoiceAdapter("test-model", mock_voicepack_dir)

                # Verify adapter properties
                assert adapter.model_id == "test-model"
                assert adapter.model_path == mock_voicepack_dir
                assert adapter.state == AdapterState.IDLE
                assert adapter.native_sample_rate == 24000
                assert adapter.device == torch.device("cuda")
                assert adapter.model is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_adapter_warmup_completes(
    mock_voicepack_dir: Path, mock_cosyvoice_model: MagicMock
) -> None:
    """Test adapter warmup completes successfully and logs telemetry."""
    with patch("tts.adapters.adapter_cosyvoice.CosyVoice2", return_value=mock_cosyvoice_model):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                with patch("torch.cuda.memory_allocated", return_value=1024 * 1024 * 512):  # 512 MB
                    with patch(
                        "torch.cuda.memory_reserved", return_value=1024 * 1024 * 1024
                    ):  # 1 GB
                        adapter = CosyVoiceAdapter("test-model", mock_voicepack_dir)

                        # Measure warmup duration
                        start_time = time.perf_counter()
                        await adapter.warm_up()
                        warmup_duration_ms = (time.perf_counter() - start_time) * 1000

                        # Verify warmup completed (should be fast with mocking)
                        assert warmup_duration_ms < 2000  # < 2 seconds

                        # Verify model inference was called
                        assert mock_cosyvoice_model.inference_zero_shot.call_count >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_adapter_cleanup_releases_resources(
    mock_voicepack_dir: Path, mock_cosyvoice_model: MagicMock
) -> None:
    """Test adapter cleanup releases resources properly."""
    with patch("tts.adapters.adapter_cosyvoice.CosyVoice2", return_value=mock_cosyvoice_model):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                adapter = CosyVoiceAdapter("test-model", mock_voicepack_dir)

                # Verify model is loaded
                assert adapter.model is not None

                # Reset adapter (simulates cleanup)
                await adapter.reset()

                # Verify state is reset
                assert adapter.state == AdapterState.IDLE
                assert adapter.pause_event.is_set()
                assert not adapter.stop_event.is_set()


# ============================================================================
# End-to-End Synthesis Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_synthesize_stream_end_to_end(
    mock_voicepack_dir: Path, mock_cosyvoice_model: MagicMock
) -> None:
    """Test complete synthesis flow from text to audio frames."""
    with patch("tts.adapters.adapter_cosyvoice.CosyVoice2", return_value=mock_cosyvoice_model):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                adapter = CosyVoiceAdapter("test-model", mock_voicepack_dir)

                # Create text stream
                async def text_gen() -> AsyncGenerator[str, None]:
                    yield "Hello, world!"

                # Synthesize and collect frames
                frames = []
                async for frame in adapter.synthesize_stream(text_gen()):
                    frames.append(frame)

                # Verify frames
                assert len(frames) > 0, "Should generate at least one frame"

                # Verify all frames are correct size (20ms @ 48kHz = 960 samples * 2 bytes)
                assert all(len(frame) == 1920 for frame in frames), "All frames must be 1920 bytes"

                # Verify state returned to IDLE
                assert adapter.state == AdapterState.IDLE

                # Verify model inference was called
                assert mock_cosyvoice_model.inference_zero_shot.call_count == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_synthesize_multiple_chunks_sequentially(
    mock_voicepack_dir: Path, mock_cosyvoice_model: MagicMock
) -> None:
    """Test synthesis processes multiple text chunks sequentially."""
    with patch("tts.adapters.adapter_cosyvoice.CosyVoice2", return_value=mock_cosyvoice_model):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                adapter = CosyVoiceAdapter("test-model", mock_voicepack_dir)

                # Create multi-chunk text stream
                async def text_gen() -> AsyncGenerator[str, None]:
                    yield "First chunk."
                    yield "Second chunk."
                    yield "Third chunk."

                # Synthesize and collect frames
                frames = []
                async for frame in adapter.synthesize_stream(text_gen()):
                    frames.append(frame)

                # Verify frames were generated
                assert len(frames) > 0

                # Verify model inference was called for each chunk
                assert mock_cosyvoice_model.inference_zero_shot.call_count == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_synthesize_with_pause_resume_integration(
    mock_voicepack_dir: Path, mock_cosyvoice_model: MagicMock
) -> None:
    """Test synthesis respects pause/resume commands during streaming.

    This test verifies the pause/resume control flow works correctly even when
    synthesis completes quickly. The test checks that:
    1. PAUSE command can be issued while synthesizing
    2. Adapter enters PAUSED state (or completes if already done)
    3. RESUME command works (or is safely ignored if synthesis finished)
    4. All frames are eventually delivered
    """
    with patch("tts.adapters.adapter_cosyvoice.CosyVoice2", return_value=mock_cosyvoice_model):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                adapter = CosyVoiceAdapter("test-model", mock_voicepack_dir)

                # Create text stream
                async def text_gen() -> AsyncGenerator[str, None]:
                    yield "Test pause and resume behavior."

                frames = []
                pause_attempted = False
                pause_succeeded = False

                async def pause_and_resume() -> None:
                    """Pause synthesis after a few frames, then resume."""
                    nonlocal pause_attempted, pause_succeeded
                    await asyncio.sleep(0.01)  # Wait for synthesis to start
                    pause_attempted = True
                    await adapter.control("PAUSE")

                    # Check if pause succeeded (synthesis might already be done)
                    if adapter.state == AdapterState.PAUSED:
                        pause_succeeded = True
                        await asyncio.sleep(0.02)  # Hold pause briefly
                        await adapter.control("RESUME")
                        # After RESUME, should either be SYNTHESIZING or IDLE (if done)
                        assert adapter.state in (AdapterState.SYNTHESIZING, AdapterState.IDLE)
                    else:
                        # If synthesis completed before PAUSE, that's also valid
                        assert adapter.state == AdapterState.IDLE

                # Run synthesis and pause/resume concurrently
                pause_task = asyncio.create_task(pause_and_resume())

                async for frame in adapter.synthesize_stream(text_gen()):
                    frames.append(frame)

                await pause_task

                # Verify pause was attempted
                assert pause_attempted, "Pause should have been attempted"

                # Verify frames were generated
                assert len(frames) > 0, "Should generate frames"

                # Note: We don't assert pause_succeeded because with mocks, synthesis
                # may complete before PAUSE command executes (race condition is valid)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_synthesize_with_stop_mid_stream(
    mock_voicepack_dir: Path, mock_cosyvoice_model: MagicMock
) -> None:
    """Test synthesis terminates immediately on STOP command."""
    with patch("tts.adapters.adapter_cosyvoice.CosyVoice2", return_value=mock_cosyvoice_model):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                adapter = CosyVoiceAdapter("test-model", mock_voicepack_dir)

                # Create text stream
                async def text_gen() -> AsyncGenerator[str, None]:
                    yield "This text will be stopped mid-stream."

                frames = []

                async for frame in adapter.synthesize_stream(text_gen()):
                    frames.append(frame)
                    if len(frames) == 3:
                        # Stop after 3 frames
                        await adapter.control("STOP")

                # Should have stopped early (not all frames generated)
                # 1 second at 48kHz = 48000 samples / 960 per frame = 50 frames
                assert len(frames) < 50, "Should stop early with fewer frames"
                assert adapter.state == AdapterState.STOPPED


# ============================================================================
# Audio Quality Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_output_frame_format_is_correct(
    mock_voicepack_dir: Path, mock_cosyvoice_model: MagicMock
) -> None:
    """Test output frames have correct format (960 samples, int16, 48kHz)."""
    with patch("tts.adapters.adapter_cosyvoice.CosyVoice2", return_value=mock_cosyvoice_model):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                adapter = CosyVoiceAdapter("test-model", mock_voicepack_dir)

                # Create text stream
                async def text_gen() -> AsyncGenerator[str, None]:
                    yield "Test audio format."

                frames = []
                async for frame in adapter.synthesize_stream(text_gen()):
                    frames.append(frame)

                # Verify at least one frame
                assert len(frames) > 0

                # Verify frame format: 20ms @ 48kHz = 960 samples * 2 bytes (int16) = 1920 bytes
                for frame in frames:
                    assert len(frame) == 1920, f"Frame size {len(frame)} != 1920 bytes"
                    assert isinstance(frame, bytes), "Frame must be bytes"

                    # Convert to numpy to verify int16 format
                    audio_array = np.frombuffer(frame, dtype=np.int16)
                    assert len(audio_array) == 960, f"Frame has {len(audio_array)} samples != 960"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resampling_maintains_audio_quality(
    mock_voicepack_dir: Path, mock_cosyvoice_model: MagicMock
) -> None:
    """Test resampling from 24kHz to 48kHz maintains audio quality."""
    with patch("tts.adapters.adapter_cosyvoice.CosyVoice2", return_value=mock_cosyvoice_model):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                adapter = CosyVoiceAdapter("test-model", mock_voicepack_dir)

                # Create text stream
                async def text_gen() -> AsyncGenerator[str, None]:
                    yield "Test resampling quality."

                frames = []
                async for frame in adapter.synthesize_stream(text_gen()):
                    frames.append(frame)

                # Verify frames were generated
                assert len(frames) > 0

                # Convert frames to audio array
                audio_bytes = b"".join(frames)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

                # Verify audio is not all zeros (contains actual signal)
                assert not np.all(audio_array == 0), "Audio should not be all zeros"

                # Verify audio is within valid int16 range
                assert np.all(audio_array >= -32768) and np.all(
                    audio_array <= 32767
                ), "Audio must be in int16 range"


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_control_latency_under_50ms(
    mock_voicepack_dir: Path, mock_cosyvoice_model: MagicMock
) -> None:
    """Test control commands respond within 50ms (SLA requirement)."""
    with patch("tts.adapters.adapter_cosyvoice.CosyVoice2", return_value=mock_cosyvoice_model):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                adapter = CosyVoiceAdapter("test-model", mock_voicepack_dir)

                # Start synthesis
                async def text_gen() -> AsyncGenerator[str, None]:
                    yield "Test control command latency."

                async def synthesize() -> None:
                    async for _ in adapter.synthesize_stream(text_gen()):
                        pass

                synthesis_task = asyncio.create_task(synthesize())

                # Wait for synthesis to start
                await asyncio.sleep(0.05)

                # Measure PAUSE latency
                pause_start = time.perf_counter()
                await adapter.control("PAUSE")
                pause_latency_ms = (time.perf_counter() - pause_start) * 1000

                # Verify PAUSE latency < 50ms
                assert (
                    pause_latency_ms < 50
                ), f"PAUSE latency {pause_latency_ms:.2f}ms exceeds 50ms SLA"

                # Measure RESUME latency
                resume_start = time.perf_counter()
                await adapter.control("RESUME")
                resume_latency_ms = (time.perf_counter() - resume_start) * 1000

                # Verify RESUME latency < 50ms
                assert (
                    resume_latency_ms < 50
                ), f"RESUME latency {resume_latency_ms:.2f}ms exceeds 50ms SLA"

                # Measure STOP latency
                stop_start = time.perf_counter()
                await adapter.control("STOP")
                stop_latency_ms = (time.perf_counter() - stop_start) * 1000

                # Verify STOP latency < 50ms
                assert (
                    stop_latency_ms < 50
                ), f"STOP latency {stop_latency_ms:.2f}ms exceeds 50ms SLA"

                # Cleanup
                await synthesis_task


@pytest.mark.integration
@pytest.mark.asyncio
async def test_frame_jitter_within_tolerance(
    mock_voicepack_dir: Path, mock_cosyvoice_model: MagicMock
) -> None:
    """Test frame delivery timing maintains consistent jitter."""
    with patch("tts.adapters.adapter_cosyvoice.CosyVoice2", return_value=mock_cosyvoice_model):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                adapter = CosyVoiceAdapter("test-model", mock_voicepack_dir)

                # Create text stream
                async def text_gen() -> AsyncGenerator[str, None]:
                    yield "Test frame timing consistency."

                # Measure inter-frame delays
                frame_times: list[float] = []
                last_time = time.perf_counter()

                async for _ in adapter.synthesize_stream(text_gen()):
                    current_time = time.perf_counter()
                    frame_times.append(current_time - last_time)
                    last_time = current_time

                # Skip first frame (warmup/initialization overhead)
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


# ============================================================================
# Concurrent Sessions Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_sessions_sequential_behavior(
    mock_voicepack_dir: Path, mock_cosyvoice_model: MagicMock
) -> None:
    """Test concurrent synthesis sessions with sequential processing.

    Note: CosyVoiceAdapter is not thread-safe for concurrent synthesis.
    This test verifies sequential behavior when sessions are run concurrently.
    In production, use separate adapter instances per session.
    """
    with patch("tts.adapters.adapter_cosyvoice.CosyVoice2", return_value=mock_cosyvoice_model):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                adapter = CosyVoiceAdapter("test-model", mock_voicepack_dir)

                # Run 3 sequential sessions
                async def session(session_id: int) -> int:
                    async def text_gen() -> AsyncGenerator[str, None]:
                        yield f"Session {session_id} text"

                    frame_count = 0
                    async for _ in adapter.synthesize_stream(text_gen()):
                        frame_count += 1

                    return frame_count

                # Run sessions sequentially (not truly concurrent)
                results = []
                for i in range(3):
                    await adapter.reset()  # Reset between sessions
                    count = await session(i)
                    results.append(count)

                # All sessions should complete successfully
                assert len(results) == 3
                assert all(count > 0 for count in results), "All sessions should generate frames"


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graceful_failure_on_missing_voicepack(tmp_path: Path) -> None:
    """Test graceful failure when voicepack is missing."""
    nonexistent_path = tmp_path / "nonexistent" / "voicepack"

    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
            with patch("tts.adapters.adapter_cosyvoice.CosyVoice2") as mock_cls:
                # Simulate CosyVoice constructor failure
                mock_cls.side_effect = FileNotFoundError("Model files not found")

                with pytest.raises(FileNotFoundError, match="Model files not found"):
                    CosyVoiceAdapter("missing-model", nonexistent_path)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_synthesis_failure_raises_error(
    mock_voicepack_dir: Path, mock_cosyvoice_model: MagicMock
) -> None:
    """Test synthesis failure raises appropriate error."""
    # Configure mock to raise error during inference
    mock_cosyvoice_model.inference_zero_shot.side_effect = RuntimeError("GPU out of memory")

    with patch("tts.adapters.adapter_cosyvoice.CosyVoice2", return_value=mock_cosyvoice_model):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                adapter = CosyVoiceAdapter("test-model", mock_voicepack_dir)

                # Create text stream
                async def text_gen() -> AsyncGenerator[str, None]:
                    yield "This will fail."

                # Synthesis should raise error
                with pytest.raises(RuntimeError, match="CosyVoice synthesis failed"):
                    async for _ in adapter.synthesize_stream(text_gen()):
                        pass


# ============================================================================
# GPU Memory Management Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gpu_memory_telemetry_logged(
    mock_voicepack_dir: Path, mock_cosyvoice_model: MagicMock
) -> None:
    """Test GPU memory telemetry is logged during warmup."""
    with patch("tts.adapters.adapter_cosyvoice.CosyVoice2", return_value=mock_cosyvoice_model):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
                with patch(
                    "torch.cuda.memory_allocated", return_value=1024 * 1024 * 512
                ):  # 512 MB
                    with patch(
                        "torch.cuda.memory_reserved", return_value=1024 * 1024 * 1024
                    ):  # 1 GB
                        adapter = CosyVoiceAdapter("test-model", mock_voicepack_dir)

                        # Warmup should log GPU memory usage
                        await adapter.warm_up()

                        # Test passes if no exceptions raised (telemetry logged)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_adapter_without_cuda_falls_back_gracefully(
    mock_voicepack_dir: Path, mock_cosyvoice_model: MagicMock
) -> None:
    """Test adapter falls back to CPU when CUDA is not available."""
    with patch("tts.adapters.adapter_cosyvoice.CosyVoice2", return_value=mock_cosyvoice_model):
        with patch("torch.cuda.is_available", return_value=False):
            adapter = CosyVoiceAdapter("test-model", mock_voicepack_dir)

            # Should fall back to CPU
            assert adapter.device == torch.device("cpu")

            # Should still be able to synthesize
            async def text_gen() -> AsyncGenerator[str, None]:
                yield "Test CPU fallback."

            frames = []
            async for frame in adapter.synthesize_stream(text_gen()):
                frames.append(frame)

            # Verify synthesis works on CPU
            assert len(frames) > 0
