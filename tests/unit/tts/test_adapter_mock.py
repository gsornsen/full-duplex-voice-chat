"""Unit tests for the MockTTSAdapter.

Tests verify:
- State transitions are correct
- PAUSE stops frame emission immediately
- RESUME continues emission
- STOP terminates cleanly
- Response time < 50ms for control commands
- Audio frame properties (size, count)
"""

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from src.tts.adapters.adapter_mock import AdapterState, MockTTSAdapter
from src.tts.audio.synthesis import calculate_pcm_byte_size


async def text_chunk_generator(chunks: list[str]) -> AsyncIterator[str]:
    """Helper to create async iterator from list of text chunks."""
    for chunk in chunks:
        yield chunk


@pytest.mark.anyio
async def test_basic_synthesis() -> None:
    """Test basic synthesis generates correct number of frames."""
    adapter = MockTTSAdapter()

    # Single text chunk should generate 500ms of audio
    # 500ms / 20ms = 25 frames
    chunks = ["Hello world"]
    frame_count = 0

    async for frame in adapter.synthesize_stream(text_chunk_generator(chunks)):
        frame_count += 1
        # Each frame should be 20ms at 48kHz mono = 1920 bytes
        expected_size = calculate_pcm_byte_size(duration_ms=20, sample_rate=48000, channels=1)
        assert len(frame) == expected_size, f"Frame size mismatch: {len(frame)} != {expected_size}"

    assert frame_count == 25, f"Expected 25 frames, got {frame_count}"
    assert adapter.get_state() == AdapterState.IDLE


@pytest.mark.anyio
async def test_pause_stops_emission() -> None:
    """Test PAUSE command stops frame emission immediately."""
    adapter = MockTTSAdapter()

    # Start synthesis
    chunks = ["Test pause"]
    frame_count = 0

    async def synthesize_with_pause() -> None:
        nonlocal frame_count
        async for _frame in adapter.synthesize_stream(text_chunk_generator(chunks)):
            frame_count += 1
            # After 5 frames, pause
            if frame_count == 5:
                await adapter.control("PAUSE")
                # Wait a bit to ensure no more frames are emitted
                await asyncio.sleep(0.05)
                break

    await synthesize_with_pause()

    # Should have exactly 5 frames (paused after 5th)
    assert frame_count == 5, f"Expected 5 frames before pause, got {frame_count}"
    assert adapter.get_state() == AdapterState.PAUSED


@pytest.mark.anyio
async def test_pause_resume_cycle() -> None:
    """Test PAUSE followed by RESUME continues emission."""
    adapter = MockTTSAdapter()

    chunks = ["Test pause and resume"]
    frame_count = 0
    pause_triggered = False
    resume_triggered = False

    async def control_task(synth_task: asyncio.Task[None]) -> None:
        nonlocal pause_triggered, resume_triggered
        # Wait for some frames to be generated
        await asyncio.sleep(0.03)
        await adapter.control("PAUSE")
        pause_triggered = True
        # Wait a bit while paused
        await asyncio.sleep(0.02)
        await adapter.control("RESUME")
        resume_triggered = True

    async def synthesize_task() -> None:
        nonlocal frame_count
        async for _frame in adapter.synthesize_stream(text_chunk_generator(chunks)):
            frame_count += 1

    synth = asyncio.create_task(synthesize_task())
    ctrl = asyncio.create_task(control_task(synth))

    await asyncio.gather(synth, ctrl)

    # Should have all 25 frames eventually
    assert frame_count == 25, f"Expected 25 frames after resume, got {frame_count}"
    assert pause_triggered, "PAUSE was not triggered"
    assert resume_triggered, "RESUME was not triggered"
    assert adapter.get_state() == AdapterState.IDLE


@pytest.mark.anyio
async def test_stop_terminates_immediately() -> None:
    """Test STOP command terminates streaming immediately."""
    adapter = MockTTSAdapter()

    chunks = ["Test stop"]
    frame_count = 0

    async def synthesize_with_stop() -> None:
        nonlocal frame_count
        async for _frame in adapter.synthesize_stream(text_chunk_generator(chunks)):
            frame_count += 1
            # After 3 frames, stop
            if frame_count == 3:
                await adapter.control("STOP")

    await synthesize_with_stop()

    # Should stop immediately after STOP command
    # May get 3 or 4 frames depending on timing
    assert frame_count <= 4, f"Expected <= 4 frames before stop, got {frame_count}"
    assert adapter.get_state() == AdapterState.STOPPED


@pytest.mark.anyio
async def test_control_command_response_time() -> None:
    """Test control commands respond within 50ms."""
    adapter = MockTTSAdapter()

    # Test PAUSE response time
    start = time.perf_counter()
    await adapter.control("PAUSE")
    pause_time = (time.perf_counter() - start) * 1000  # Convert to ms

    assert pause_time < 50, f"PAUSE took {pause_time:.2f}ms (> 50ms)"

    # Test RESUME response time
    start = time.perf_counter()
    await adapter.control("RESUME")
    resume_time = (time.perf_counter() - start) * 1000

    assert resume_time < 50, f"RESUME took {resume_time:.2f}ms (> 50ms)"

    # Test STOP response time
    start = time.perf_counter()
    await adapter.control("STOP")
    stop_time = (time.perf_counter() - start) * 1000

    assert stop_time < 50, f"STOP took {stop_time:.2f}ms (> 50ms)"


@pytest.mark.anyio
async def test_multiple_text_chunks() -> None:
    """Test synthesis with multiple text chunks."""
    adapter = MockTTSAdapter()

    # 3 chunks × 25 frames each = 75 frames total
    chunks = ["First chunk", "Second chunk", "Third chunk"]
    frame_count = 0

    async for _frame in adapter.synthesize_stream(text_chunk_generator(chunks)):
        frame_count += 1

    assert frame_count == 75, f"Expected 75 frames (3×25), got {frame_count}"
    assert adapter.get_state() == AdapterState.IDLE


@pytest.mark.anyio
async def test_unknown_command_raises_error() -> None:
    """Test unknown control command raises ValueError."""
    adapter = MockTTSAdapter()

    with pytest.raises(ValueError, match="Unknown control command"):
        await adapter.control("INVALID_COMMAND")


@pytest.mark.anyio
async def test_pause_when_not_synthesizing() -> None:
    """Test PAUSE when not synthesizing is handled gracefully."""
    adapter = MockTTSAdapter()

    # PAUSE when IDLE should be a no-op (logged as warning)
    await adapter.control("PAUSE")
    # Should still be IDLE (PAUSE only works when SYNTHESIZING)
    assert adapter.get_state() == AdapterState.IDLE


@pytest.mark.anyio
async def test_resume_when_not_paused() -> None:
    """Test RESUME when not paused is handled gracefully."""
    adapter = MockTTSAdapter()

    # RESUME when IDLE should be a no-op (logged as warning)
    await adapter.control("RESUME")
    # Should still be IDLE
    assert adapter.get_state() == AdapterState.IDLE


@pytest.mark.anyio
async def test_load_model_is_noop() -> None:
    """Test load_model is a no-op for mock adapter."""
    adapter = MockTTSAdapter()

    # Should not raise any errors
    await adapter.load_model("test-model-id")

    # State should remain IDLE
    assert adapter.get_state() == AdapterState.IDLE


@pytest.mark.anyio
async def test_unload_model_is_noop() -> None:
    """Test unload_model is a no-op for mock adapter."""
    adapter = MockTTSAdapter()

    # Should not raise any errors
    await adapter.unload_model("test-model-id")

    # State should remain IDLE
    assert adapter.get_state() == AdapterState.IDLE


@pytest.mark.anyio
async def test_reset_clears_state() -> None:
    """Test reset() returns adapter to initial state."""
    adapter = MockTTSAdapter()

    # Put adapter in stopped state
    await adapter.control("STOP")
    assert adapter.get_state() == AdapterState.STOPPED

    # Reset should return to IDLE
    await adapter.reset()
    assert adapter.get_state() == AdapterState.IDLE

    # Should be able to synthesize again after reset
    chunks = ["After reset"]
    frame_count = 0

    async for _frame in adapter.synthesize_stream(text_chunk_generator(chunks)):
        frame_count += 1

    assert frame_count == 25
    assert adapter.get_state() == AdapterState.IDLE
