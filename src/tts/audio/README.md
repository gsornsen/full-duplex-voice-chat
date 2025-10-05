# Audio Processing Utilities

This module provides audio processing utilities for the TTS system, including synthesis, framing, and normalization.

## Synthesis Module (`synthesis.py`)

The synthesis module provides utilities for generating test signals and warmup audio for the TTS system.

### Core Functions

#### `generate_sine_wave(frequency: int, duration_ms: int, sample_rate: int) -> bytes`

Generate a sine wave at the specified frequency.

**Parameters:**

- `frequency`: Frequency in Hz (must be positive and ≤ sample_rate/2)
- `duration_ms`: Duration in milliseconds (must be non-negative)
- `sample_rate`: Sample rate in Hz (will be 48000 for our system)

**Returns:**

- int16 PCM audio data as bytes (little-endian)

**Example:**

```python
from src.tts.audio import generate_sine_wave

# Generate 100ms of 440Hz sine wave at 48kHz
audio_bytes = generate_sine_wave(frequency=440, duration_ms=100, sample_rate=48000)
# Returns 9600 bytes (100ms × 48kHz × 2 bytes/sample)
```

#### `generate_sine_wave_frames(frequency: int, duration_ms: int, sample_rate: int, frame_duration_ms: int = 20) -> list[bytes]`

Generate sine wave audio framed into fixed-duration chunks.

**Parameters:**

- `frequency`: Frequency in Hz
- `duration_ms`: Total duration in milliseconds
- `sample_rate`: Sample rate in Hz (48000 for our system)
- `frame_duration_ms`: Frame duration in milliseconds (default 20ms)

**Returns:**

- List of PCM audio frames, each containing frame_duration_ms of audio

**Example:**

```python
from src.tts.audio import generate_sine_wave_frames

# Generate 100ms of 440Hz sine wave framed into 20ms chunks at 48kHz
frames = generate_sine_wave_frames(
    frequency=440, duration_ms=100, sample_rate=48000, frame_duration_ms=20
)
# Returns 5 frames, each 1920 bytes (20ms × 48kHz × 2 bytes/sample)
```

#### `generate_silence(duration_ms: int, sample_rate: int) -> bytes`

Generate silence (zeros) for the specified duration.

**Parameters:**

- `duration_ms`: Duration in milliseconds
- `sample_rate`: Sample rate in Hz

**Returns:**

- int16 PCM audio data as bytes (all zeros)

**Example:**

```python
from src.tts.audio import generate_silence

# Generate 20ms of silence at 48kHz
silence = generate_silence(duration_ms=20, sample_rate=48000)
# Returns 1920 bytes of zeros
```

### Helper Functions

#### `float32_to_int16_pcm(audio: NDArray[np.float32]) -> bytes`

Convert float32 audio array to int16 PCM bytes.

**Parameters:**

- `audio`: Float32 audio array with values in range [-1.0, 1.0]

**Returns:**

- PCM audio data as bytes (little-endian int16)

**Example:**

```python
import numpy as np
from src.tts.audio import float32_to_int16_pcm

audio = np.array([0.5, -0.5, 0.0], dtype=np.float32)
pcm_bytes = float32_to_int16_pcm(audio)
```

#### `calculate_frame_count(duration_ms: int, frame_duration_ms: int = 20) -> int`

Calculate the number of complete frames for a given duration.

**Parameters:**

- `duration_ms`: Total duration in milliseconds
- `frame_duration_ms`: Frame duration in milliseconds (default 20ms)

**Returns:**

- Number of complete frames

**Example:**

```python
from src.tts.audio import calculate_frame_count

# Calculate how many 20ms frames fit in 100ms
frame_count = calculate_frame_count(duration_ms=100, frame_duration_ms=20)
# Returns 5
```

#### `calculate_pcm_byte_size(duration_ms: int, sample_rate: int, channels: int = 1) -> int`

Calculate the size in bytes of PCM audio for given parameters.

**Parameters:**

- `duration_ms`: Duration in milliseconds
- `sample_rate`: Sample rate in Hz
- `channels`: Number of channels (default 1 for mono)

**Returns:**

- Size in bytes (int16 PCM)

**Example:**

```python
from src.tts.audio import calculate_pcm_byte_size

# Calculate size for 20ms mono at 48kHz
size = calculate_pcm_byte_size(duration_ms=20, sample_rate=48000, channels=1)
# Returns 1920 bytes (960 samples × 2 bytes/sample)
```

## Audio Format Specifications

### Standard TTS Frame Format

The TTS system uses the following standard frame format:

- **Duration**: 20 ms per frame
- **Sample Rate**: 48000 Hz (48 kHz)
- **Channels**: 1 (mono)
- **Encoding**: PCM int16 (little-endian)
- **Samples per frame**: 960 (20ms × 48kHz ÷ 1000)
- **Bytes per frame**: 1920 (960 samples × 2 bytes/sample)

### Frame Timing

For proper real-time audio streaming:

- **Frame duration**: Exactly 20 ms
- **Frame rate**: 50 frames per second
- **Buffer timing**: Emit frames at regular 20 ms intervals
- **Jitter tolerance**: p95 < 10 ms under 3 concurrent sessions

## Testing

Comprehensive unit tests are available in `tests/unit/audio/test_synthesis.py`:

```bash
# Run synthesis tests
uv run pytest tests/unit/audio/test_synthesis.py -v

# Run all audio tests
uv run pytest tests/unit/audio/ -v
```

## Usage Examples

See `examples/test_synthesis.py` for comprehensive usage examples:

```bash
# Run example script
uv run python examples/test_synthesis.py
```

## Type Safety

All functions include complete type annotations and pass strict mypy type checking:

```bash
# Check types
uv run mypy src/tts/audio/synthesis.py --strict
```

## Performance Considerations

- **NumPy vectorization**: All audio generation uses efficient NumPy operations
- **Memory efficiency**: Direct int16 conversion avoids intermediate copies
- **Framing overhead**: Minimal overhead for splitting into frames
- **Zero-copy**: Uses `.tobytes()` for efficient conversion to bytes

## Error Handling

All functions validate inputs and raise `ValueError` with descriptive messages for:

- Invalid frequencies (≤ 0 or above Nyquist limit)
- Invalid durations (negative values)
- Invalid sample rates (≤ 0)
- Empty audio arrays
- Invalid frame durations (≤ 0)

## Integration with TTS Workers

This module is designed to support TTS worker operations:

1. **Model warmup**: Generate synthetic utterances (~300 ms) for model warmup
2. **Testing**: Generate test signals for adapter validation
3. **Debugging**: Create known audio patterns for troubleshooting
4. **Silence padding**: Add silence frames for timing alignment
