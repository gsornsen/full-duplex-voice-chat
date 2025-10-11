# Piper TTS Technical Reference

**Version**: 1.0
**Last Updated**: 2025-10-10
**Python Version**: 3.12+
**Milestone**: M5 - CPU-based TTS Baseline

This document provides comprehensive technical details for the Piper TTS adapter integration, including voicepack structure, configuration, initialization sequence, and operational characteristics.

## Overview

The Piper TTS adapter (`src/tts/adapters/adapter_piper.py`) integrates Piper TTS as the first real TTS model in the system, establishing the baseline for future GPU adapters (M6-M8). Piper is a fast, CPU-only neural TTS system using ONNX Runtime, ideal for edge deployments and low-latency inference.

### Key Characteristics

- **Runtime**: ONNX Runtime (CPU-only, no CUDA)
- **Native Sample Rate**: Typically 22050 Hz (model-dependent)
- **Output Sample Rate**: 48000 Hz (resampled automatically)
- **Frame Format**: 20ms PCM frames at 48kHz (1920 bytes per frame)
- **Model Format**: ONNX (.onnx) + JSON config (.onnx.json)
- **Warmup Time**: <1s on modern CPU (target)
- **First Audio Latency (FAL)**: <500ms p95 (CPU baseline)

## Voicepack Directory Structure

### Required Layout

```
voicepacks/
└── piper/
    └── {voice-name}/           # e.g., "en-us-lessac-medium"
        ├── *.onnx              # ONNX model file (required, exactly one)
        ├── *.onnx.json         # Model configuration (required, matches .onnx name)
        └── metadata.yaml       # Voice metadata (required)
```

### File Naming Convention

The ONNX model and config files **must** follow this naming pattern:
- Model file: `{voice-name}.onnx` (e.g., `en_US-lessac-medium.onnx`)
- Config file: `{voice-name}.onnx.json` (same base name with `.onnx.json` extension)

**Important**: The adapter searches for `*.onnx` files in the directory and expects a matching `.onnx.json` file with the exact same base name.

### Example Voicepack Structure

```
voicepacks/piper/en-us-lessac-medium/
├── en_US-lessac-medium.onnx           # 60.3 MB ONNX model
├── en_US-lessac-medium.onnx.json      # 4.9 KB JSON config
└── metadata.yaml                       # Voice metadata
```

## Required Files

### 1. ONNX Model File (`*.onnx`)

**Purpose**: The trained Piper neural TTS model in ONNX format.

**Requirements**:
- Exactly one `.onnx` file per voicepack directory
- Valid ONNX model compatible with ONNX Runtime
- Typical size: 20-80 MB (varies by model quality)

**Detection**: Adapter uses `glob("*.onnx")` to find the model file.

### 2. Model Configuration (`*.onnx.json`)

**Purpose**: Model configuration including audio parameters, phoneme mappings, and inference settings.

**Required Fields**:
```json
{
  "audio": {
    "sample_rate": 22050,      // Native sample rate (REQUIRED)
    "quality": "medium"         // Model quality variant
  },
  "espeak": {
    "voice": "en-us"           // eSpeak voice for phonemization
  },
  "inference": {
    "noise_scale": 0.667,
    "length_scale": 1,
    "noise_w": 0.8
  },
  "language": {
    "code": "en_US",
    "family": "en",
    "region": "US"
  },
  "piper_version": "1.0.0"     // Piper version compatibility
}
```

**Critical Field**: `audio.sample_rate` - Used by adapter to determine if resampling is needed (target is 48000 Hz).

### 3. Metadata File (`metadata.yaml`)

**Purpose**: Voice pack metadata for model discovery and routing.

**Required Fields**:
```yaml
model_id: "piper-en-us-lessac-medium"  # REQUIRED: Must match worker config
family: "piper"                         # REQUIRED: Model family
language: "en-US"                       # REQUIRED: Language code
voice_name: "Lessac"                    # Voice name
variant: "medium"                       # Quality variant
tags:                                   # Capability tags
  - cpu_ok
  - streaming
  - neural
sample_rate: 22050                      # Native sample rate
description: "High-quality English (US) voice from Piper TTS"
onnx_model: "en_US-lessac-medium.onnx"  # Reference to ONNX file
config_file: "en_US-lessac-medium.onnx.json"  # Reference to config
```

**Tag Definitions**:
- `cpu_ok`: Can run on CPU (required for Piper)
- `streaming`: Supports streaming synthesis
- `neural`: Neural network-based model
- `expressive`: Supports expressive/emotional speech (optional)
- `zero_shot`: Supports zero-shot voice cloning (not applicable to Piper)

## Model Naming and Routing

### Model ID Format

Piper models **must** use the prefix `piper-` followed by the voice name:

```
piper-{voice-name}
```

**Examples**:
- `piper-en-us-lessac-medium`
- `piper-en-us-ryan-high`
- `piper-en-gb-alan-low`

### Routing Logic (ModelManager)

The ModelManager routes model loading based on the model ID prefix:

```python
# In src/tts/model_manager.py (_load_model_impl method)

if model_id.startswith("piper-"):
    # Extract voice name from model_id
    # "piper-en-us-lessac-medium" -> "en-us-lessac-medium"
    voice_name = model_id.replace("piper-", "", 1)
    voicepack_path = Path(f"voicepacks/piper/{voice_name}")

    if not voicepack_path.exists():
        raise ModelNotFoundError(...)

    return PiperTTSAdapter(model_id=model_id, model_path=voicepack_path)
```

**Important**: The voice name portion (after `piper-`) **must** match the voicepack directory name exactly.

## Worker Configuration

### Default Model Configuration (`configs/worker.yaml`)

```yaml
model_manager:
  # Required default model (must exist at startup)
  default_model_id: "piper-en-us-lessac-medium"

  # Optional preload list (load at startup for faster first use)
  preload_model_ids: []

  # Warmup settings
  warmup_enabled: true
  warmup_text: "This is a warmup test."
```

### Environment Variables (Docker)

When running in Docker, the voicepacks directory must be mounted:

```yaml
# docker-compose.yml
services:
  tts0:
    volumes:
      - ./voicepacks:/app/voicepacks:ro  # Mount as read-only
    environment:
      - CUDA_VISIBLE_DEVICES=0  # Not used by Piper (CPU-only)
```

## Initialization Sequence

### Adapter Initialization

1. **Voicepack Discovery**:
   ```python
   onnx_files = list(model_path.glob("*.onnx"))
   if not onnx_files:
       raise FileNotFoundError(f"No ONNX model found in {model_path}")

   onnx_path = onnx_files[0]
   config_path = onnx_path.with_suffix(".onnx.json")
   ```

2. **Load Piper Voice**:
   ```python
   self.voice = PiperVoice.load(
       str(onnx_path),
       str(config_path),
       use_cuda=False  # Always False for Piper
   )
   ```

3. **Read Configuration**:
   ```python
   with open(config_path) as f:
       config = json.load(f)
       self.native_sample_rate = config["audio"]["sample_rate"]
   ```

4. **Initialize State Machine**:
   - State: `IDLE`
   - Pause event: `set()` (unpaused)
   - Stop event: `clear()` (not stopped)

### ModelManager Initialization

1. **Load Default Model**:
   - Calls `load(default_model_id)`
   - Increments refcount to 1
   - Prevents eviction of default model while in use

2. **Warmup** (if enabled):
   - Synthesizes `warmup_text` ("This is a warmup test.")
   - Resamples output to 48kHz
   - Discards audio (warming up ONNX runtime cache)
   - Target: <1s warmup duration

3. **Start Eviction Loop**:
   - Background task checks every 30s (configurable)
   - Evicts models idle > 10min (TTL)
   - Respects min residency (2min minimum)

## Audio Processing Pipeline

### Synthesis Flow

```
Text Input
    ↓
Piper Synthesis (native rate, e.g., 22050 Hz)
    ↓
Resampling (scipy.signal.resample to 48000 Hz)
    ↓
Repacketization (split into 20ms frames = 960 samples)
    ↓
Frame Output (1920 bytes per frame, int16 PCM)
```

### Frame Format

**Specification**:
- **Sample Rate**: 48000 Hz (TARGET_SAMPLE_RATE_HZ)
- **Frame Duration**: 20 ms (FRAME_DURATION_MS)
- **Samples per Frame**: 960 (48000 * 0.020)
- **Bytes per Frame**: 1920 (960 samples * 2 bytes per int16)
- **Encoding**: int16 little-endian PCM
- **Channels**: Mono (1 channel)

**Frame Validation**:
```python
expected_size = 48000 * 20 // 1000 * 2  # 1920 bytes
assert len(frame) == expected_size
```

### Resampling Details

**Method**: `scipy.signal.resample` (high-quality spectral resampling)

**Process**:
1. Convert int16 to float32
2. Calculate target sample count: `int(len(audio) * 48000 / native_rate)`
3. Resample using FFT-based method
4. Convert back to int16

**Performance**: ~5-10ms for typical utterances (500ms audio)

## Control Commands

### PAUSE Command

**Purpose**: Stop emitting frames immediately (barge-in support)

**Behavior**:
- State: `SYNTHESIZING` → `PAUSED`
- `pause_event.clear()` - blocks `synthesize_stream()`
- Response time: <50ms (SLA requirement)

**Implementation**:
```python
async with self.lock:
    if self.state == AdapterState.SYNTHESIZING:
        self.state = AdapterState.PAUSED
        self.pause_event.clear()
```

### RESUME Command

**Purpose**: Continue emitting frames after pause

**Behavior**:
- State: `PAUSED` → `SYNTHESIZING`
- `pause_event.set()` - unblocks `synthesize_stream()`
- Response time: <50ms

### STOP Command

**Purpose**: Terminate streaming permanently

**Behavior**:
- State: any → `STOPPED`
- `stop_event.set()` - signals termination
- `pause_event.set()` - unblock if paused
- Stream returns immediately

## Performance Characteristics

### Latency Measurements (Typical)

| Metric | Target | Typical (Mock) | Notes |
|--------|--------|----------------|-------|
| **First Audio Latency (FAL)** | <500ms p95 | 100-200ms | Time from text input to first frame |
| **PAUSE Latency** | <50ms p95 | 5-15ms | Time from PAUSE command to frame stop |
| **Warmup Duration** | <1s | 200-500ms | Model initialization synthesis |
| **Frame Jitter** | <10ms p95 | <5ms | Inter-frame delay variance |
| **Processing Latency** | N/A | <5ms/frame | Resampling + repacketization |

### Memory Usage (Typical)

| Component | Size | Notes |
|-----------|------|-------|
| **ONNX Model** | 20-80 MB | Depends on quality (low/medium/high) |
| **Runtime Cache** | 10-50 MB | ONNX Runtime working memory |
| **Audio Buffers** | <1 MB | Temporary buffers for resampling |
| **Total per Model** | 30-130 MB | Approximate resident memory |

### Real-Time Factor (RTF)

**Definition**: `RTF = synthesis_time / audio_duration`

**Typical Values**:
- Low quality: RTF ~0.1-0.2 (10-20% of real-time)
- Medium quality: RTF ~0.2-0.4 (20-40% of real-time)
- High quality: RTF ~0.4-0.6 (40-60% of real-time)

**Example**: 1 second of audio synthesized in 200ms → RTF = 0.2

## Error Handling

### Missing Voicepack

**Error**: `FileNotFoundError`

**Cause**: No `.onnx` file found in voicepack directory

**Fix**:
1. Verify voicepack directory exists: `voicepacks/piper/{voice-name}/`
2. Ensure `.onnx` file is present
3. Check file permissions (must be readable)

### Missing Config File

**Error**: `FileNotFoundError: Config file not found: {path}`

**Cause**: No matching `.onnx.json` file for the ONNX model

**Fix**:
1. Ensure config file matches ONNX filename exactly
2. Example: `model.onnx` requires `model.onnx.json`

### Invalid Sample Rate

**Error**: `KeyError: 'sample_rate'` or `KeyError: 'audio'`

**Cause**: Malformed `.onnx.json` config file

**Fix**:
1. Verify JSON is valid: `python -m json.tool config.onnx.json`
2. Ensure `audio.sample_rate` field exists
3. Use Piper official voicepacks as reference

### Model Load Timeout

**Symptom**: Worker startup hangs or times out

**Possible Causes**:
1. ONNX Runtime installation issue
2. Large model on slow disk
3. Missing dependencies

**Fix**:
1. Verify ONNX Runtime installed: `uv pip list | grep onnxruntime`
2. Check worker logs for detailed error
3. Reduce warmup timeout or disable: `warmup_enabled: false`

## Testing and Validation

### Unit Tests

**Location**: `tests/unit/test_piper_adapter.py`

**Coverage**:
- Adapter initialization
- Synthesis stream generation
- Control commands (PAUSE/RESUME/STOP)
- Audio format validation
- Error handling

**Run**:
```bash
just test  # Excludes integration tests
```

### Integration Tests

**Location**: `tests/integration/test_piper_integration.py`

**Coverage**:
- ModelManager with Piper adapter
- End-to-end synthesis sessions
- Barge-in latency validation
- First Audio Latency measurement
- Concurrent sessions
- TTL eviction with Piper models

**Run**:
```bash
# Start Docker services first
docker compose up -d

# Run integration tests with process isolation
just test-integration
```

### Manual Testing with CLI

```bash
# Start Docker Compose (includes Piper worker)
docker compose up --build

# In another terminal, run CLI client
just cli HOST="ws://localhost:8080"

# Send text to synthesize
> Hello, this is a test of Piper TTS.
```

## Common Configuration Mistakes

### 1. Model ID Mismatch

**Mistake**:
```yaml
# worker.yaml
model_manager:
  default_model_id: "en-us-lessac-medium"  # ❌ Missing "piper-" prefix
```

**Fix**:
```yaml
model_manager:
  default_model_id: "piper-en-us-lessac-medium"  # ✅ Correct prefix
```

### 2. Directory Name Mismatch

**Mistake**:
```
voicepacks/piper/lessac/  # Directory name doesn't match model ID
```

**Fix**:
```
voicepacks/piper/en-us-lessac-medium/  # Must match model_id minus "piper-" prefix
```

### 3. File Naming Inconsistency

**Mistake**:
```
voicepacks/piper/en-us-lessac-medium/
├── model.onnx
└── en_US-lessac-medium.onnx.json  # ❌ Names don't match
```

**Fix**:
```
voicepacks/piper/en-us-lessac-medium/
├── en_US-lessac-medium.onnx
└── en_US-lessac-medium.onnx.json  # ✅ Same base name
```

### 4. Missing Metadata

**Mistake**: No `metadata.yaml` file

**Impact**: Model loads but may not be discoverable by routing logic (M9+)

**Fix**: Create `metadata.yaml` with required fields (see template above)

### 5. Incorrect Volume Mount (Docker)

**Mistake**:
```yaml
volumes:
  - ./voicepacks/piper:/app/voicepacks  # ❌ Mounts only piper directory
```

**Fix**:
```yaml
volumes:
  - ./voicepacks:/app/voicepacks:ro  # ✅ Mounts entire voicepacks tree
```

## Code Examples

### Load Piper Model Programmatically

```python
from pathlib import Path
from src.tts.adapters.adapter_piper import PiperTTSAdapter

# Initialize adapter
adapter = PiperTTSAdapter(
    model_id="piper-en-us-lessac-medium",
    model_path=Path("voicepacks/piper/en-us-lessac-medium")
)

# Warm up model
await adapter.warm_up()

# Synthesize text
async def text_chunks():
    yield "Hello, world!"
    yield "This is a test."

async for frame in adapter.synthesize_stream(text_chunks()):
    # Process 20ms audio frame (1920 bytes)
    print(f"Received frame: {len(frame)} bytes")
```

### Test Control Commands

```python
import asyncio
from src.tts.adapters.adapter_piper import PiperTTSAdapter, AdapterState

adapter = PiperTTSAdapter(...)

# Start synthesis
async def text_stream():
    yield "This is a long text to demonstrate pause and resume control."

synthesis_task = asyncio.create_task(
    adapter.synthesize_stream(text_stream())
)

# Wait for synthesis to start
await asyncio.sleep(0.1)

# Pause synthesis
await adapter.control("PAUSE")
assert adapter.get_state() == AdapterState.PAUSED

# Resume synthesis
await adapter.control("RESUME")
assert adapter.get_state() == AdapterState.SYNTHESIZING

# Stop synthesis
await adapter.control("STOP")
assert adapter.get_state() == AdapterState.STOPPED
```

### Measure First Audio Latency

```python
import time
from src.tts.adapters.adapter_piper import PiperTTSAdapter

adapter = PiperTTSAdapter(...)
await adapter.warm_up()  # Warmup first for accurate measurement

async def text_stream():
    yield "Test FAL measurement"

# Measure FAL
start_time = time.perf_counter()

first_frame_received = False
async for frame in adapter.synthesize_stream(text_stream()):
    if not first_frame_received:
        fal_ms = (time.perf_counter() - start_time) * 1000
        print(f"First Audio Latency: {fal_ms:.2f} ms")
        first_frame_received = True
        break
```

## Platform-Specific Notes

### WSL2 (Windows Subsystem for Linux)

**Issue**: gRPC tests may segfault during teardown in WSL2

**Solution**: Use `--forked` flag for process isolation:
```bash
just test-integration  # Automatically uses --forked
```

**Alternative**: Run tests in Docker or native Linux

### macOS

**ONNX Runtime**: May require ARM64 build for Apple Silicon

**Install**:
```bash
uv pip install onnxruntime  # Uses correct platform wheel
```

### Linux

**Recommended Platform**: Native Linux or Docker

**No Known Issues**: Full compatibility with ONNX Runtime

## Python Dependencies

### Required Packages

From `pyproject.toml`:

```toml
[project]
requires-python = ">=3.12,<3.13"
dependencies = [
    "piper-tts>=1.2.0",        # Piper TTS library
    "onnxruntime>=1.16.0",     # ONNX Runtime (CPU)
    "numpy>=1.26.0",           # Numerical operations
    "scipy>=1.11.0",           # Signal processing (resampling)
]
```

### Version Compatibility

| Package | Minimum Version | Tested Version | Notes |
|---------|----------------|----------------|-------|
| **piper-tts** | 1.2.0 | 1.2.0 | Official Piper library |
| **onnxruntime** | 1.16.0 | 1.16.3 | CPU-only version |
| **numpy** | 1.26.0 | 1.26.4 | Array operations |
| **scipy** | 1.11.0 | 1.12.0 | Resampling algorithm |
| **Python** | 3.12 | 3.12.7 | Type hints require 3.12+ |

### Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or install manually
uv pip install piper-tts onnxruntime numpy scipy
```

## Future Enhancements (M6+)

### Planned Features

1. **Dynamic Model Loading** (M9+):
   - Load Piper models on-demand via gRPC `LoadModel` request
   - Support multiple Piper voices simultaneously
   - Automatic TTL-based eviction for unused models

2. **Loudness Normalization** (M6+):
   - Target: -16 LUFS
   - Implement in `src/tts/audio/loudness.py`
   - Apply post-resampling, pre-framing

3. **Voice Cloning** (not applicable to Piper):
   - Piper doesn't support zero-shot cloning
   - Use XTTS-v2 (M7) or CosyVoice 2 (M6) for cloning

4. **Multi-Speaker Support**:
   - Some Piper models support multiple speakers
   - Future: expose speaker selection in API

5. **Streaming Phonemization**:
   - Current: full text → phonemes → audio
   - Future: stream phonemes for lower FAL

## References

### Official Documentation

- **Piper TTS**: https://github.com/rhasspy/piper
- **ONNX Runtime**: https://onnxruntime.ai/docs/
- **eSpeak NG**: https://github.com/espeak-ng/espeak-ng (phonemization)

### Internal Documentation

- `CLAUDE.md`: Project overview and development guidelines
- `docs/CURRENT_STATUS.md`: Implementation status
- `project_documentation/TDD.md`: Technical design document (v2.1)
- `project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md`: Milestone M5

### Source Code

- `src/tts/adapters/adapter_piper.py`: Piper adapter implementation
- `src/tts/model_manager.py`: Model lifecycle management
- `src/tts/worker.py`: gRPC worker server
- `tests/integration/test_piper_integration.py`: Integration tests

### Configuration Files

- `configs/worker.yaml`: Worker configuration template
- `docker-compose.yml`: Docker deployment configuration
- `pyproject.toml`: Python dependencies and tool configuration

## Troubleshooting Checklist

### Before Reporting Issues

1. **Verify Voicepack Structure**:
   ```bash
   ls -R voicepacks/piper/
   # Should show: voicepack-dir/*.onnx, *.onnx.json, metadata.yaml
   ```

2. **Check Model ID Configuration**:
   ```bash
   grep "default_model_id" configs/worker.yaml
   # Should be: piper-{voice-name}
   ```

3. **Validate JSON Config**:
   ```bash
   python -m json.tool voicepacks/piper/*/en_US-lessac-medium.onnx.json
   # Should parse without errors
   ```

4. **Test Worker Startup**:
   ```bash
   docker compose up tts0
   # Watch for "Model loaded successfully" log
   ```

5. **Check Logs for Errors**:
   ```bash
   docker compose logs tts0 | grep -i error
   ```

6. **Run Integration Tests**:
   ```bash
   just test-integration
   # Should pass all Piper-related tests
   ```

### Common Solutions

| Symptom | Solution |
|---------|----------|
| Worker fails to start | Check model ID matches voicepack directory |
| "No ONNX model found" | Verify `.onnx` file exists and is named correctly |
| "Config file not found" | Ensure `.onnx.json` matches `.onnx` filename |
| Synthesis fails silently | Check logs for ONNX Runtime errors |
| Frames have wrong size | Verify resampling to 48kHz is working |
| High CPU usage | Normal for Piper (CPU-intensive synthesis) |
| Memory leak | Check for unreleased ModelManager references |

---

**Document Maintenance**: Update this reference when:
- Piper adapter implementation changes
- New voicepack formats are added
- Performance characteristics change significantly
- New configuration options are introduced
