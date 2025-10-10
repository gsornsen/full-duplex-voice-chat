# TTS Worker

## Overview

The TTS worker is a GPU-accelerated synthesis server that implements the unified streaming ABI via gRPC. Workers host model-specific adapters and manage model lifecycle.

**Responsibilities:**
- Serve gRPC streaming synthesis API
- Load and manage TTS models in VRAM
- Repacketize model output to 20ms @ 48kHz frames
- Normalize audio loudness
- Register with Redis for service discovery
- Report capabilities and health metrics

---

## Architecture

```
tts/
├── worker.py              # gRPC server and main entry point
├── model_manager.py       # Model lifecycle management (M4+)
├── tts_base.py            # Adapter protocol/interface
├── adapters/
│   ├── adapter_mock.py    # Mock adapter (440Hz sine wave)
│   ├── adapter_piper.py   # Piper CPU adapter (M5+)
│   ├── adapter_cosyvoice2.py  # CosyVoice 2 GPU (M6+)
│   ├── adapter_xtts.py    # XTTS-v2 GPU (M7+)
│   └── adapter_sesame.py  # Sesame/Unsloth LoRA (M8+)
├── audio/
│   ├── framing.py         # 20ms frame repacketization
│   └── loudness.py        # RMS/LUFS normalization
└── utils/
    ├── logging.py         # Structured logging
    └── timers.py          # Performance timing
```

---

## Key Modules

### worker.py

**Purpose:** gRPC server implementation and adapter host.

**Key Classes:**
- `TTSWorker`: Main worker class hosting gRPC server
- `TTSServicer`: gRPC service implementation

**gRPC Methods Implemented:**
- `StartSession`: Initialize synthesis session
- `EndSession`: Cleanup session resources
- `Synthesize`: Streaming text → audio synthesis
- `Control`: Runtime control (PAUSE/RESUME/STOP)
- `ListModels`: Query available models
- `LoadModel`: Dynamically load model (M4+)
- `UnloadModel`: Unload model from VRAM (M4+)
- `GetCapabilities`: Report worker capabilities

**Example:**
```python
from src.tts.worker import TTSWorker

worker = TTSWorker(
    config_path="configs/worker.yaml",
    adapter_name="mock"  # or "piper", "cosyvoice2", etc.
)

# Start gRPC server
await worker.start()
```

**Usage:**
```bash
# Run worker with mock adapter
just run-tts-mock

# Or directly
python -m src.tts.worker --adapter mock

# With environment overrides
WORKER_NAME=tts-worker-1 LOG_LEVEL=DEBUG python -m src.tts.worker --adapter mock
```

---

### model_manager.py

**Purpose:** Model lifecycle management with TTL eviction and LRU caching (M4+ feature).

**Key Classes:**
- `ModelManager`: Manages model loading, unloading, warmup, and eviction

**Features:**
- Default model preloading at startup
- Optional additional models preloaded
- TTL-based eviction (idle models unloaded after timeout)
- LRU eviction when resident models exceed capacity
- Reference counting (never evict active models)
- Warmup to trigger JIT compilation
- Min residency to prevent thrashing

**Configuration:**
```yaml
model_manager:
  default_model_id: "cosyvoice2-en-base"  # Required
  preload_model_ids: []                   # Optional
  ttl_ms: 600000                          # 10 minutes
  min_residency_ms: 120000                # 2 minutes
  evict_check_interval_ms: 30000          # 30 seconds
  resident_cap: 3                         # Max models in VRAM
  max_parallel_loads: 1                   # Prevent OOM
  warmup_enabled: true
  warmup_text: "This is a warmup test."
```

**Example (M4+):**
```python
from src.tts.model_manager import ModelManager

manager = ModelManager(config)

# Load model
await manager.load("cosyvoice2-en-base")

# Get adapter
adapter = manager.get_adapter("cosyvoice2-en-base")

# Release when session ends (decrement refcount)
manager.release("cosyvoice2-en-base")

# Background eviction runs automatically
```

**Eviction Logic:**
1. Check every `evict_check_interval_ms`
2. For each model:
   - Skip if `refcount > 0` (active sessions)
   - Skip if `loaded_time < min_residency_ms`
   - Unload if `idle_time > ttl_ms`
3. If `resident_count > resident_cap`:
   - Evict LRU model with `refcount == 0`

---

### tts_base.py

**Purpose:** Abstract base class defining the adapter interface.

**Key Classes:**
- `TTSAdapter`: Protocol for all TTS adapters

**Required Methods:**
```python
class TTSAdapter:
    async def synthesize_streaming(
        self,
        text: str,
        session_id: str,
        options: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Stream PCM audio frames (20ms @ 48kHz)."""
        ...

    async def pause(self) -> None:
        """Pause synthesis (< 50ms)."""
        ...

    async def resume(self) -> None:
        """Resume paused synthesis."""
        ...

    async def stop(self) -> None:
        """Stop synthesis and clear queue."""
        ...

    async def warmup(self, text: str) -> None:
        """Warmup model (JIT compilation)."""
        ...

    @property
    def model_info(self) -> dict[str, Any]:
        """Return model metadata."""
        ...
```

**Implementing a New Adapter:**

1. Inherit from `TTSAdapter`
2. Implement all required methods
3. Ensure output is 20ms @ 48kHz PCM frames
4. Register in `worker.py`

**Example:**
```python
from src.tts.tts_base import TTSAdapter
from collections.abc import AsyncIterator

class MyCustomAdapter(TTSAdapter):
    async def synthesize_streaming(
        self, text: str, session_id: str, options: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        # Generate audio
        raw_audio = await self.model.synthesize(text)

        # Resample to 48kHz
        audio_48k = resample(raw_audio, target_rate=48000)

        # Reframe to 20ms chunks (960 samples @ 48kHz)
        for frame in chunk_audio(audio_48k, frame_size=960):
            # Normalize loudness
            frame = normalize_loudness(frame, target_lufs=-16.0)

            # Yield as bytes
            yield frame.astype(np.int16).tobytes()

    async def pause(self) -> None:
        self._paused = True

    async def resume(self) -> None:
        self._paused = False

    async def stop(self) -> None:
        self._stopped = True
        self._queue.clear()
```

---

### adapters/

#### adapter_mock.py

**Mock adapter for testing (M2).**

**Features:**
- Generates 440Hz sine wave
- No GPU required
- Instant synthesis
- Configurable duration

**Usage:**
```bash
just run-tts-mock
```

**Example:**
```python
from src.tts.adapters.adapter_mock import MockAdapter

adapter = MockAdapter()

async for frame in adapter.synthesize_streaming("Hello", "session-1", {}):
    # frame = 1920 bytes of 440Hz sine wave @ 48kHz
    print(f"Frame: {len(frame)} bytes")
```

---

#### adapter_piper.py (M5+)

**Piper TTS adapter (CPU-based).**

**Features:**
- CPU-only synthesis (no GPU required)
- Fast inference (~2-5x real-time)
- Multiple voice models
- Lightweight (~20MB per model)

**Voice Pack Structure:**
```
voicepacks/piper/
├── en-us-lessac-medium/
│   ├── model.onnx
│   ├── config.json
│   └── metadata.yaml
```

---

#### adapter_cosyvoice2.py (M6+)

**CosyVoice 2 adapter (GPU streaming).**

**Features:**
- GPU-accelerated synthesis
- Streaming mode
- Multi-language (en, zh)
- High quality
- ~2-4GB VRAM

**Voice Pack Structure:**
```
voicepacks/cosyvoice2/
├── en-base/
│   ├── model.safetensors
│   ├── config.json
│   └── metadata.yaml
```

---

#### adapter_xtts.py (M7+)

**XTTS-v2 adapter (voice cloning).**

**Features:**
- Zero-shot voice cloning
- Reference audio support
- Expressive synthesis
- ~4-6GB VRAM

**Voice Pack Structure:**
```
voicepacks/xtts-v2/
├── en-demo/
│   ├── model.safetensors
│   ├── config.json
│   ├── metadata.yaml
│   └── ref/
│       └── seed.wav    # Reference audio for cloning
```

---

#### adapter_sesame.py (M8+)

**Sesame/Unsloth adapter (LoRA fine-tuning).**

**Features:**
- LoRA fine-tuned models
- Unsloth optimization
- Fast training
- ~3-5GB VRAM

---

### audio/

#### framing.py

**Purpose:** Repacketize audio to 20ms @ 48kHz frames.

**Key Functions:**
- `resample(audio, target_rate)`: Resample to 48kHz
- `frame_audio(audio, frame_size)`: Chunk into 960-sample frames

**Example:**
```python
from src.tts.audio.framing import resample, frame_audio

# Model outputs 22050 Hz audio
raw_audio = model.synthesize(text)  # shape: (N,)

# Resample to 48000 Hz
audio_48k = resample(raw_audio, target_rate=48000)

# Frame to 20ms chunks (960 samples @ 48kHz)
for frame in frame_audio(audio_48k, frame_size=960):
    # frame.shape = (960,)
    yield frame.astype(np.int16).tobytes()  # 1920 bytes
```

---

#### loudness.py

**Purpose:** Normalize audio loudness to consistent level.

**Key Functions:**
- `normalize_rms(audio, target_rms)`: RMS normalization
- `normalize_lufs(audio, target_lufs)`: LUFS normalization (recommended)

**Example:**
```python
from src.tts.audio.loudness import normalize_lufs

# Normalize to -16 LUFS (streaming/YouTube standard)
normalized = normalize_lufs(audio, target_lufs=-16.0)
```

**Why Normalize?**
- Different models have different output levels
- Consistent volume across sessions
- Avoid clipping or excessively quiet audio

---

## Configuration

**File:** `configs/worker.yaml`

**Key Settings:**

```yaml
worker:
  name: "tts-worker-0"
  grpc_host: "0.0.0.0"
  grpc_port: 7002
  capabilities:
    streaming: true
    zero_shot: true
    lora: false
    cpu_ok: false
    languages: ["en"]
    max_concurrent_sessions: 3

model_manager:
  default_model_id: "cosyvoice2-en-base"
  resident_cap: 3
  ttl_ms: 600000

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
  format: "json"
  include_session_id: true
```

**See:** [Configuration Reference](../../docs/CONFIGURATION_REFERENCE.md)

---

## Usage

### Running TTS Worker

**Docker Compose (Recommended):**
```bash
docker compose up tts-worker
```

**Local Development:**
```bash
# Mock adapter (no GPU)
just run-tts-mock

# CosyVoice adapter (requires GPU)
just run-tts-cosy DEFAULT="cosyvoice2-en-base"

# XTTS adapter (requires GPU)
just run-tts-xtts DEFAULT="xtts-v2-en-demo"

# Piper adapter (CPU)
just run-tts-piper DEFAULT="piper-en-us-lessac-medium"
```

**Multi-GPU Setup:**
```bash
# Worker 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 WORKER_NAME=tts-worker-0 just run-tts-cosy

# Worker 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 WORKER_NAME=tts-worker-1 just run-tts-xtts

# Worker 2 on GPU 2
CUDA_VISIBLE_DEVICES=2 WORKER_NAME=tts-worker-2 just run-tts-sesame
```

---

## Testing

### Unit Tests

```bash
# Test audio framing
pytest tests/unit/test_framing.py

# Test loudness normalization
pytest tests/unit/test_loudness.py

# Test mock adapter
pytest tests/unit/test_adapter_mock.py
```

### Integration Tests

```bash
# Requires Docker (Redis)
pytest tests/integration/test_worker_registration.py

# gRPC synthesis test
pytest tests/integration/test_grpc_synthesis.py
```

---

## Voice Pack Format

### Directory Structure

```
voicepacks/<family>/<model_id>/
├── model.safetensors    # Model weights (or .onnx, .pt)
├── config.json          # Model configuration
├── metadata.yaml        # Model metadata and capabilities
└── ref/                 # Optional reference audio (XTTS)
    └── seed.wav
```

### metadata.yaml

```yaml
family: "cosyvoice2"
model_id: "en-base"
version: "2.0"

capabilities:
  streaming: true
  zero_shot: true
  lora: false
  cpu_ok: false
  emotive_zero_prompt: true

languages: ["en"]

audio:
  sample_rate: 22050
  channels: 1

size:
  vram_mb: 4096
  disk_mb: 2048

tags:
  - "english"
  - "streaming"
  - "high-quality"
```

---

## Performance Tuning

### Concurrent Sessions

```yaml
worker:
  capabilities:
    max_concurrent_sessions: 3  # Tune based on GPU memory
```

**Guidelines:**
- Small models (Piper): 10-20 sessions
- Medium models (CosyVoice): 3-5 sessions
- Large models (XTTS): 1-3 sessions

**VRAM Impact:**
- Each session: ~500MB-2GB depending on model
- Monitor with `nvidia-smi`

### Model Resident Cap

```yaml
model_manager:
  resident_cap: 3  # Max models in VRAM
```

**Calculation:**
```
resident_cap = GPU_VRAM_GB / Average_Model_Size_GB

Examples:
- 24GB GPU, 4GB models → resident_cap = 6
- 12GB GPU, 4GB models → resident_cap = 3
```

---

## Troubleshooting

### Worker Not Registering

**Symptom:** Worker not appearing in Redis

**Solutions:**
1. Check Redis connectivity:
   ```bash
   docker exec -it redis redis-cli ping
   # Expected: PONG
   ```

2. Verify worker logs:
   ```bash
   docker logs tts-worker | grep -i redis
   ```

3. Check Redis keys:
   ```bash
   docker exec -it redis redis-cli
   > KEYS worker:*
   > GET worker:tts-worker-0
   ```

---

### CUDA Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce concurrent sessions:
   ```yaml
   worker:
     capabilities:
       max_concurrent_sessions: 2  # Reduce from 3
   ```

2. Reduce resident models:
   ```yaml
   model_manager:
     resident_cap: 2  # Reduce from 3
   ```

3. Use smaller model variants

4. Monitor VRAM usage:
   ```bash
   watch -n 1 nvidia-smi
   ```

---

### Audio Quality Issues

**Symptom:** Distorted, clipped, or inconsistent audio

**Solutions:**
1. Check loudness normalization:
   ```yaml
   audio:
     normalization_enabled: true
     loudness_target_lufs: -16.0
   ```

2. Verify sample rate:
   ```yaml
   audio:
     output_sample_rate: 48000  # Must be 48000
   ```

3. Check model output format (mono vs stereo)

4. Inspect audio in logs (verbose mode)

---

## Development Workflows

### Adding a New Adapter

1. Create `adapters/adapter_<name>.py`
2. Inherit from `TTSAdapter` in `tts_base.py`
3. Implement all required methods:
   - `synthesize_streaming`
   - `pause`, `resume`, `stop`
   - `warmup`
   - `model_info` property
4. Register in `worker.py`
5. Create voice pack in `voicepacks/<family>/`
6. Add tests in `tests/unit/test_adapter_<name>.py`

**Template:**
```python
from src.tts.tts_base import TTSAdapter

class MyAdapter(TTSAdapter):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    async def synthesize_streaming(self, text, session_id, options):
        # Implement synthesis
        pass

    async def pause(self):
        self._paused = True

    async def resume(self):
        self._paused = False

    async def stop(self):
        self._stopped = True

    @property
    def model_info(self):
        return {"name": "my-model", "version": "1.0"}
```

---

## Further Reading

- [Architecture Diagrams](../../docs/architecture/ARCHITECTURE.md) - Worker architecture
- [Configuration Reference](../../docs/CONFIGURATION_REFERENCE.md) - Worker config
- [Voice Pack Specification](../../docs/VOICE_PACKS.md) - Model format (P2)
- [Performance Tuning](../../docs/PERFORMANCE.md) - Optimization guide (P2)
