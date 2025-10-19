# 🎙️ Realtime Duplex Voice Demo — Technical Design (v2.1)

*Last updated: Oct 2025*
*Owner: Gerald Sornsen*

---

## 0) Platform & Environment (Oct 2025)

* **CUDA Toolkit:** 13.0.1 available, but for stability pair **PyTorch 2.7.0** with **CUDA 12.8** prebuilt wheels.
* **Python:** 3.13.x (let **uv** resolve; lock via `uv.lock`).
* **Docker Engine:** 28.x.
* **Base container pairing:** Prefer `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`. If standardizing on CUDA 13, build or use nightly PyTorch.

---

## 1) System Architecture

```
             ┌──────────────┐              ┌────────────────────────────┐
Browser/CLI  │  Client(s)   │  WebRTC/WS   │ Orchestrator (LiveKit Agent│
(no GPU)     └──────┬───────┘◀────────────▶│ + VAD/ASR + Router + API)  │
                    │                      └───────────┬────────────────┘
                    │                                  gRPC
                    │                                   │
                    │                                   ▼
                    │                          ┌────────────────────────┐
                    │                          │ TTS Worker(s)          │
                    │                          │ (Adapters + ModelMgr)  │
                    │                          └────────────────────────┘
                    │
                    └── LLM Host → (WS/gRPC) → Orchestrator (optional)
```

### Key points

* **Orchestrator** (LiveKit agent): WebRTC, VAD, interruptions, sessioning, routing, ASR (Whisper small/distil).
* **TTS Workers**: Each runs an **Adapter** plus a **Model Manager** providing hot load/unload and TTL-based eviction.
* **Two-process single-GPU default**: *orchestrator* + one *tts-worker@0*.
* **Multi-GPU**: N workers pinned via `CUDA_VISIBLE_DEVICES=i`; discovery via Redis.
* **Multi-host**: Shared Redis for service registry; gRPC between orchestrator and workers.

---

## 2) Realtime behavior & barge-in

* Orchestrator state machine: `LISTENING → SPEAKING → BARGED_IN → LISTENING`.
* VAD edge triggers **immediate** `PAUSE` to worker (target < 50 ms).
* TTS emits **20 ms, 48 kHz mono PCM** frames (or Opus optionally) with a 40–60 ms jitter buffer client-side.
* LLM token streams (optional): orchestrator forwards partial text; applies backpressure during barge-in.

---

## 3) Unified streaming ABI (gRPC)

**IDL (abridged) — `src/rpc/tts.proto`**

```proto
service TTS {
  rpc StartSession (StartSessionReq) returns (StartSessionResp);
  rpc Synthesize (stream TextChunk) returns (stream AudioFrame);
  rpc Control (ControlReq) returns (ControlResp); // PAUSE | RESUME | STOP | RELOAD
  rpc EndSession (EndSessionReq) returns (EndSessionResp);

  // Model lifecycle & registry
  rpc ListModels (ListModelsReq) returns (ListModelsResp);
  rpc LoadModel (LoadModelReq) returns (LoadModelResp);
  rpc UnloadModel (UnloadModelReq) returns (UnloadModelResp);
  rpc GetCapabilities (GetCapReq) returns (GetCapResp);
}
```

**Stream contracts**

* `TextChunk { string text; bool is_final; }`
* `AudioFrame { bytes pcm; uint32 sample_rate=48000; uint32 frame_ms=20; }`

**Control semantics**

* `PAUSE`: stop *emitting* new frames immediately; keep generator/vocoder hot.
* `RESUME`: resume emission from the next chunk boundary (no replay).
* `STOP`: terminate current stream gracefully.
* `RELOAD`: hot-reload voice packs or config (worker-specific).

---

## 4) Model lifecycle management (NEW)

### Goals

* **Default**: load **exactly one default** model to VRAM at startup and warm it.
* **Optional preload list**: load N additional models at startup if requested (e.g., for a demo scenario).
* **Dynamic load/unload**: load on first use; **evict** (unload/free VRAM) after **TTL** when unused.
* **No-thrash** guardrails: min residency time, max concurrent loads, LRU bias.

### Model Manager (inside each TTS worker)

* **Catalog** (`model_catalog.yaml` + discovered `voicepacks/`):

  * `model_id`, `family` (sesame, xtts, cosy, piper, …), `device` prefs, `needs_gpu`, `supports_lora`, `native_sr`, `estimated_vram_gb`, tags.
* **Startup policy**:

  * `default_model_id`: *must* exist → **load & warmup** (~300 ms utterance).
  * `preload_model_ids: [...]` (optional): load sequentially (or max N parallel) then warmup each.
* **Runtime ops**:

  * `LoadModel(model_id)`: if not resident, load (blocking or queued); register in **Resident Map** with `last_used_ts`.
  * `UnloadModel(model_id)`: only if **no active sessions** and **past TTL**, then free VRAM and remove from Resident Map.
  * **Evictor loop** (every `evict_check_interval_ms`): unload any model with `now - last_used_ts > ttl_ms` and `active_sessions == 0`.
  * **LRU cap**: If `resident_models > resident_cap`, evict oldest idle first.
* **Concurrency**:

  * Per-worker **load semaphore** (e.g., 1–2) to avoid VRAM spikes.
  * Per-model refcounts to prevent mid-stream eviction.
* **Warmup**:

  * After load, synthesize a small hidden utterance (e.g., "warm up") to fill caches and prime kernels.

**Config (worker)**

```yaml
worker:
  device: "cuda:0"
  sample_rate: 48000
  frame_ms: 20

model_manager:
  default_model_id: "cosyvoice2-en-base"
  preload_model_ids: []          # e.g., ["xtts-v2-en", "sesame-unsloth-lora01"]
  ttl_ms: 600000                 # 10 minutes idle → unload
  min_residency_ms: 120000       # keep at least 2 minutes
  evict_check_interval_ms: 30000 # every 30s
  resident_cap: 3                # at most 3 models resident
  max_parallel_loads: 1
```

**CLI flags (worker)**

```
--default_model cosyvoice2-en-base
--preload_models xtts-v2-en,sesame-unsloth-lora01
--ttl_ms 600000
--resident_cap 3
--max_parallel_loads 1
```

**Runtime API support**

* Orchestrator may call `LoadModel/UnloadModel` in response to user model switches.
* Router can **prefer resident models** to minimize FAL (first audio latency).

---

## 5) Model families & adapters

**All adapters implement the same streaming surface** and **repacketize** to 20 ms frames.

* **Sesame / Unsloth-Sesame (+LoRA)**

  * PEFT (`peft.PeftModel.from_pretrained`) for LoRA.
  * Option A: in-proc generation; Option B: wrap community HTTP streaming server.
  * Emits PCM; if Mimi tokens are produced, decode to PCM first.
* **Coqui XTTS-v2**

  * Zero-shot & cloning; cache speaker embeddings; optional ONNX export for CPU.
* **CosyVoice 2**

  * Strong "unstated" emotion; expose explicit emotion controls in `settings` as available.

### CosyVoice 2: PyTorch Version Constraint (M6)

**⚠️ IMPORTANT**: CosyVoice 2 requires **PyTorch 2.3.1 + CUDA 12.1**, incompatible with the main project (PyTorch 2.7.0 + CUDA 12.8).

**Problem**: Binary incompatibility prevents running CosyVoice in the same environment as other workers (Orchestrator, Whisper, future XTTS/Sesame).

**Solution**: Docker container isolation with separate PyTorch environment.

**Deployment Strategy**:

1. **Production**: Isolated Docker container (`Dockerfile.tts-cosyvoice`)
   - Base Image: `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`
   - Python 3.10 (CosyVoice tested version)
   - PyTorch 2.3.1 + CUDA 12.1 wheels
   - CosyVoice repository cloned and installed
   - Minimal project dependencies (gRPC, Redis, audio processing only)

2. **Development**: Mock mode with conditional import
   - Adapter implements fallback behavior when CosyVoice not installed
   - Enables fast iteration without Docker overhead
   - Unit tests run in mock mode (main CI pipeline)

3. **Testing Strategy**:
   ```bash
   # Mock unit tests (fast, main CI)
   pytest tests/unit/test_cosyvoice_adapter.py

   # Docker integration tests (complete, separate CI job)
   docker compose run --rm tts-cosyvoice pytest tests/integration/test_cosyvoice.py -v

   # Performance benchmarks (Docker, GPU required)
   docker compose run --rm tts-cosyvoice pytest tests/performance/test_cosyvoice_fal.py -v
   ```

**Configuration** (`.env.cosyvoice`):
```bash
# PyTorch & CUDA
PYTORCH_VERSION=2.3.1
CUDA_VERSION=12.1
PYTHON_VERSION=3.10

# Model paths (volume-mounted from host)
COSYVOICE_MODEL_PATH=/models/cosyvoice2/en-base
COSYVOICE_VOICEPACK_DIR=/models/cosyvoice2

# Worker settings
WORKER_PORT=7002
REDIS_URL=redis://redis:6379
CUDA_VISIBLE_DEVICES=0

# Model Manager
DEFAULT_MODEL_ID=cosyvoice2-en-base
TTL_MS=600000
RESIDENT_CAP=2
```

**Docker Compose Integration**:
```yaml
services:
  tts-cosyvoice:
    build:
      context: .
      dockerfile: Dockerfile.tts-cosyvoice
    environment:
      - REDIS_URL=redis://redis:6379
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./voicepacks/cosyvoice2:/models/cosyvoice2:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
    depends_on: [redis]
```

**Documentation**:
- **Primary**: [docs/COSYVOICE_PYTORCH_CONFLICT.md](../docs/COSYVOICE_PYTORCH_CONFLICT.md) - Detailed analysis and solution strategies
- **Environment**: `.env.cosyvoice` - Configuration template
- **Setup**: [docs/MULTI_GPU.md](../docs/MULTI_GPU.md) - Multi-container deployment guide

**Impact on M6 Timeline**:
- Original estimate: 5-7 days
- Revised estimate: 8-12 days (+60-70% overhead)
- Additional work: Docker setup (2-3 days), deployment complexity (1-2 days), testing infrastructure (1-2 days)

---

* **Spark-TTS / Parler-TTS / MeloTTS / Piper**

  * Similar adapter surface; Piper is CPU/edge fast path; mark capabilities accordingly.


**Voice pack layout**

```
voicepacks/
  <family>/<model_id>/
    model.safetensors | onnx/ | model.pt  # Model files (format varies by family)
    config.json                            # Model configuration
    tokenizer.json (if needed)             # Tokenizer for text processing
    metadata.yaml                          # tags: lang, expressive, cpu_ok, lora, domain, etc.
    ref/seed.wav                           # optional reference audio
```

**Voicepack Documentation**:
- **CosyVoice 2**: [docs/VOICEPACK_COSYVOICE2.md](../docs/VOICEPACK_COSYVOICE2.md) - Complete voicepack structure specification
- **Setup Script**: `scripts/setup_cosyvoice_voicepack.sh` - Automated model download and organization

---

## 6) Routing & capabilities

**Worker announces to Redis**:

```json
{
  "name": "tts-cosyvoice2@0",
  "addr": "grpc://tts-cosy:7002",
  "capabilities": {
    "streaming": true,
    "zero_shot": true,
    "lora": false,
    "cpu_ok": false,
    "languages": ["en", "zh"],
    "emotive_zero_prompt": true
  },
  "audio": {"sr": 48000, "frame_ms": 20},
  "hardware": {"gpu": true, "vram_gb": 8},
  "resident_models": ["cosyvoice2-en-base"],
  "metrics": {"rtf": 0.2, "queue_depth": 0}
}
```

**Routing policy YAML**

```yaml
routing:
  default:
    prefer_tags: ["expressive","low-jitter"]
    prefer_resident: true
    fallbacks: ["cosyvoice2","xtts-v2","sesame"]
  edge:
    require: ["cpu_ok"]
    fallbacks: ["piper","melo"]
  lora:
    require: ["lora"]
```

**Selection logic**

1. Filter by mode (`lora`, `edge`, `default`), language, and sample rate.
2. Prefer **resident** models; pick **lowest queue_depth**, then **best p50 latency**.
3. If requested model not resident, optionally **pre-load** (async) and route to an already-resident fallback for the current utterance.

---

## 7) Developer Experience & Tooling

### `pyproject.toml` (keep uv-driven resolution; no Python pin)

```toml
[project]
name = "realtime-duplex-demo"
version = "0.1.0"
dependencies = [
  "aiortc",
  "websockets",
  "livekit>=0.6",
  "pydantic>=2",
  "grpcio",
  "grpcio-tools",
  "protobuf",
  "redis",
  "numpy",
  "soundfile",
  "webrtcvad",
  "torch",
  "torchaudio",
  "transformers",
  "accelerate",
  "peft",
  "huggingface_hub",
  "openai-whisper",
  "watchfiles",
  "pytest",
  "mypy",
  "ruff",
]

[tool.ruff]
line-length = 100
lint.select = ["E","F","I","UP","B","S"]
target-version = "py310"

[tool.mypy]
disallow_untyped_defs = true
strict_optional = true
warn_unused_ignores = true
ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "8.0"
addopts = "-q"
```

### `justfile`

```make
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# ===== Quality =====
lint:        uv run ruff check .
fix:         uv run ruff check . --fix
typecheck:   uv run mypy src
test:        uv run pytest -q
ci:          just lint && just typecheck && just test

# ===== Infra =====
redis:
  docker run --rm -p 6379:6379 --name redis redis:7

# ===== Generators =====
gen-proto:
  uv run python -m grpc_tools.protoc -Isrc/rpc --python_out=src/rpc/generated --grpc_python_out=src/rpc/generated src/rpc/tts.proto

# ===== Runtime (Single-GPU Option B) =====
run-tts-sesame DEFAULT?="cosyvoice2-en-base" PRELOAD?="":
  export CUDA_VISIBLE_DEVICES=0
  uv run watchfiles --filter python "uv run python -m src.tts.worker --adapter sesame --port 7001 --default_model {{DEFAULT}} {{ if PRELOAD != '' { printf \"--preload_models %s\" PRELOAD } }}"

run-tts-cosy DEFAULT?="cosyvoice2-en-base" PRELOAD?="":
  export CUDA_VISIBLE_DEVICES=0
  uv run watchfiles --filter python "uv run python -m src.tts.worker --adapter cosyvoice2 --port 7001 --default_model {{DEFAULT}} {{ if PRELOAD != '' { printf \"--preload_models %s\" PRELOAD } }}"

run-orch: redis
  uv run watchfiles --filter python "uv run python -m src.orchestrator.server --livekit --ws"

# ===== Clients =====
cli HOST="ws://localhost:8080":
  uv run python -m src.client.cli_client --host {{HOST}}

# ===== Profiling (py-spy) =====
spy-top PID:
  py-spy top --pid {{PID}}

spy-record PID OUT="profile.svg":
  py-spy record -o {{OUT}} --pid {{PID}} --duration 30

# ===== GPU Profiling =====
nsys-tts:
  nsys profile -t cuda,nvtx -o nsys_tts uv run python -m src.tts.worker --adapter sesame --port 7001

ncu-tts:
  ncu --set full --target-processes all -- uv run python -m src.tts.worker --adapter sesame --port 7001
```

---

## 8) Containers & Compose

### `Dockerfile.orchestrator`

```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml .
RUN pip3 install uv
COPY . .
RUN uv pip install -e .
ENV REDIS_URL=redis://redis:6379
EXPOSE 8080 8443
CMD ["uv","run","python","-m","src.orchestrator.server","--livekit","--ws"]
```

### `Dockerfile.tts`

```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml .
RUN pip3 install uv
COPY . .
RUN uv pip install -e .
EXPOSE 7001
ENTRYPOINT ["uv","run","python","-m","src.tts.worker","--adapter","sesame","--port","7001"]
```

### `docker-compose.yml`

```yaml
version: "3.9"
services:
  redis:
    image: redis:7
    ports: ["6379:6379"]

  orchestrator:
    build:
      context: .
      dockerfile: Dockerfile.orchestrator
    environment:
      - REDIS_URL=redis://redis:6379
    ports: ["8080:8080"]
    depends_on: [redis]

  tts0:
    build:
      context: .
      dockerfile: Dockerfile.tts
    environment:
      - REDIS_URL=redis://redis:6379
    command:
      - "uv"; "run"; "python"; "-m"; "src.tts.worker"; "--adapter"; "sesame"
      - "--port"; "7001"; "--default_model"; "cosyvoice2-en-base"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
    depends_on: [redis]
```

---

## 9) Repo structure (tree-style)

```
realtime-duplex-demo/
├─ pyproject.toml
├─ uv.lock                        # generated after first lock
├─ justfile
├─ docker-compose.yml
├─ Dockerfile.orchestrator
├─ Dockerfile.tts
├─ .env.example
├─ README.md
├─ docs/
│  ├─ design_v2.1.md              # this document
│  └─ profiling.md
├─ configs/
│  ├─ orchestrator.yaml
│  └─ worker.yaml                 # model_manager config (default/preload/ttl)
├─ voicepacks/
│  ├─ cosyvoice2/en-base/
│  │  ├─ model.safetensors
│  │  ├─ config.json
│  │  └─ metadata.yaml
│  └─ xtts-v2/en-demo/
│     ├─ model.safetensors
│     ├─ config.json
│     └─ metadata.yaml
├─ src/
│  ├─ orchestrator/
│  │  ├─ server.py                # LiveKit Agent + WS fallback, sessions, VAD, ASR, routing
│  │  ├─ vad.py
│  │  ├─ asr.py
│  │  ├─ llm_bridge.py
│  │  ├─ routing.py               # capability-aware, prefer resident, Redis discovery
│  │  ├─ registry.py              # Redis announce/discover
│  │  └─ config.py
│  ├─ tts/
│  │  ├─ worker.py                # gRPC server; adapter host; ModelManager
│  │  ├─ model_manager.py         # load/unload, TTL eviction, warmup, LRU
│  │  ├─ tts_base.py              # Protocol, common stream/repacketize utils
│  │  ├─ adapters/
│  │  │  ├─ adapter_sesame.py
│  │  │  ├─ adapter_unsloth_sesame.py
│  │  │  ├─ adapter_xtts.py
│  │  │  ├─ adapter_cosyvoice2.py
│  │  │  ├─ adapter_spark_tts.py
│  │  │  ├─ adapter_parler_tts.py
│  │  │  ├─ adapter_melo.py
│  │  │  └─ adapter_piper.py
│  │  ├─ audio/
│  │  │  ├─ framing.py            # 20ms framing, resample to 48kHz
│  │  │  └─ loudness.py           # RMS/LUFS normalization
│  │  ├─ utils/
│  │  │  ├─ logging.py
│  │  │  └─ timers.py
│  │  └─ config.py
│  ├─ rpc/
│  │  ├─ tts.proto
│  │  └─ generated/               # grpcio-tools output
│  └─ client/
│     ├─ cli_client.py
│     └─ web/
│        ├─ index.html
│        └─ app.js
├─ tests/
│  ├─ unit/
│  │  ├─ test_vad_edges.py
│  │  ├─ test_routing.py
│  │  ├─ test_tts_control.py      # PAUSE/RESUME/STOP semantics
│  │  ├─ test_model_manager.py    # load/unload/ttl/evict/LRU
│  │  └─ test_audio_framing.py
│  └─ integration/
│     ├─ test_loopback_ws.py      # asserts 20ms frames & FAL
│     ├─ test_barge_in.py
│     └─ test_preload_defaults.py
└─ .github/
   └─ workflows/
      └─ ci.yml                   # just ci
```

---

## 10) Key modules (behavioral details)

### `src/tts/model_manager.py`

* **Data structures**

  * `ResidentModel { model_id, family, loaded_at, last_used_ts, refcount, device, tags, is_default }`
  * `Catalog { model_id → ModelMeta }` loaded from `voicepacks/*/metadata.yaml` and `model_catalog.yaml`.
* **API**

  * `await load(model_id)`: respects `max_parallel_loads`; increments refcount on acquire.
  * `release(model_id)`: decrement refcount, update `last_used_ts`.
  * `evict_idle(now)`: unload models with `refcount==0` and `idle>ttl_ms`, honoring `min_residency_ms`.
  * `list_resident()` / `list_catalog()`.
* **Warmup**: after load, run `synthesize("warm up", ~300ms)` through adapter → discard audio.

### `src/tts/worker.py`

* **Startup**: parse CLI/config; **load + warmup** `default_model_id`; preload list; start **evictor task**.
* **gRPC handlers**: `StartSession`, `Synthesize` (streams), `Control`, `EndSession`.
* **Model switch during a session**: not allowed; client must end session & start a new one; orchestrator handles UX.
* **Backpressure**: queue with bounded size per session; drop or stall gracefully (configurable).

### `src/orchestrator/routing.py`

* **Inputs**: session request (language, need_lora?, preferred model/family, expressiveness), worker metrics from Redis.
* **Policy**: prefer **resident** models; lowest `queue_depth`, then lowest reported p50; if requested model not loaded, **fire async LoadModel**; route current request to best resident candidate now.

---

## 11) Config samples

### Orchestrator (`configs/orchestrator.yaml`)

```yaml
transport:
  webrtc: true
  ws_fallback: true
asr:
  engine: "whisper-small"
vad:
  aggressiveness: 2
routing:
  policy: "default"
  prefer_resident: true
redis:
  url: "redis://localhost:6379"
```

### Worker (`configs/worker.yaml`)

```yaml
worker:
  device: "cuda:0"
  sample_rate: 48000
  frame_ms: 20
  adapter: "cosyvoice2"

model_manager:
  default_model_id: "cosyvoice2-en-base"
  preload_model_ids: []
  ttl_ms: 600000
  min_residency_ms: 120000
  evict_check_interval_ms: 30000
  resident_cap: 3
  max_parallel_loads: 1
```

---

## 12) Testing, profiling, observability

* **Unit**: VAD edges, routing policy, adapter control semantics, framing, **model_manager (load/unload/TTL/LRU)**.
* **Integration**: loopback (FAL, 20 ms cadence), barge-in stop within 50 ms, preload defaults honored.
* **CI**: `just ci` (ruff+mypy+pytest).
* **Python profiling**: `py-spy` (`just spy-top`, `just spy-record`).
* **GPU profiling**: Nsight Systems/Compute; PyTorch Profiler with NVTX ranges around phases: ASR, LLM bridge, TTS, repacketize.
* **Metrics**: Prometheus counters for first-audio latency, frame jitter p95, queue depth, barge-in count; structured JSON logs.

---

## 13) Security, privacy, operational notes

* **Audio retention**: off by default; opt-in with retention policy.
* **TLS**: terminate HTTPS/WSS at nginx/traefik if WAN exposed.
* **Auth**: demo scope = API key; mTLS between orchestrator and worker optional.
* **Isolation**: each worker process pinned to device; crashes don't take sessions down.

---

## 14) Runbooks

**Single-GPU (default)**

```bash
just redis
just run-tts-cosy DEFAULT="cosyvoice2-en-base" PRELOAD="xtts-v2-en"
just run-orch
just cli
```

**Switch model at runtime**

* Orchestrator calls `LoadModel(model_id)` on the target worker.
* Route next session to that `model_id`.
* Evictor unloads old models after `ttl_ms` if idle.

**Multi-GPU (same host)**

```bash
CUDA_VISIBLE_DEVICES=0 just run-tts-cosy
CUDA_VISIBLE_DEVICES=1 just run-tts-xtts
just run-orch
```

**Multi-host**

* Central Redis; workers announce `addr` with host:port reachable on LAN.

---

## 15) Acceptance criteria

* ✅ Realtime speech↔speech with **barge-in < 50 ms** pause latency.
* ✅ **Stream-TTS** mode from external LLM partial text.
* ✅ **Hot model swap** with **runtime load/unload** and **TTL eviction**.
* ✅ Single-GPU (two-process) & multi-GPU topologies.
* ✅ DX: `uv`, `ruff`, `mypy`, `pytest`, `just`, Docker, Compose.
* ✅ CI (`just ci`) passes on reference adapters & smoke tests.

---

If you want, I can next turn this doc into a **scaffolded repo** with stubbed adapters (Cosy/XTTS/Piper), the **ModelManager** (load/TTL/evict), the **gRPC service**, and the **justfile/docker** so you can run end-to-end immediately.
