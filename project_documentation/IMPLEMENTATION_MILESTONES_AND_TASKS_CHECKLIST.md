# 🧭 Realtime Duplex Voice Demo — Implementation Milestones & Task Checklists

*Last updated: Oct 2025*
*Linked PRD: `PRD.md`*
*Linked Design Doc: `TDD.md`*

---

## **M0 — Repo Scaffold & CI Skeleton**

**Goal:** Establish repo layout, dev tooling, and CI foundation.

### ✅ Tasks

* [x] Initialize repo with base `src/`, `tests/`, `configs/`, `docs/` directories.
* [x] Create `pyproject.toml` using **uv**, with dependencies only (no Python pin).
* [x] Add `ruff`, `mypy`, `pytest`, and `watchfiles` setup.
* [x] Add `justfile` with commands: `lint`, `typecheck`, `test`, `ci`, `run-*`.
* [x] Add `.github/workflows/ci.yml` for lint + type + test.
* [x] Create base `Dockerfile.orchestrator` and `Dockerfile.tts`.
* [x] `docker-compose.yml` for Redis + Orchestrator + TTS skeleton.
* [x] Add placeholder configs (`orchestrator.yaml`, `worker.yaml`).
* [x] Confirm `just ci` passes on empty stubs.

### 🧪 Validation

* [x] `uv lock` works and generates valid environment.
* [x] CI passes in GitHub Actions.
* [x] `just lint`, `just typecheck`, and `just test` succeed.

---

## **M1 — gRPC ABI + Mock Worker**

**Goal:** Define and generate TTS streaming interface and control messages.

### ✅ Tasks

* [x] Create `src/rpc/tts.proto`:

  * [x] `Synthesize(stream TextChunk) → stream AudioFrame`
  * [x] `Control`, `ListModels`, `LoadModel`, `UnloadModel`
* [x] Run codegen (`just gen-proto`).
* [x] Implement mock TTS worker with synthetic sine-wave output.
* [x] Implement orchestrator stub to connect to worker and log frames.

### 🧪 Validation

* [x] Integration test: orchestrator→worker→audio bytes verified.
* [x] Unit test: proto roundtrip, `Control` commands.
* [x] Mock worker responds to `PAUSE`, `RESUME`, `STOP`.

---

## **M2 — Orchestrator Transport + WS Fallback**

**Goal:** Add WebRTC (browser) and WS (CLI) ingress.

### ✅ Tasks

* [x] Integrate **LiveKit Agent SDK** in a way that can be swapped out with minimal effort for a different/custom orchestrator in the future if necessary.
* [x] Add CLI WS  client for local speech demo.
* [ ] Integrate https://github.com/livekit-examples/agent-starter-react front-end that uses self-hosted livekit infrastructure for local speech web demo
* [x] Implement `VAD` stub and message routing loop.
* [x] Add Redis service registration/discovery skeleton.

### 🧪 Validation

* [ ] WS echo test: text→worker→audio→back.
* [ ] WebRTC echo works locally.
* [ ] Redis keys reflect worker registration.

---

## **M3 — Barge-In & State Machine**

**Goal:** Implement pause/resume logic and state transitions.

### ✅ Tasks

* [ ] Add VAD detection using `webrtcvad` (20 ms frames).
* [ ] Implement orchestrator session FSM: `LISTENING → SPEAKING → BARGED_IN`.
* [ ] Send `PAUSE`/`RESUME` to worker on VAD transitions.
* [ ] Add test harness with recorded speech to verify timing.

### 🧪 Validation

* [ ] Barge-in latency p95 < 50 ms (local).
* [ ] Unit tests: FSM transitions.
* [ ] Manual test: voice playback halts on interruption.

---

## **M4 — Model Manager v1 (Default/Preload/TTL)**

**Goal:** Introduce model loading, warmup, and eviction system.

### ✅ Tasks

* [ ] Create `model_manager.py` with:

  * [ ] Default model load.
  * [ ] Optional preload list from CLI/config.
  * [ ] Per-model refcounts.
  * [ ] TTL and LRU eviction.
  * [ ] Warmup synth (~300 ms).
* [ ] Expose gRPC methods: `ListModels`, `LoadModel`, `UnloadModel`.
* [ ] Configurable params via YAML + CLI.

### 🧪 Validation

* [ ] Unit tests: load/unload lifecycle.
* [ ] Integration test: dynamic load + TTL eviction.
* [ ] VRAM usage stable under repeated model swaps.

---

## **M5 — Piper Adapter (CPU baseline)**

**Goal:** First real TTS adapter; enable end-to-end speech↔speech loop.

### ✅ Tasks

* [ ] Implement `adapter_piper.py` conforming to base `TTSAdapter`.
* [ ] Integrate ONNX runtime for CPU inference.
* [ ] Normalize loudness to target RMS.
* [ ] Return 20 ms PCM frames.
* [ ] Add to worker registry.

### 🧪 Validation

* [ ] Speech↔speech loop runs on CPU-only machine.
* [ ] Latency p95 < 500 ms.
* [ ] CLI/web demo plays coherent output.

---

## **M6 — CosyVoice 2 Adapter (GPU, streaming)**

**Goal:** GPU streaming TTS parity with Piper path.

### ✅ Tasks

* [ ] Implement `adapter_cosyvoice2.py`.
* [ ] Integrate model initialization and streaming inference.
* [ ] Warmup test phrase.
* [ ] Validate pause/resume fidelity.
* [ ] Add unit test for chunk pacing and frame size.

### 🧪 Validation

* [ ] FAL p95 < 300 ms (single GPU).
* [ ] Continuous playback verified under WebRTC.
* [ ] Barge-in stops/resumes smoothly.

---

## **M7 — XTTS-v2 Adapter + Voice Cloning**

**Goal:** Add expressive multi-speaker model with reference cloning.

### ✅ Tasks

* [ ] Implement `adapter_xtts_v2.py`.
* [ ] Support reference voice (6–10 s sample).
* [ ] Validate streaming mode and chunk pacing.
* [ ] Add metadata in `voicepacks/xtts-v2`.

### 🧪 Validation

* [ ] Quality vs latency benchmark recorded.
* [ ] Clone voice demo runs locally.
* [ ] Frame pacing p95 < 10 ms jitter.

---

## **M8 — Sesame / Unsloth Adapter (+LoRA)**

**Goal:** Support LoRA-fine-tuned Sesame model path.

### ✅ Tasks

* [ ] Implement `adapter_sesame.py` (vanilla).
* [ ] Implement `adapter_unsloth_sesame.py` (LoRA variant).
* [ ] Integrate `peft.PeftModel.from_pretrained`.
* [ ] Add CLI `--lora_path` arg for optional adapter.
* [ ] Add to preload list in config.

### 🧪 Validation

* [ ] LoRA swap works at runtime (no restart).
* [ ] Latency p95 < 350 ms (GPU).
* [ ] Model unload verified when idle.

---

## **M9 — Routing v1 (Capabilities + Load Balancing)**

**Goal:** Enable orchestrator to intelligently route sessions.

### ✅ Tasks

* [ ] Implement routing policies: least-busy, prefer-resident, random fallback.
* [ ] Redis-based worker heartbeat and capability registry.
* [ ] Add metrics: queue depth, avg latency per worker.

### 🧪 Validation

* [ ] Multiple workers registered, sessions distributed correctly.
* [ ] Load balancing confirmed via logs.
* [ ] Graceful worker death → reroute next request.

---

## **M10 — ASR Integration (Whisper small/distil)**

**Goal:** Add real-time transcription path in orchestrator.

### ✅ Tasks

* [ ] Integrate `openai-whisper` small or `whisperx` model.
* [ ] Stream transcriptions to LLM bridge (placeholder).
* [ ] Enable full speech↔speech pipeline (ASR→LLM→TTS).

### 🧪 Validation

* [ ] Roundtrip conversation demo works.
* [ ] Whisper CPU/GPU path verified.
* [ ] Logs show ASR partials feeding TTS text stream.

---

## **M11 — Observability & Profiling**

**Goal:** Enable measurement and diagnostics.

### ✅ Tasks

* [ ] Add structured logging (JSON).
* [ ] Add Prometheus counters: FAL, jitter, barge-in, RTF.
* [ ] Add `py-spy` targets in `justfile`.
* [ ] Integrate `torch.profiler` NVTX ranges for GPU tracing.

### 🧪 Validation

* [ ] Metrics available via HTTP scrape or logs.
* [ ] py-spy captures valid flamegraph.
* [ ] Nsight Systems/Compute traces export correctly.

---

## **M12 — Docker/Compose Smoke & Docs**

**Goal:** Fully reproducible single-GPU demo stack.

### ✅ Tasks

* [ ] Finalize Dockerfiles (CUDA 12.8 base).
* [ ] Ensure `docker compose up --build` spins up redis, tts0, orchestrator.
* [ ] Validate CLI & web client in container network.
* [ ] Add `README.md` setup, `docs/design_v2.1.md`.

### 🧪 Validation

* [ ] Smoke test: local run from clean checkout.
* [ ] End-to-end voice conversation works inside containers.
* [ ] Docs validated for correctness.

---

## **M13 — Multi-GPU & Multi-Host Scale-Out**

**Goal:** Expand deployment to N GPUs and hosts via Redis discovery.

### ✅ Tasks

* [ ] Implement per-GPU `CUDA_VISIBLE_DEVICES` orchestration.
* [ ] Extend registry for multi-host IP + port discovery.
* [ ] Add worker auto-registration heartbeat TTL.
* [ ] Validate cross-host gRPC communication.

### 🧪 Validation

* [ ] 2–4 GPUs handle simultaneous sessions.
* [ ] Multi-host LAN demo (Redis central) works.
* [ ] No cross-worker model load contention.

---

## **Final Acceptance Gate**

✅ All milestones integrated and validated:

* [ ] End-to-end speech↔speech with barge-in under 50 ms p95.
* [ ] Runtime model switching with TTL unload.
* [ ] Single-GPU + Multi-GPU working demos.
* [ ] CI green; docker compose clean build.
* [ ] Profiling data and metrics available.
* [ ] Docs finalized.

---

**Filename suggestion:** `MILESTONES.md`
You can import each milestone section as a GitHub Issue or Milestone to track progress with checklists.
