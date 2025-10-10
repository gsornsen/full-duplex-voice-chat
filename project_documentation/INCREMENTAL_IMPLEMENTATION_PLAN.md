# Realtime Duplex Voice Demo — Incremental Implementation Plan

*Last updated: 2025-10-09*
*Owner: Gerald Sornsen*

---

## Milestone 0 — Repo scaffold & CI skeleton

**Status**: ✅ Complete (2025-09)
**Goal:** Stand up the repo with tooling, no runtime yet.

**Scope**

* Create folder structure (exactly as in design).
* `pyproject.toml`, `justfile`, baseline `README.md`, `.github/workflows/ci.yml`.
* Empty/stub modules; compile the gRPC stubs from `src/rpc/tts.proto`.

**Deliverables**

* Tree structure present.
* `just gen-proto` works.
* `just ci` runs ruff, mypy (allow stubs), pytest (0 tests ok).

**Tests / Validation**

* CI green on PR.
* `uv lock` creates `uv.lock`.

**Exit criteria**

* CI passes with stubs.
* Dev can run `just` tasks locally without errors.

**Risks / Mitigation**

* Dependency drift → lock with `uv.lock` early.

---

## Milestone 1 — Core gRPC ABI + Mock TTS Worker

**Status**: ✅ Complete (2025-09)
**Goal:** Nail the streaming contract & control plane before models.

**Scope**

* Implement `src/tts/worker.py` with a **MockAdapter** that emits a 1 kHz tone or synthetic PCM sentences at 20 ms/48 kHz.
* Implement `StartSession/Synthesize/Control/EndSession`.
* Implement `AudioFrame` pacing and `PAUSE/RESUME/STOP` semantics.
* Simple `model_manager.py` stub returning a single "mock" model as resident.

**Deliverables**

* Runnable worker: `uv run python -m src.tts.worker --adapter mock --port 7001`.

**Tests / Validation**

* Unit: `test_tts_control.py` (pause within <50 ms), `test_audio_framing.py` (exact 20 ms cadence, 48 kHz).
* Integration: local client feeds `TextChunk` stream; receives PCM; verifies timing.

**Exit criteria**

* FAL (first-audio latency) for mock <100 ms.
* PAUSE stops new frames within 50 ms (measured).

**Risks / Mitigation**

* Jitter in async loop → use monotonic clock & pacing buffer.

---

## Milestone 2 — Orchestrator (LiveKit agent) + WS fallback, no ASR yet

**Status**: ✅ Complete - Enhanced (2025-10)
**Goal:** Realtime transport & session loop to the mock worker.

**Scope**

* `src/orchestrator/server.py`: LiveKit session handling, WebRTC I/O, WS fallback for CLI.
* Minimal routing to a single worker (static addr).
* Basic VAD wiring (webrtcvad), but barge-in can be stubbed.

**Deliverables**

* `just run-orch` starts LiveKit agent server; `just cli` connects via WS and hears mock audio from text.

**Tests / Validation**

* Integration: loopback WS test (`test_loopback_ws.py`) verifies 20 ms frames, FAL is stable.
* Manual browser test with sample text box.

**Exit criteria**

* End-to-end text→audio via orchestrator with <200 ms added overhead.

**Risks / Mitigation**

* Media threading complexity → keep data channel for text, do audio only from worker.

**Notes**

* Exceeded original scope: LiveKit implemented as PRIMARY transport with full WebRTC infrastructure (Caddy reverse proxy, TLS, Docker compose)
* Foundation ready for future scale-out and production deployment

---

## Milestone 3 — Barge-in (transport→worker) + Hard Stop Semantics

**Status**: ✅ Complete (2025-10-09)
**Goal:** Real barge-in: PAUSE/RESUME over control RPC with state machine.

**Scope**

* Implement LISTENING/SPEAKING/BARGED_IN transitions.
* Integrate VAD using webrtcvad (20ms frames).
* Orchestrator sends PAUSE on VAD speech start; RESUME on VAD stop.
* Worker honors control immediately (mock adapter already does).

**Deliverables**

* Barge-in demo where user speech halts playback mid-utterance.
* VAD processor with configurable aggressiveness and debouncing.
* Audio resampling pipeline (48kHz → 16kHz) for VAD processing.

**Tests / Validation**

* Unit: 29/29 VAD tests passing (`test_vad.py`)
  - Configuration validation
  - Speech/silence detection
  - Event callbacks
  - Debouncing logic
  - Audio resampling
* Integration: 8/8 VAD integration tests passing (`test_vad_integration.py`)
  - Speech detection validation
  - Aggressiveness levels
  - Processing latency measurement
  - Multiple speech segments
  - Real audio characteristics

**Exit criteria**

* 95th percentile pause latency <50 ms ✅ Validated
* VAD processing latency <5ms per frame ✅ Validated
* All tests passing (37/37) ✅ Complete

**Risks / Mitigation**

* Race conditions → centralize control handling with atomic state in worker.

**Implementation Notes**

* VAD implementation: `src/orchestrator/vad.py`
* Audio resampler: `src/orchestrator/audio/resampler.py`
* Configuration support: `src/orchestrator/config.py` (VADConfig)
* Test coverage: comprehensive unit and integration tests
* Performance validated: <50ms barge-in latency, <5ms VAD processing

**Completion Date**: 2025-10-09

---

## Milestone 4 — Model Manager v1: default+preload, warmup, TTL eviction (no real models yet)

**Status**: 📝 Planned
**Goal:** Implement the **real** lifecycle manager against the mock adapter.

**Scope**

* Full `model_manager.py`: resident map, TTL eviction loop, min residency, resident cap, LRU.
* CLI/config flags: `--default_model`, `--preload_models`, `--ttl_ms`, etc.
* gRPC: `ListModels`, `LoadModel`, `UnloadModel`.

**Deliverables**

* Worker loads default model, optionally preloads others, warms each (~300 ms utterance).
* Evicts idle models after TTL.

**Tests / Validation**

* Unit: `test_model_manager.py` for load/unload, TTL, LRU, min residency.
* Integration: orchestrator issues `LoadModel`, switches sessions to new model, old model evicted after TTL.

**Exit criteria**

* Deterministic eviction & zero mid-stream unloads (refcounted).

**Risks / Mitigation**

* VRAM fragmentation (later with real models) → serialize loads with semaphore.

---

## Milestone 5 — First real adapter: **Piper (CPU)** (edge baseline)

**Status**: 📝 Planned
**Goal:** Bring a real TTS with trivial deps to prove adapter surface.

**Scope**

* `adapter_piper.py` with ONNX path; resample to 48 kHz if needed; 20 ms framing.
* Metadata & voicepack discovery for Piper voices.

**Deliverables**

* Worker can run Piper on CPU; orchestrator routes to it.

**Tests / Validation**

* MOS-lite: confirm intelligibility and cadence stability.
* FAL target: <250 ms on a modern CPU; RTF <1 for short utterances.

**Exit criteria**

* Integration tests pass with Piper as default; barge-in still <50 ms.

**Risks / Mitigation**

* ONNX perf variability → cache graph sessions; warmup.

---

## Milestone 6 — **CosyVoice 2** adapter (GPU) + normalization

**Status**: 📝 Planned
**Goal:** A high-quality expressive model with streaming.

**Scope**

* `adapter_cosyvoice2.py` using PyTorch.
* Optional style parameters; strict repacketization to 20 ms.
* Add loudness normalization (`loudness.py`) to ~−16 LUFS.

**Deliverables**

* CosyVoice runs on single GPU; selectable via `model_id`.

**Tests / Validation**

* FAL target: ~150–250 ms on 4090/4080-class GPUs.
* Subjective prosody check on 5 gold prompts.
* Barriers: queue depth fairness under concurrent sessions.

**Exit criteria**

* FAL p95 <300 ms; jitter p95 <10 ms under 3 concurrent sessions.

**Risks / Mitigation**

* Large chunk bursts → internal buffer + repacketizer.

---

## Milestone 7 — **XTTS-v2** adapter (GPU) + reference voice cloning

**Status**: 📝 Planned
**Goal:** Another high-quality option + cloning path.

**Scope**

* `adapter_xtts.py` with embedding cache for reference clips (6–10 s).
* Configurable inference speed/quality.
* Voicepack for a demo voice and a cloning example.

**Deliverables**

* End-to-end with XTTS; switching between Cosy/XTTS confirmed.

**Tests / Validation**

* FAL, jitter similar to Cosy; cloning latency measured.
* Barge-in verified across both adapters.

**Exit criteria**

* Adapter parity with Cosy on API & control behavior.

**Risks / Mitigation**

* Embedding cache staleness → invalidate on model reload.

---

## Milestone 8 — **Sesame / Unsloth-Sesame** adapter (+LoRA)

**Status**: 📝 Planned
**Goal:** Your original target model + LoRA path.

**Scope**

* `adapter_sesame.py` (direct) and/or wrapper for community "OpenAI-compatible" CSM server.
* `adapter_unsloth_sesame.py` with `peft.PeftModel.from_pretrained`.
* Voice pack format for LoRA adapters.

**Deliverables**

* LoRA-enabled model switching; warmup correctness.

**Tests / Validation**

* LoRA load/unload timing; VRAM change tracked.
* Quality: compare 5 gold prompts—tone/expressiveness vs Cosy/XTTS.

**Exit criteria**

* Clean hot-swap between Sesame base and LoRA variants without process restart.

**Risks / Mitigation**

* PEFT adapter memory spikes → serialize loads; pre-check VRAM headroom.

---

## Milestone 9 — **Routing v1**: capability-aware + prefer resident

**Status**: 📝 Planned
**Goal:** Smart selection without manual targeting.

**Scope**

* Redis registry reports worker caps: `expressive`, `cpu_ok`, `lora`, latency p50/queue depth.
* Policy YAML honored (`default`, `edge`, `lora`, fallbacks).
* Prefer **resident** models; async `LoadModel` for requested-but-absent, route current utterance to best-resident now.

**Deliverables**

* `routing.py` with clear selection logic and tie-breakers.

**Tests / Validation**

* Unit: preference ordering; resident bias; proper fallback when requested is unavailable.
* Integration: model switch scenario where next session lands on new model after it finishes loading.

**Exit criteria**

* Misroute rate 0 in tests; no dead-ends when workers are busy.

**Risks / Mitigation**

* Thundering herd `LoadModel` → orchestrator-side de-dup of inflight loads per model_id.

---

## Milestone 10 — **ASR on Orchestrator** + full barge-in loop

**Status**: 📝 Planned
**Goal:** True speech↔speech.

**Scope**

* `asr.py` with Whisper small/distil (GPU if available, else CPU).
* Mic → ASR text → (optional LLM bridge) → TTS.
* Token stream path for LLM (can stub with "echo" stream).

**Deliverables**

* Browser demo: speak, system replies, interrupt mid-reply.

**Tests / Validation**

* End-to-end latency budget measured (ASR + TTS).
* Barge-in remains <50 ms across adapters.

**Exit criteria**

* Conversational demo works smoothly on single GPU.

**Risks / Mitigation**

* ASR-TTS co-scheduling → keep separate processes (already done).

---

## Milestone 11 — **Observability & Profiling**

**Status**: 📝 Planned
**Goal:** Make performance measurable and fixable.

**Scope**

* Prometheus counters: FAL, RTF, jitter p95, queue depth, barge-in count.
* Structured logs with session ids.
* PyTorch Profiler (NVTX ranges) and Nsight Systems/Compute just targets.

**Deliverables**

* `docs/profiling.md`, dashboards or simple scrapes.

**Tests / Validation**

* Load test (N=5 concurrent sessions) produces stable jitter and barge-in.

**Exit criteria**

* Metrics exported; profiling runs produce usable traces.

**Risks / Mitigation**

* Overhead → sample/enable profiling only on demand.

---

## Milestone 12 — **Containers & Compose** + Smoke Tests

**Status**: 📝 Planned
**Goal:** Reproducible env.

**Scope**

* Finalize Dockerfiles (orchestrator, tts).
* `docker-compose up --build` path for single-GPU.
* Health checks; small smoke script.

**Deliverables**

* One-liner bring-up.

**Tests / Validation**

* CLI client connects, plays audio; barge-in works.

**Exit criteria**

* Green smoke suite in CI (can be a nightly job with a GPU runner).

**Risks / Mitigation**

* GPU in CI → use self-hosted runner or skip GPU integration tests by tag.

---

## Milestone 13 — **Multi-GPU** & **Multi-Host (LAN)**

**Status**: 📝 Planned
**Goal:** Scale-out.

**Scope**

* Spawn `tts@i` with `CUDA_VISIBLE_DEVICES=i`.
* Redis service registry across hosts; orchestrator routes to least-busy/affinity.

**Deliverables**

* Demo: 2 GPUs, 3 workers; balanced routing.

**Tests / Validation**

* Synthetic test: queue depth balancing; no head-of-line blocking.

**Exit criteria**

* p95 FAL and jitter remain within single-GPU envelopes ±10%.

**Risks / Mitigation**

* Clock skew on multi-host → rely on monotonic local clocks, not NTP, for pacing.

---

## Milestone 14 — **Docs, Runbooks, and Hardening**

**Status**: 📝 Planned
**Goal:** Production-ready demo polish.

**Scope**

* `docs/design_v2.1.md` (final), `docs/runbooks.md` (single-GPU, multi-GPU, multi-host).
* TLS/WSS reverse proxy examples (nginx/traefik).
* Security pass: API key, log redaction, retention switches.

**Deliverables**

* Clear "how to run" and "how to extend with a new model" guides.
* Example voicepacks.

**Tests / Validation**

* New engineer can follow docs and get a working single-GPU demo in <30 minutes.

**Exit criteria**

* Docs PR approved; onboarding test successful.

---

# Cross-cutting Quality Gates (apply continuously)

* **Coding standards:** ruff clean, mypy strict (allow adapter libs as `ignore_missing_imports` where needed).
* **Streaming cadence:** 20 ms @ 48 kHz, validated by `test_audio_framing.py`.
* **Barge-in latency:** p95 <50 ms on every merge that touches orchestrator or adapters. ✅ Validated (M3)
* **FAL budget:** p95 <300 ms for GPU adapters; <500 ms for Piper CPU (targets can be tuned by hardware).
* **No deadlocks:** soak test with N=10 minute run; watchdog alerts.

---

# Rollout & Feature Flags

* `FEATURE_WS_FALLBACK`, `FEATURE_ASR`, `FEATURE_LLM_BRIDGE`, `FEATURE_OPUS` (codec), `FEATURE_MODEL_AUTOPRELOAD`.
* Toggle features in `configs/*` and env vars; keep core loop usable even if optional features are off.

---

# Resourcing & Sequencing Hints

* **One engineer** can do M0–M4 in a week with focus (mock path first).
* Add adapters in parallel (Piper vs Cosy/XTTS vs Sesame/LoRA).
* Keep **adapter bring-up** predictable: start from the **mock adapter harness** + **framing utils** + **loudness**.

---

## "Go/No-Go" Gate Suggestions

* **Gate A (after M3):** Real-time loop + barge-in verified on mock → proceed to real models. ✅ PASSED
* **Gate B (after M6):** One high-quality GPU adapter (Cosy) meets latency/jitter SLAs → proceed to multiple adapters.
* **Gate C (after M9):** Routing stable under load, resident preference works → enable dynamic model loading in demos.
* **Gate D (after M12):** Dockerized smoke green → demos okay to share with stakeholders.

---
