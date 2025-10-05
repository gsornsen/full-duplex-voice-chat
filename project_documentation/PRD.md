# Realtime Duplex Voice Demo — Product Requirements Document (PRD)

*Last updated: Oct 2025*
*Owner: Gerald Sornsen*

---

## 1. Summary

Build a production-style **realtime speech↔speech** demo that supports **low-latency streaming TTS**, **barge-in**, and **hot-swapping across multiple open TTS models** (e.g., Sesame/Unsloth with LoRA, CosyVoice 2, XTTS-v2, Piper, etc.). The system should run on **single-GPU** and **multi-GPU** (same host and multi-host LAN) setups and provide a clean developer experience (DX) using `uv`, `ruff`, `mypy`, `pytest`, `just`, **Docker**, and **Docker Compose**.

The orchestrator layer leverages an existing realtime framework (e.g., LiveKit Agents) for WebRTC transport, VAD/interruptions, and sessioning, while **TTS workers** follow a **uniform streaming ABI**. A **Model Manager** loads a **default model to VRAM** at startup, supports **optional preloading**, and can **load/unload models dynamically** with **TTL-based eviction** when idle.

---

## 2. Goals & Non-Goals

### 2.1 Goals

* **Realtime duplex conversation** with **barge-in** (pause/resume in < 50 ms).
* **Streaming TTS** with 20 ms, 48 kHz PCM frames; jitter-tolerant playback.
* **Model modularity**: swap among Sesame/Unsloth (+LoRA), CosyVoice 2, XTTS-v2, Spark-TTS, Parler-TTS, MeloTTS, Piper via a shared streaming ABI.
* **Lifecycle management**: default model preloaded and warmed; optional preload list; runtime **Load/Unload**; **TTL**-based eviction; **resident model preference**.
* **Scale**: single-GPU (two-process), multi-GPU (same host), multi-host (LAN).
* **DX**: modern Python tooling (`uv`, `ruff`, `mypy`), `justfile` for tasks, Dockerized bring-up, clear repo structure, CI sanity tests.
* **Observability**: metrics (FAL, jitter, RTF, queue depth, barge-in counts), structured logs, and profiling hooks.

### 2.2 Non-Goals

* High-accuracy ASR research (we’ll use Whisper small/distil).
* WAN-grade auth, billing, quotas (demo scope only; simple API key).
* Telephony features beyond basic examples (SIP trunks optional later).
* Proprietary cloud model integrations (focus on open(-ish) models first).

---

## 3. Target Users & Personas

* **Applied Researcher (Alice)**: wants to compare emotional prosody & latency across models using the same transport loop.
* **ML Engineer (Ben)**: needs to test LoRA variants for a given base model and validate memory/latency trade-offs.
* **Solutions Prototyper (Cara)**: needs a stable demo to show real-time voice conversational UX in a browser with barge-in.
* **Infra Engineer (Dev)**: validates multi-GPU/multi-host deployment and observes system metrics under load.

---

## 4. Key Use Cases (User Stories)

1. **Speech↔Speech conversation**
   *As a user, I speak to the browser client and hear a low-latency reply, with the ability to interrupt the reply mid-utterance.*
2. **Stream-TTS** (text-in, audio-out)
   *As a developer, I stream partial tokens from an external LLM to the orchestrator and get live audio from the TTS worker.*
3. **Switch model at runtime**
   *As a user, I choose a different TTS model/voice; the system loads it on-demand and uses it for subsequent sessions.*
4. **LoRA variant**
   *As an ML engineer, I test a LoRA fine-tuned voice for Sesame (Unsloth path) without restarting services.*
5. **Edge CPU path**
   *As a prototyper, I run a CPU-only model (Piper) and still get realtime playback with barge-in enabled.*
6. **Multi-GPU scale-out**
   *As an infra engineer, I add workers on additional GPUs/hosts and see traffic routed to the least-busy compatible worker.*

---

## 5. Scope

### 5.1 In Scope

* LiveKit-based orchestrator with WebRTC + WS fallback, VAD, sessioning, barge-in.
* gRPC streaming ABI between orchestrator and TTS workers.
* **Model Manager** with default preload, optional preload list, warmup, dynamic load/unload, TTL eviction, LRU bias.
* Adapters for **CosyVoice 2**, **XTTS-v2**, **Piper** (MVP), followed by **Sesame**/**Unsloth+LoRA** (post-MVP).
* CLI + minimal web client.
* Docker/Compose bring-up, `just` recipes, CI (ruff/mypy/pytest).
* Metrics/logging/profiling scaffolding.

### 5.2 Out of Scope

* Production authN/Z, rate-limiting, and billing.
* Cross-cloud autoscaling and multi-region failover.
* Comprehensive multilingual ASR/TTS benchmarking.

---

## 6. Functional Requirements

### 6.1 Orchestrator

* **Transport**: WebRTC (browser) with WS fallback for CLI.
* **ASR**: Whisper small/distil; GPU if available, CPU fallback.
* **VAD & barge-in**:

  * Send `PAUSE` to the active worker within **< 50 ms** of speech detection.
  * Resume with `RESUME` upon silence.
* **Sessioning**: track per-conversation state and model selection.
* **Routing**:

  * Discover workers via Redis.
  * Select worker based on capabilities, language, **resident model preference**, and load (queue depth, p50 latency).
  * Optionally trigger `LoadModel(model_id)` and route next session when available.

### 6.2 TTS Workers & Adapters

* Implement gRPC ABI:

  * `StartSession`, `Synthesize(stream<TextChunk>) → stream<AudioFrame>`, `Control(PAUSE|RESUME|STOP|RELOAD)`, `EndSession`.
  * Lifecycle: `ListModels`, `LoadModel`, `UnloadModel`, `GetCapabilities`.
* **Audio streaming**: 20 ms frames, 48 kHz mono PCM (Opus optional later).
* **Model Manager**:

  * Load **default model** at startup; **optional preload list**; warmup each loaded model (~300 ms).
  * Dynamic load on demand; reference counting to block eviction while in use.
  * TTL-based eviction of idle models; **resident cap**; LRU bias.
  * Configurable via YAML and CLI flags.
* **Adapters**:

  * Conform to shared interface; repacketize internal chunk sizes to 20 ms frames.
  * Normalize loudness (~−16 LUFS target or RMS).
  * Respect `PAUSE/RESUME/STOP` immediately.

### 6.3 Voice Packs

* On-disk structure per model family with `model.safetensors | onnx/`, `config.json`, `metadata.yaml`, optional `ref/seed.wav`.
* Hot reload on `RELOAD` command or worker restart.

---

## 7. Non-Functional Requirements

* **Latency**:

  * **Barge-in pause latency**: p95 **< 50 ms** from VAD edge to last emitted frame.
  * **First Audio Latency (FAL)**: p95 **< 300 ms** for GPU adapters; p95 **< 500 ms** for Piper CPU.
* **Jitter**: frame pacing p95 **< 10 ms** under 3 concurrent sessions per worker.
* **Reliability**: worker crash does not kill orchestrator sessions; auto-reconnect and reroute.
* **Scalability**: run N workers across GPUs/hosts; route by capability + load.
* **Security**: API key (demo), TLS termination via reverse proxy option, audio retention off by default.
* **DX**: `uv` for resolution/lock, `ruff` clean, `mypy` strict (except 3p libs), `pytest` integration tests, `just` one-liners, Dockerized.

---

## 8. UX / Flows

### 8.1 Speech↔Speech (browser)

1. User clicks “Connect” → WebRTC session established.
2. User speaks → ASR produces text; LLM (optional) streams partial tokens.
3. Orchestrator forwards partial text to the chosen TTS worker.
4. Worker streams **20 ms frames** → browser plays; barge-in halts playback immediately.

### 8.2 Model Switch

1. User selects a new model in UI or via CLI param.
2. Orchestrator requests `LoadModel(model_id)` on target worker.
3. Current session continues with resident fallback; next session uses loaded model.
4. Evictor unloads idle models after TTL.

---

## 9. Configuration

### 9.1 Worker `configs/worker.yaml` (example)

```yaml
worker:
  device: "cuda:0"
  sample_rate: 48000
  frame_ms: 20
  adapter: "cosyvoice2"

model_manager:
  default_model_id: "cosyvoice2-en-base"
  preload_model_ids: []
  ttl_ms: 600000            # 10 min
  min_residency_ms: 120000  # 2 min
  evict_check_interval_ms: 30000
  resident_cap: 3
  max_parallel_loads: 1
```

### 9.2 CLI flags (worker)

```shell
--adapter cosyvoice2
--default_model cosyvoice2-en-base
--preload_models xtts-v2-en,sesame-unsloth-lora01
--ttl_ms 600000
--resident_cap 3
--max_parallel_loads 1
```

### 9.3 Orchestrator `configs/orchestrator.yaml`

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

---

## 10. Telemetry & Metrics

* **Core**: first-audio-latency (FAL), frame jitter p95, real-time factor (RTF), queue depth, barge-in events, active sessions, model load/unload durations, eviction counts.
* **Logs**: structured JSON with session IDs; redacted PII/audio paths.
* **Profiling**: py-spy (CPU), PyTorch Profiler with NVTX ranges (GPU), Nsight Systems/Compute one-liners.
* **Dashboards**: simple Grafana (optional) or log-based summaries.

---

## 11. Success Metrics (KPIs)

* **Latency SLAs met**: barge-in p95 < 50 ms; FAL p95 < 300 ms (GPU), < 500 ms (CPU).
* **Uptime**: > 99% during 8-hour demos.
* **Model swap time**: load+warmup p95 < 5 s for GPU models (hardware-dependent); no session drop.
* **Developer onboarding**: new dev reaches working single-GPU demo in **< 30 min** following docs.
* **CI health**: `just ci` green on main; integration smoke passes.

---

## 12. Dependencies

* **LiveKit** (Agents SDK) or equivalent realtime framework.
* **Redis** for worker registry.
* **PyTorch 2.7 + CUDA 12.8** (stable wheels), or CPU ONNX for Piper.
* **Docker Engine 28.x**; NVIDIA container runtime for GPU workers.

---

## 13. Risks & Mitigations

| Risk                                   | Impact               | Mitigation                                                    |
| -------------------------------------- | -------------------- | ------------------------------------------------------------- |
| CUDA/PyTorch version drift             | Crashes or poor perf | Pin to CUDA 12.8 + PyTorch 2.7; use `uv.lock`                 |
| Barge-in jitter under load             | UX regression        | Pre-warmed vocoder, tight pacing buffers, separate processes  |
| VRAM thrash on frequent model switches | Latency spikes       | `resident_cap`, TTL, min residency, load semaphore            |
| Adapter API inconsistencies            | Integration bugs     | Contract tests (framing, control, FAL), shared base utilities |
| Transport complexity (NAT/WebRTC)      | Connectivity issues  | WS fallback & CLI; provide TURN guidance in docs              |
| CPU-only environments                  | Latency regressions  | Piper path; set expectations in docs                          |

---

## 14. Release Plan & Milestones (high-level)

1. **M0**: Repo scaffold + CI skeleton
2. **M1**: gRPC ABI + Mock worker (stream + control)
3. **M2**: Orchestrator transport + WS fallback
4. **M3**: Barge-in end-to-end (mock)
5. **M4**: Model Manager v1 (default/preload/TTL)
6. **M5**: Piper adapter (CPU)
7. **M6**: CosyVoice 2 adapter (GPU) + loudness norm
8. **M7**: XTTS-v2 adapter (GPU) + cloning
9. **M8**: Sesame / Unsloth (+LoRA) adapter
10. **M9**: Routing v1 (capabilities + prefer resident)
11. **M10**: ASR in orchestrator; full speech↔speech
12. **M11**: Observability & profiling
13. **M12**: Docker/Compose smoke; docs polish
14. **M13**: Multi-GPU & multi-host scale-out

Each milestone ships with tests, acceptance checks, and updated docs.

---

## 15. Acceptance Criteria (MVP)

* Browser demo: user speaks; system replies with **barge-in** working; **CosyVoice 2** and **Piper** selectable; **model switch** persists for next session.
* Latency targets met: barge-in < 50 ms (p95); FAL < 300 ms (GPU) / < 500 ms (CPU).
* `just` commands work locally; `docker-compose up --build` provides a running single-GPU demo.
* Model Manager loads **default model**, honors **preload list**, performs **TTL eviction** with no mid-stream unloads.
* CI (`just ci`) passes (ruff/mypy/pytest); minimal integration smoke test green.

---

## 16. Appendix

### 16.1 Minimal API (gRPC) — summary

* **Synthesize**: `stream<TextChunk> → stream<AudioFrame>`
* **Control**: `PAUSE | RESUME | STOP | RELOAD`
* **Lifecycle**: `ListModels`, `LoadModel`, `UnloadModel`, `GetCapabilities`, `StartSession`, `EndSession`

### 16.2 Audio Framing

* Output: **20 ms**, **48 kHz**, mono PCM frames (Opus optional later).
* Internal adapters may step at 80–160 ms; must repacketize to 20 ms.

### 16.3 Repo Tree (abridged)

```shell
realtime-duplex-demo/
├─ pyproject.toml
├─ uv.lock
├─ justfile
├─ docker-compose.yml
├─ Dockerfile.orchestrator
├─ Dockerfile.tts
├─ .env.example
├─ README.md
├─ docs/
│  ├─ design_v2.1.md
│  └─ profiling.md
├─ configs/
│  ├─ orchestrator.yaml
│  └─ worker.yaml
├─ voicepacks/
│  ├─ cosyvoice2/en-base/...
│  └─ xtts-v2/en-demo/...
├─ src/
│  ├─ orchestrator/ (server.py, vad.py, asr.py, routing.py, registry.py, config.py)
│  ├─ tts/
│  │  ├─ worker.py, model_manager.py, tts_base.py
│  │  ├─ adapters/ (adapter_*.py for sesame, unsloth, cosyvoice2, xtts, piper, ...)
│  │  ├─ audio/ (framing.py, loudness.py)
│  │  └─ utils/ (logging.py, timers.py)
│  ├─ rpc/ (tts.proto, generated/)
│  └─ client/ (cli_client.py, web/)
├─ tests/
│  ├─ unit/ (vad, routing, tts_control, model_manager, audio_framing)
│  └─ integration/ (loopback_ws, barge_in, preload_defaults)
└─ .github/workflows/ci.yml
```

---

*This PRD defines the scope, outcomes, and quality bars for the Realtime Duplex Voice Demo. It pairs with the detailed technical design and milestone implementation plan to guide day-to-day execution.*
