# Configuration Reference

## Overview

The system uses YAML configuration files for both the orchestrator and TTS workers. This document provides complete reference for all configuration options, their valid ranges, defaults, and usage.

**Configuration Files:**
- `configs/orchestrator.yaml` - Orchestrator settings (transport, VAD, routing)
- `configs/worker.yaml` - TTS worker settings (model manager, capabilities)

**Environment Variables:**
Configuration values can be overridden using environment variables (see [Environment Overrides](#environment-overrides)).

---

## Orchestrator Configuration

**File:** `configs/orchestrator.yaml`

### Complete Example

```yaml
# Orchestrator Configuration
# M2: WebSocket + LiveKit transport, VAD stub, static routing

transport:
  websocket:
    enabled: true
    host: "0.0.0.0"
    port: 8080
    max_connections: 100
    frame_queue_size: 50

  livekit:
    enabled: true
    url: "http://localhost:7880"
    api_key: "devkey"
    api_secret: "secret"
    room_prefix: "tts-session"

redis:
  url: "redis://localhost:6379"
  db: 0
  worker_key_prefix: "worker:"
  worker_ttl_seconds: 30
  connection_pool_size: 10

routing:
  static_worker_addr: "grpc://localhost:7001"
  prefer_resident_models: true
  load_balance_strategy: "queue_depth"

vad:
  enabled: true
  aggressiveness: 2
  sample_rate: 16000
  frame_duration_ms: 20

log_level: "INFO"
graceful_shutdown_timeout_s: 10
```

---

### transport.websocket

WebSocket transport configuration for browser and CLI clients.

#### `transport.websocket.enabled`

**Type:** `boolean`
**Default:** `true`
**Valid Values:** `true`, `false`

Enable WebSocket transport.

**M2 Status:** ✅ Fully implemented

**Example:**
```yaml
transport:
  websocket:
    enabled: true
```

#### `transport.websocket.host`

**Type:** `string`
**Default:** `"0.0.0.0"`
**Valid Values:** Any valid IP address or hostname

Host address to bind WebSocket server.

**Common Values:**
- `"0.0.0.0"` - All interfaces (recommended for containers)
- `"127.0.0.1"` - Localhost only (secure development)
- `"192.168.1.100"` - Specific interface

**Example:**
```yaml
transport:
  websocket:
    host: "0.0.0.0"
```

#### `transport.websocket.port`

**Type:** `integer`
**Default:** `8080`
**Valid Range:** `1024-65535`

Port for WebSocket server.

**Recommendations:**
- Use `8080` for HTTP-like services
- Avoid ports < 1024 (require root privileges)
- Check for conflicts: `netstat -tuln | grep <port>`

**Example:**
```yaml
transport:
  websocket:
    port: 8080
```

#### `transport.websocket.max_connections`

**Type:** `integer`
**Default:** `100`
**Valid Range:** `>= 1`

Maximum concurrent WebSocket connections.

**Guidelines:**
- Development: `10-50`
- Production (single orchestrator): `100-500`
- Production (load balanced): `500-1000` per instance

**Resource Impact:**
- Each connection: ~1-2 MB memory
- CPU: ~1% per active session (VAD + routing)

**Example:**
```yaml
transport:
  websocket:
    max_connections: 100
```

#### `transport.websocket.frame_queue_size`

**Type:** `integer`
**Default:** `50`
**Valid Range:** `>= 10`

Audio frame buffer size per connection (number of 20ms frames).

**Buffer Duration:** `frame_queue_size × 20ms`
- `50` frames = 1000ms (1 second) buffer
- `100` frames = 2000ms (2 seconds) buffer

**Tuning:**
- **Smaller (10-30):** Lower latency, risk of frame drops
- **Larger (100-200):** Higher latency, more resilient to jitter

**Example:**
```yaml
transport:
  websocket:
    frame_queue_size: 50  # 1 second buffer
```

---

### transport.livekit

LiveKit transport configuration for WebRTC support (optional).

**M2 Status:** ⚠️ Optional - system works without it

#### `transport.livekit.enabled`

**Type:** `boolean`
**Default:** `true`
**Valid Values:** `true`, `false`

Enable LiveKit transport.

**Note:** M2 implementation is partial. WebSocket is the primary transport.

**Example:**
```yaml
transport:
  livekit:
    enabled: true
```

#### `transport.livekit.url`

**Type:** `string`
**Default:** `"http://localhost:7880"`
**Valid Values:** Any valid HTTP/HTTPS URL

LiveKit server URL.

**Example:**
```yaml
transport:
  livekit:
    url: "http://localhost:7880"
```

#### `transport.livekit.api_key`

**Type:** `string`
**Default:** `"devkey"`
**Valid Values:** Any string

LiveKit API key for authentication.

**Security:** Use environment variables for production credentials.

**Example:**
```yaml
transport:
  livekit:
    api_key: "${LIVEKIT_API_KEY}"
```

#### `transport.livekit.api_secret`

**Type:** `string`
**Default:** `"secret"`
**Valid Values:** Any string

LiveKit API secret for authentication.

**Security:** Never commit secrets to version control.

**Example:**
```yaml
transport:
  livekit:
    api_secret: "${LIVEKIT_API_SECRET}"
```

#### `transport.livekit.room_prefix`

**Type:** `string`
**Default:** `"tts-session"`
**Valid Values:** Any valid room name prefix

Prefix for LiveKit room names.

**Example:**
```yaml
transport:
  livekit:
    room_prefix: "tts-session"
```

---

### redis

Redis configuration for worker discovery and service registry.

#### `redis.url`

**Type:** `string`
**Default:** `"redis://localhost:6379"`
**Valid Values:** Redis connection URL

Redis server connection string.

**URL Formats:**
- `redis://host:port` - Unencrypted
- `redis://host:port/db` - Specific database
- `redis://:password@host:port` - With authentication
- `rediss://host:port` - TLS encrypted

**Examples:**
```yaml
# Local development
redis:
  url: "redis://localhost:6379"

# Remote with auth
redis:
  url: "redis://:mypassword@redis-server:6379"

# Docker Compose service name
redis:
  url: "redis://redis:6379"
```

#### `redis.db`

**Type:** `integer`
**Default:** `0`
**Valid Range:** `0-15`

Redis database number.

**Best Practice:** Use different databases for dev/staging/production if sharing Redis instance.

**Example:**
```yaml
redis:
  db: 0  # Development
  # db: 1  # Staging
  # db: 2  # Production
```

#### `redis.worker_key_prefix`

**Type:** `string`
**Default:** `"worker:"`
**Valid Values:** Any string

Key prefix for worker registry entries.

**Purpose:** Namespace isolation for worker keys.

**Example:**
```yaml
redis:
  worker_key_prefix: "worker:"
  # Keys will be: worker:tts-worker-0, worker:tts-worker-1, etc.
```

#### `redis.worker_ttl_seconds`

**Type:** `integer`
**Default:** `30`
**Valid Range:** `>= 5`

Worker registration TTL (time-to-live) in seconds.

**How it works:**
- Workers refresh their registration every `heartbeat_interval`
- If no refresh within `worker_ttl_seconds`, worker considered dead
- Orchestrator excludes stale workers from routing

**Tuning:**
- **Shorter (5-15s):** Fast failure detection, more Redis traffic
- **Longer (60-120s):** Lower overhead, slower failure detection
- **Recommended:** `30s` (2x heartbeat interval)

**Example:**
```yaml
redis:
  worker_ttl_seconds: 30
```

#### `redis.connection_pool_size`

**Type:** `integer`
**Default:** `10`
**Valid Range:** `>= 1`

Redis connection pool size.

**Guidelines:**
- Minimum: `max_connections / 10`
- Recommended: `10-50` for most deployments
- Each connection: ~1 MB memory

**Example:**
```yaml
redis:
  connection_pool_size: 10
```

---

### routing

Worker routing configuration.

**M2 Status:**
- Static routing: ✅ Implemented
- Dynamic routing: 🚧 M9+ feature

#### `routing.static_worker_addr`

**Type:** `string`
**Default:** `"grpc://localhost:7001"`
**Valid Values:** gRPC URL

**M2:** Static worker address (single worker).

**URL Format:** `grpc://host:port`

**Examples:**
```yaml
# Local development
routing:
  static_worker_addr: "grpc://localhost:7001"

# Docker Compose
routing:
  static_worker_addr: "grpc://tts-worker:7002"

# Remote worker
routing:
  static_worker_addr: "grpc://192.168.1.100:7001"
```

**M9+ Migration:** Remove this field when dynamic routing is implemented.

#### `routing.prefer_resident_models`

**Type:** `boolean`
**Default:** `true`
**Valid Values:** `true`, `false`

**M9+ Feature:** Prefer workers with models already loaded in VRAM.

**Effect:**
- `true`: Route to workers with model resident (faster, no load latency)
- `false`: Load balance regardless of model residency

**M2 Status:** 🚧 Planned for M9+, currently ignored

**Example:**
```yaml
routing:
  prefer_resident_models: true
```

#### `routing.load_balance_strategy`

**Type:** `string`
**Default:** `"queue_depth"`
**Valid Values:** `"queue_depth"`, `"round_robin"`, `"latency"`

**M9+ Feature:** Load balancing strategy for multi-worker routing.

**Strategies:**
- `queue_depth`: Choose worker with fewest active sessions
- `round_robin`: Distribute requests evenly
- `latency`: Choose worker with lowest p50 latency

**M2 Status:** 🚧 Planned for M9+, currently ignored

**Example:**
```yaml
routing:
  load_balance_strategy: "queue_depth"
```

---

### vad

Voice Activity Detection configuration.

**M2 Status:** ✅ Implemented (stub for barge-in detection)

#### `vad.enabled`

**Type:** `boolean`
**Default:** `true`
**Valid Values:** `true`, `false`

Enable Voice Activity Detection.

**Purpose:** Detect speech for barge-in interruption.

**Example:**
```yaml
vad:
  enabled: true
```

#### `vad.aggressiveness`

**Type:** `integer`
**Default:** `2`
**Valid Range:** `0-3`

WebRTC VAD aggressiveness mode.

**Modes:**
- `0` - Least aggressive (high sensitivity, more false positives)
- `1` - Low aggressiveness
- `2` - Moderate (recommended, balanced)
- `3` - Most aggressive (high specificity, may miss soft speech)

**Tuning:**
- **Noisy environments:** Use `3` (reduce false positives)
- **Quiet environments:** Use `1` (detect soft speech)
- **General use:** Use `2` (balanced)

**Example:**
```yaml
vad:
  aggressiveness: 2
```

#### `vad.sample_rate`

**Type:** `integer`
**Default:** `16000`
**Valid Values:** `8000`, `16000`, `32000`, `48000`

Sample rate for VAD processing (Hz).

**Constraint:** WebRTC VAD requires specific sample rates.

**Recommended:** `16000` (optimal for speech detection)

**Note:** Different from TTS output sample rate (48000 Hz).

**Example:**
```yaml
vad:
  sample_rate: 16000
```

#### `vad.frame_duration_ms`

**Type:** `integer`
**Default:** `20`
**Valid Values:** `10`, `20`, `30`

VAD frame duration in milliseconds.

**Constraint:** WebRTC VAD supports only these values.

**Tradeoffs:**
- `10ms`: Lower latency, more CPU usage
- `20ms`: Balanced (recommended)
- `30ms`: Lower CPU, higher latency

**Example:**
```yaml
vad:
  frame_duration_ms: 20
```

---

### Operational Settings

#### `log_level`

**Type:** `string`
**Default:** `"INFO"`
**Valid Values:** `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`

Logging verbosity level.

**Levels:**
- `DEBUG`: All messages (verbose, development)
- `INFO`: Informational messages (recommended for production)
- `WARNING`: Warnings and errors only
- `ERROR`: Errors only
- `CRITICAL`: Critical errors only

**Example:**
```yaml
log_level: "INFO"
```

**Override with Environment:**
```bash
LOG_LEVEL=DEBUG python -m src.orchestrator.server
```

#### `graceful_shutdown_timeout_s`

**Type:** `integer`
**Default:** `10`
**Valid Range:** `>= 1`

Graceful shutdown timeout in seconds.

**Behavior:**
1. SIGTERM received
2. Stop accepting new connections
3. Wait up to `graceful_shutdown_timeout_s` for active sessions to complete
4. Force shutdown after timeout

**Recommendations:**
- Development: `5-10s`
- Production: `30-60s` (allow sessions to complete)

**Example:**
```yaml
graceful_shutdown_timeout_s: 10
```

---

## Worker Configuration

**File:** `configs/worker.yaml`

### Complete Example

```yaml
# TTS Worker Configuration

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
    emotive_zero_prompt: true
    max_concurrent_sessions: 3

model_manager:
  default_model_id: "cosyvoice2-en-base"
  preload_model_ids: []
  ttl_ms: 600000
  min_residency_ms: 120000
  evict_check_interval_ms: 30000
  resident_cap: 3
  max_parallel_loads: 1
  warmup_enabled: true
  warmup_text: "This is a warmup test."

audio:
  output_sample_rate: 48000
  frame_duration_ms: 20
  loudness_target_lufs: -16.0
  normalization_enabled: true

redis:
  url: "redis://localhost:6379"
  registration_ttl_seconds: 30
  heartbeat_interval_seconds: 10

metrics:
  enabled: true
  prometheus_port: 9090
  track_latency: true
  track_rtf: true
  track_queue_depth: true

logging:
  level: "INFO"
  format: "json"
  include_session_id: true
```

---

### worker

Worker identity and network configuration.

#### `worker.name`

**Type:** `string`
**Default:** `"tts-worker-0"`
**Valid Values:** Any unique string

Unique worker identifier.

**Naming Convention:** `tts-<adapter>-<index>`

**Examples:**
```yaml
worker:
  name: "tts-worker-0"      # Generic
  name: "tts-cosy-0"        # CosyVoice worker 0
  name: "tts-xtts-gpu1"     # XTTS on GPU 1
```

**Important:** Must be unique across all workers in the system.

#### `worker.grpc_host`

**Type:** `string`
**Default:** `"0.0.0.0"`
**Valid Values:** Any valid IP address or hostname

Host address to bind gRPC server.

**Common Values:**
- `"0.0.0.0"` - All interfaces (recommended for containers)
- `"127.0.0.1"` - Localhost only
- Specific IP address

**Example:**
```yaml
worker:
  grpc_host: "0.0.0.0"
```

#### `worker.grpc_port`

**Type:** `integer`
**Default:** `7002`
**Valid Range:** `1024-65535`

Port for gRPC server.

**Port Assignment:**
- Worker 0: `7001`
- Worker 1: `7002`
- Worker N: `7000 + N + 1`

**Example:**
```yaml
worker:
  grpc_port: 7002
```

---

### worker.capabilities

Worker capability advertisement for routing.

**M9+ Feature:** Used for capability-aware routing.

#### `worker.capabilities.streaming`

**Type:** `boolean`
**Default:** `true`

Supports streaming synthesis (20ms frames).

**Example:**
```yaml
worker:
  capabilities:
    streaming: true
```

#### `worker.capabilities.zero_shot`

**Type:** `boolean`
**Default:** `true`

Supports zero-shot synthesis (no reference audio).

**Example:**
```yaml
worker:
  capabilities:
    zero_shot: true
```

#### `worker.capabilities.lora`

**Type:** `boolean`
**Default:** `false`

Supports LoRA fine-tuned models.

**Adapters with LoRA:**
- Sesame/Unsloth: `true`
- CosyVoice: `false`
- XTTS: `false`
- Piper: `false`

**Example:**
```yaml
worker:
  capabilities:
    lora: true  # For Sesame/Unsloth adapters
```

#### `worker.capabilities.cpu_ok`

**Type:** `boolean`
**Default:** `false`

Worker can run on CPU (no GPU required).

**CPU-Compatible Adapters:**
- Piper: `true`
- All others: `false` (require GPU)

**Example:**
```yaml
worker:
  capabilities:
    cpu_ok: false  # GPU required
```

#### `worker.capabilities.languages`

**Type:** `array[string]`
**Default:** `["en"]`

Supported language codes (ISO 639-1).

**Common Codes:**
- `en` - English
- `zh` - Chinese
- `es` - Spanish
- `fr` - French

**Example:**
```yaml
worker:
  capabilities:
    languages: ["en", "zh"]  # English and Chinese
```

#### `worker.capabilities.emotive_zero_prompt`

**Type:** `boolean`
**Default:** `true`

Supports emotional control without reference audio.

**Example:**
```yaml
worker:
  capabilities:
    emotive_zero_prompt: true
```

#### `worker.capabilities.max_concurrent_sessions`

**Type:** `integer`
**Default:** `3`
**Valid Range:** `>= 1`

Maximum concurrent synthesis sessions.

**Guidelines (per GPU):**
- Small models (Piper): `10-20`
- Medium models (CosyVoice): `3-5`
- Large models (XTTS): `1-3`

**Resource Impact:**
- Each session: ~500MB-2GB VRAM
- CPU: ~10-50% per session

**Example:**
```yaml
worker:
  capabilities:
    max_concurrent_sessions: 3
```

---

### model_manager

Model lifecycle management configuration.

**M4+ Feature:** Dynamic model loading/unloading.

#### `model_manager.default_model_id`

**Type:** `string`
**Default:** `"cosyvoice2-en-base"`
**Valid Values:** Any model ID in `voicepacks/` directory

**Required:** Default model loaded at startup.

**Format:** `<family>-<variant>` (maps to `voicepacks/<family>/<variant>/`)

**Examples:**
```yaml
model_manager:
  default_model_id: "cosyvoice2-en-base"
  # Expects: voicepacks/cosyvoice2/en-base/
```

**Validation:** Model must exist or startup fails.

#### `model_manager.preload_model_ids`

**Type:** `array[string]`
**Default:** `[]`

**M4+ Feature:** Additional models to preload at startup.

**Example:**
```yaml
model_manager:
  preload_model_ids:
    - "cosyvoice2-zh-base"
    - "xtts-v2-en-demo"
```

**Constraint:** Total preloaded models ≤ `resident_cap`.

#### `model_manager.ttl_ms`

**Type:** `integer`
**Default:** `600000` (10 minutes)
**Valid Range:** `>= 0`

**M4+ Feature:** Model eviction TTL in milliseconds.

**Behavior:**
- After `ttl_ms` of idle time (no active sessions), model unloaded
- `0` = never evict idle models

**Tuning:**
- **Short (60000-300000):** Aggressive VRAM reclamation
- **Long (900000-1800000):** Fewer reloads, higher VRAM usage
- **Recommended:** `600000` (10 minutes)

**Example:**
```yaml
model_manager:
  ttl_ms: 600000  # 10 minutes
```

#### `model_manager.min_residency_ms`

**Type:** `integer`
**Default:** `120000` (2 minutes)
**Valid Range:** `>= 0`

**M4+ Feature:** Minimum time model must stay loaded after load.

**Purpose:** Prevent thrashing (rapid load/unload cycles).

**Example:**
```yaml
model_manager:
  min_residency_ms: 120000  # 2 minutes
```

#### `model_manager.evict_check_interval_ms`

**Type:** `integer`
**Default:** `30000` (30 seconds)
**Valid Range:** `>= 0`

**M4+ Feature:** How often to check for evictable models.

**Tuning:**
- **Shorter (5000-15000):** More responsive eviction, higher CPU
- **Longer (60000-120000):** Lower overhead, delayed reclamation

**Example:**
```yaml
model_manager:
  evict_check_interval_ms: 30000
```

#### `model_manager.resident_cap`

**Type:** `integer`
**Default:** `3`
**Valid Range:** `>= 1`

**M4+ Feature:** Maximum models resident in VRAM simultaneously.

**Calculation:**
```
resident_cap = GPU_VRAM_GB / Average_Model_Size_GB
```

**Examples:**
- 24GB GPU, 4GB models: `resident_cap = 6`
- 12GB GPU, 4GB models: `resident_cap = 3`
- 8GB GPU, 2GB models: `resident_cap = 4`

**Example:**
```yaml
model_manager:
  resident_cap: 3
```

#### `model_manager.max_parallel_loads`

**Type:** `integer`
**Default:** `1`
**Valid Range:** `>= 1`

**M4+ Feature:** Maximum concurrent model loads.

**Purpose:** Prevent OOM from loading multiple large models simultaneously.

**Recommendation:** Keep at `1` unless you have >32GB VRAM.

**Example:**
```yaml
model_manager:
  max_parallel_loads: 1
```

#### `model_manager.warmup_enabled`

**Type:** `boolean`
**Default:** `true`

Enable model warmup at load time.

**Purpose:**
- JIT compilation (first inference slow)
- Allocate CUDA kernels
- Verify model works

**Warmup Duration:** ~300ms

**Example:**
```yaml
model_manager:
  warmup_enabled: true
```

#### `model_manager.warmup_text`

**Type:** `string`
**Default:** `"This is a warmup test."`

Text used for model warmup inference.

**Example:**
```yaml
model_manager:
  warmup_text: "This is a warmup test."
```

---

### audio

Audio processing configuration.

#### `audio.output_sample_rate`

**Type:** `integer`
**Default:** `48000`
**Valid Range:** `>= 8000`

Output audio sample rate in Hz.

**Standard Rates:**
- `48000` - Professional audio (recommended)
- `44100` - CD quality
- `22050` - Telephone quality
- `16000` - Narrow-band speech

**Constraint:** Fixed at `48000` in M2.

**Example:**
```yaml
audio:
  output_sample_rate: 48000
```

#### `audio.frame_duration_ms`

**Type:** `integer`
**Default:** `20`
**Valid Range:** `>= 10`

Audio frame duration in milliseconds.

**Frame Sizes @ 48kHz:**
- `10ms` = 480 samples = 960 bytes
- `20ms` = 960 samples = 1920 bytes
- `30ms` = 1440 samples = 2880 bytes

**Constraint:** Fixed at `20ms` in M2.

**Example:**
```yaml
audio:
  frame_duration_ms: 20
```

#### `audio.loudness_target_lufs`

**Type:** `float`
**Default:** `-16.0`
**Valid Range:** `-23.0` to `-16.0` LUFS

Target loudness normalization level.

**Standards:**
- `-23 LUFS` - EBU R128 (broadcast)
- `-16 LUFS` - Streaming/YouTube (recommended)
- `-14 LUFS` - Apple Music/Spotify

**Example:**
```yaml
audio:
  loudness_target_lufs: -16.0
```

#### `audio.normalization_enabled`

**Type:** `boolean`
**Default:** `true`

Enable loudness normalization.

**Effect:**
- `true`: Normalize to `loudness_target_lufs`
- `false`: Raw model output (may vary in volume)

**Example:**
```yaml
audio:
  normalization_enabled: true
```

---

### redis (Worker)

Redis configuration for worker registration.

**M9+ Feature:** Service discovery and heartbeat.

#### `redis.url`

**Type:** `string`
**Default:** `"redis://localhost:6379"`

Redis server connection string (same format as orchestrator).

**Example:**
```yaml
redis:
  url: "redis://localhost:6379"
```

#### `redis.registration_ttl_seconds`

**Type:** `integer`
**Default:** `30`
**Valid Range:** `>= 5`

Worker registration TTL in Redis.

**Coordination:** Should match `orchestrator.redis.worker_ttl_seconds`.

**Example:**
```yaml
redis:
  registration_ttl_seconds: 30
```

#### `redis.heartbeat_interval_seconds`

**Type:** `integer`
**Default:** `10`
**Valid Range:** `>= 1`

How often worker refreshes registration.

**Recommendation:** `heartbeat_interval < registration_ttl / 2`

**Example:**
```yaml
redis:
  heartbeat_interval_seconds: 10
```

---

### metrics

Metrics and monitoring configuration.

**Future Feature:** Prometheus metrics export.

#### `metrics.enabled`

**Type:** `boolean`
**Default:** `true`

Enable metrics collection.

**Example:**
```yaml
metrics:
  enabled: true
```

#### `metrics.prometheus_port`

**Type:** `integer`
**Default:** `9090`
**Valid Range:** `1024-65535`

Port for Prometheus metrics endpoint.

**Endpoint:** `http://<worker_host>:<prometheus_port>/metrics`

**Example:**
```yaml
metrics:
  prometheus_port: 9090
```

#### `metrics.track_latency`

**Type:** `boolean`
**Default:** `true`

Track First Audio Latency (FAL) metrics.

**Example:**
```yaml
metrics:
  track_latency: true
```

#### `metrics.track_rtf`

**Type:** `boolean`
**Default:** `true`

Track Real-Time Factor (RTF) metrics.

**Example:**
```yaml
metrics:
  track_rtf: true
```

#### `metrics.track_queue_depth`

**Type:** `boolean`
**Default:** `true`

Track session queue depth.

**Example:**
```yaml
metrics:
  track_queue_depth: true
```

---

### logging (Worker)

Worker logging configuration.

#### `logging.level`

**Type:** `string`
**Default:** `"INFO"`
**Valid Values:** `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`

Logging verbosity level.

**Example:**
```yaml
logging:
  level: "INFO"
```

#### `logging.format`

**Type:** `string`
**Default:** `"json"`
**Valid Values:** `"json"`, `"text"`

Log output format.

**Formats:**
- `json`: Structured JSON logs (recommended for production)
- `text`: Human-readable plain text (development)

**Example:**
```yaml
logging:
  format: "json"
```

#### `logging.include_session_id`

**Type:** `boolean`
**Default:** `true`

Include session ID in log messages.

**Purpose:** Trace individual sessions across log files.

**Example:**
```yaml
logging:
  include_session_id: true
```

---

## Environment Overrides

Configuration values can be overridden using environment variables.

### Common Overrides

```bash
# Port override
PORT=9000 python -m src.orchestrator.server

# Log level
LOG_LEVEL=DEBUG python -m src.orchestrator.server

# Redis URL
REDIS_URL="redis://redis-prod:6379" python -m src.orchestrator.server

# Worker name (for multi-worker setups)
WORKER_NAME="tts-worker-1" python -m src.tts.worker
```

### Docker Compose Example

```yaml
services:
  orchestrator:
    environment:
      - LOG_LEVEL=INFO
      - PORT=8080
      - REDIS_URL=redis://redis:6379

  tts-worker:
    environment:
      - LOG_LEVEL=INFO
      - WORKER_NAME=tts-worker-0
      - REDIS_URL=redis://redis:6379
```

---

## Configuration Patterns

### Single-GPU Development Setup

**orchestrator.yaml:**
```yaml
transport:
  websocket:
    port: 8080
    max_connections: 10

routing:
  static_worker_addr: "grpc://localhost:7001"

redis:
  url: "redis://localhost:6379"

log_level: "DEBUG"
```

**worker.yaml:**
```yaml
worker:
  name: "tts-worker-0"
  grpc_port: 7001
  capabilities:
    max_concurrent_sessions: 3

model_manager:
  default_model_id: "cosyvoice2-en-base"
  resident_cap: 2

redis:
  url: "redis://localhost:6379"

logging:
  level: "DEBUG"
  format: "text"
```

### Multi-GPU Production Setup

**orchestrator.yaml:**
```yaml
transport:
  websocket:
    port: 8080
    max_connections: 500

routing:
  prefer_resident_models: true
  load_balance_strategy: "queue_depth"

redis:
  url: "redis://redis-prod:6379"
  connection_pool_size: 50

log_level: "INFO"
graceful_shutdown_timeout_s: 30
```

**worker-gpu0.yaml:**
```yaml
worker:
  name: "tts-cosy-gpu0"
  grpc_port: 7001
  capabilities:
    max_concurrent_sessions: 5

model_manager:
  default_model_id: "cosyvoice2-en-base"
  preload_model_ids: ["cosyvoice2-zh-base"]
  resident_cap: 4

logging:
  level: "INFO"
  format: "json"
```

**worker-gpu1.yaml:**
```yaml
worker:
  name: "tts-xtts-gpu1"
  grpc_port: 7002
  capabilities:
    max_concurrent_sessions: 3

model_manager:
  default_model_id: "xtts-v2-en-demo"
  resident_cap: 3

logging:
  level: "INFO"
  format: "json"
```

### Docker Compose Configuration

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  orchestrator:
    build: .
    ports:
      - "8080:8080"
    environment:
      - LOG_LEVEL=INFO
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    volumes:
      - ./configs/orchestrator.yaml:/app/configs/orchestrator.yaml

  tts-worker:
    build: .
    ports:
      - "7001:7001"
    environment:
      - LOG_LEVEL=INFO
      - WORKER_NAME=tts-worker-0
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    volumes:
      - ./configs/worker.yaml:/app/configs/worker.yaml
      - ./voicepacks:/app/voicepacks
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Validation

### Configuration Validation Script

**Coming Soon:** `just validate-config` (P1 task - @incident-responder)

**Manual Validation:**

```bash
# Test orchestrator config
python -c "
import yaml
from src.orchestrator.config import OrchestratorConfig
config = yaml.safe_load(open('configs/orchestrator.yaml'))
OrchestratorConfig(**config)
print('✓ orchestrator.yaml valid')
"

# Test worker config (requires Pydantic models - P1 task)
# Currently loads as dict[str, Any]
python -c "
import yaml
config = yaml.safe_load(open('configs/worker.yaml'))
print('✓ worker.yaml loaded (validation pending P1)')
"
```

---

## Related Documentation

- [Quick Start Guide](QUICKSTART.md) - System setup with default configs
- [WebSocket Protocol](WEBSOCKET_PROTOCOL.md) - Transport layer configuration
- [Docker Setup](setup/DOCKER_SETUP.md) - Container configuration
- [Performance Tuning](PERFORMANCE.md) - Performance-related config tuning

---

## Changelog

**v0.2.0 (M2):**
- Initial configuration system
- Pydantic validation for orchestrator config
- YAML-based configuration files
- Environment variable overrides
- Value range comments in YAML

**Future:**
- Worker config Pydantic validation (P1)
- Config validation tool (P1)
- Hot reload support (M4+)
- Web UI for configuration (M9+)
