# Multi-GPU Deployment Guide

This guide covers deploying the Realtime Duplex Voice Demo system across multiple GPUs for increased capacity, redundancy, and model diversity.

---

## Deployment Topologies

The system supports three multi-GPU deployment patterns:

### 1. Same-Host Multi-GPU (Primary Pattern)

**Use Case:** Single server with 2-8 GPUs

**Architecture:**
- Single orchestrator process
- N worker processes (one per GPU)
- Shared Redis instance
- All communication via localhost

**Capacity:** 3-8 concurrent sessions per GPU (model-dependent)

### 2. Multi-GPU with Model Diversity (Same Host)

**Use Case:** Offer different TTS models simultaneously

**Architecture:**
- Single orchestrator
- Multiple workers per GPU, each with different adapter
- Redis-based capability routing
- Clients select model via `model_id` parameter

**Example:** GPU 0 runs CosyVoice, GPU 1 runs XTTS-v2

### 3. Multi-Host Multi-GPU (Advanced)

**Use Case:** Scale beyond single server capacity

**Architecture:**
- Multiple hosts on LAN
- One orchestrator (or load-balanced orchestrators)
- Workers distributed across hosts
- Centralized Redis or Redis Cluster

**Capacity:** Virtually unlimited (network-bound)

**Note:** Multi-host deployment documented in M13+ milestones. This guide focuses on same-host patterns.

---

## Same-Host Multi-GPU Setup

### Prerequisites

**Hardware:**
- 2+ NVIDIA GPUs (same or different models)
- PCIe lanes: 8x or 16x per GPU (avoid 4x, limits throughput)
- VRAM: 8GB+ per GPU for most models

**Software:**
- CUDA Toolkit 12.8+ with multi-GPU support
- nvidia-smi detecting all GPUs: `nvidia-smi -L`
- Docker with `--gpus all` support (if using containers)

**Verification:**

```bash
# Verify all GPUs visible
nvidia-smi -L
# Expected output:
# GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-...)
# GPU 1: NVIDIA GeForce RTX 4090 (UUID: GPU-...)

# Verify CUDA devices
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
# Expected: CUDA devices: 2
```

---

## Pattern 1: One Worker Per GPU (Same Adapter)

**Goal:** Maximize capacity for a single TTS model across multiple GPUs.

### Configuration

**Directory Structure:**

```
config/
├── orchestrator.yaml       # Single orchestrator config
├── worker-gpu0.yaml        # Worker for GPU 0
└── worker-gpu1.yaml        # Worker for GPU 1
```

**worker-gpu0.yaml:**

```yaml
worker:
  name: "tts-cosy@0"         # Unique name per GPU
  grpc_port: 7001            # Unique port per worker
  adapter_type: "cosyvoice2"

model_manager:
  default_model_id: "cosyvoice2-en-base"
  preload_model_ids:
    - "cosyvoice2-en-base"
  resident_cap: 2

redis:
  url: "redis://localhost:6379"
  heartbeat_interval_ms: 5000
  worker_ttl_ms: 15000

audio:
  sample_rate: 48000
  frame_duration_ms: 20
```

**worker-gpu1.yaml:**

```yaml
worker:
  name: "tts-cosy@1"         # Unique name for GPU 1
  grpc_port: 7002            # Different port
  adapter_type: "cosyvoice2"

model_manager:
  default_model_id: "cosyvoice2-en-base"
  preload_model_ids:
    - "cosyvoice2-en-base"
  resident_cap: 2

redis:
  url: "redis://localhost:6379"
  heartbeat_interval_ms: 5000
  worker_ttl_ms: 15000

audio:
  sample_rate: 48000
  frame_duration_ms: 20
```

**orchestrator.yaml:**

```yaml
transport:
  type: "websocket"
  websocket:
    host: "0.0.0.0"
    port: 8080
    max_connections: 20  # Increased for multi-GPU capacity

routing:
  # M9+ feature: discover workers via Redis
  prefer_resident_models: true
  load_balancing_strategy: "least_loaded"

redis:
  url: "redis://localhost:6379"

vad:
  aggressiveness: 2
  min_speech_duration_ms: 100
  min_silence_duration_ms: 300
```

### Launching Workers

**Option 1: Using `justfile` (Recommended for Development)**

```bash
# Terminal 1: Start Redis
just redis

# Terminal 2: Start worker on GPU 0
CUDA_VISIBLE_DEVICES=0 just run-tts-cosy DEFAULT="cosyvoice2-en-base"

# Terminal 3: Start worker on GPU 1
CUDA_VISIBLE_DEVICES=1 CONFIG=config/worker-gpu1.yaml just run-tts-cosy

# Terminal 4: Start orchestrator
just run-orch
```

**Option 2: Direct Python Invocation**

```bash
# GPU 0 worker
CUDA_VISIBLE_DEVICES=0 \
  python -m src.tts.worker --config config/worker-gpu0.yaml &

# GPU 1 worker
CUDA_VISIBLE_DEVICES=1 \
  python -m src.tts.worker --config config/worker-gpu1.yaml &

# Orchestrator
python -m src.orchestrator.server --config config/orchestrator.yaml &
```

**Option 3: Systemd Services (Production)**

Create `/etc/systemd/system/tts-worker@.service`:

```ini
[Unit]
Description=TTS Worker on GPU %i
After=network.target redis.service

[Service]
Type=simple
User=voice-demo
WorkingDirectory=/opt/voice-demo
Environment="CUDA_VISIBLE_DEVICES=%i"
ExecStart=/usr/bin/python -m src.tts.worker --config config/worker-gpu%i.yaml
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable tts-worker@0.service tts-worker@1.service
sudo systemctl start tts-worker@0.service tts-worker@1.service
sudo systemctl start orchestrator.service
```

### Verification

**Check Worker Registration:**

```bash
# Query Redis for registered workers
redis-cli KEYS "worker:*"

# Expected output:
# 1) "worker:tts-cosy@0"
# 2) "worker:tts-cosy@1"

# Get worker details
redis-cli HGETALL worker:tts-cosy@0
# Expected output:
# 1) "name"
# 2) "tts-cosy@0"
# 3) "addr"
# 4) "grpc://localhost:7001"
# 5) "capabilities"
# 6) "{\"streaming\": true, \"languages\": [\"en\", \"zh\"], ...}"
```

**Test Load Distribution:**

```bash
# Run concurrent test clients
for i in {1..6}; do
  just cli HOST="ws://localhost:8080" &
done

# Monitor GPU utilization
watch -n 1 nvidia-smi

# Expected: Both GPUs show 40-60% utilization
```

**Check Session Distribution:**

```bash
# Query worker metrics
redis-cli HGET worker:tts-cosy@0 metrics
redis-cli HGET worker:tts-cosy@1 metrics

# Expected output (example):
# {"queue_depth": 2, "active_sessions": 3, "rtf": 0.22}
# {"queue_depth": 1, "active_sessions": 3, "rtf": 0.20}
```

---

## Pattern 2: Multiple Models Across GPUs

**Goal:** Offer different TTS models (CosyVoice on GPU 0, XTTS-v2 on GPU 1).

### Configuration

**worker-cosy-gpu0.yaml:**

```yaml
worker:
  name: "tts-cosy@0"
  grpc_port: 7001
  adapter_type: "cosyvoice2"

model_manager:
  default_model_id: "cosyvoice2-en-base"
  preload_model_ids:
    - "cosyvoice2-en-base"
    - "cosyvoice2-zh-expressive"

redis:
  url: "redis://localhost:6379"
```

**worker-xtts-gpu1.yaml:**

```yaml
worker:
  name: "tts-xtts@1"
  grpc_port: 7002
  adapter_type: "xtts"

model_manager:
  default_model_id: "xtts-v2-en-demo"
  preload_model_ids:
    - "xtts-v2-en-demo"
    - "xtts-v2-en-emotive"

redis:
  url: "redis://localhost:6379"
```

### Launching

```bash
# Start Redis
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Start CosyVoice worker on GPU 0
CUDA_VISIBLE_DEVICES=0 \
  python -m src.tts.worker --config config/worker-cosy-gpu0.yaml &

# Start XTTS worker on GPU 1
CUDA_VISIBLE_DEVICES=1 \
  python -m src.tts.worker --config config/worker-xtts-gpu1.yaml &

# Start orchestrator
python -m src.orchestrator.server --config config/orchestrator.yaml &
```

### Client Model Selection (M9+)

```python
# In WebSocket client
import asyncio
import websockets
import json

async def synthesize_with_model(model_id: str, text: str):
    async with websockets.connect("ws://localhost:8080") as ws:
        # Start session with specific model
        await ws.send(json.dumps({
            "type": "SessionStart",
            "model_id": model_id,  # "cosyvoice2-en-base" or "xtts-v2-en-demo"
            "sample_rate": 48000,
            "language": "en"
        }))

        response = json.loads(await ws.recv())
        print(f"Session started: {response['session_id']}")

        # Send text
        await ws.send(json.dumps({
            "type": "TextChunk",
            "text": text,
            "is_final": True
        }))

        # Receive audio frames...

# Use CosyVoice (GPU 0)
await synthesize_with_model("cosyvoice2-en-base", "Hello from CosyVoice")

# Use XTTS (GPU 1)
await synthesize_with_model("xtts-v2-en-demo", "Hello from XTTS")
```

**Orchestrator Routing Logic (M9+):**

The orchestrator queries Redis for workers supporting the requested model:

1. **Filter by capability:** Language, streaming support, model availability
2. **Prefer resident models:** Workers with model already loaded in VRAM
3. **Load balancing:** Select worker with lowest `queue_depth`

See `/home/gerald/git/full-duplex-voice-chat/src/orchestrator/routing.py` for implementation.

---

## Worker Pinning with CUDA_VISIBLE_DEVICES

### How It Works

The `CUDA_VISIBLE_DEVICES` environment variable controls which GPUs are visible to a process.

**Examples:**

```bash
# Show only GPU 0 (first GPU)
CUDA_VISIBLE_DEVICES=0 python -m src.tts.worker

# Show only GPU 1 (second GPU)
CUDA_VISIBLE_DEVICES=1 python -m src.tts.worker

# Show GPUs 0 and 2 (skip GPU 1)
CUDA_VISIBLE_DEVICES=0,2 python -m src.tts.worker

# Hide all GPUs (force CPU mode)
CUDA_VISIBLE_DEVICES= python -m src.tts.worker
```

**Inside the Process:**

```python
import torch

# If CUDA_VISIBLE_DEVICES=1, the process sees only GPU 1 as device 0
print(torch.cuda.device_count())  # Output: 1
print(torch.cuda.current_device())  # Output: 0 (logical device ID)

# The logical device 0 maps to physical GPU 1
# No code changes needed; PyTorch handles mapping
```

### Best Practices

**1. Pin One GPU Per Worker:**

```bash
# Correct: Each worker sees only its assigned GPU
CUDA_VISIBLE_DEVICES=0 python -m src.tts.worker --config worker-gpu0.yaml &
CUDA_VISIBLE_DEVICES=1 python -m src.tts.worker --config worker-gpu1.yaml &
```

**2. Avoid Shared GPU Access:**

```bash
# Incorrect: Both workers see both GPUs, may collide
CUDA_VISIBLE_DEVICES=0,1 python -m src.tts.worker --config worker-gpu0.yaml &
CUDA_VISIBLE_DEVICES=0,1 python -m src.tts.worker --config worker-gpu1.yaml &
# Risk: Both workers may allocate on GPU 0, causing OOM
```

**3. Verify GPU Assignment:**

```python
# Add to worker startup logging
import torch
import os

logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
logger.info(f"Detected {torch.cuda.device_count()} GPU(s)")
if torch.cuda.is_available():
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
```

**4. Docker GPU Assignment:**

```yaml
# In docker-compose.yml
services:
  tts-worker-gpu0:
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  tts-worker-gpu1:
    environment:
      - CUDA_VISIBLE_DEVICES=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
```

---

## Load Balancing Strategies

The orchestrator supports multiple load balancing strategies (M9+ feature).

### 1. Least Loaded (Default)

**Algorithm:** Route to worker with lowest `queue_depth`.

**Configuration:**

```yaml
# In orchestrator.yaml
routing:
  load_balancing_strategy: "least_loaded"
```

**Use Case:** Maximize utilization across workers.

**Pros:**
- Balances load evenly
- Avoids hotspots

**Cons:**
- May trigger model loading if requested model not resident

### 2. Prefer Resident Models

**Algorithm:** Route to worker with model already loaded, even if slightly more loaded.

**Configuration:**

```yaml
routing:
  load_balancing_strategy: "prefer_resident"
  resident_weight: 0.8  # Prefer resident even if queue_depth +80% higher
```

**Use Case:** Minimize model loading overhead, especially with TTL eviction.

**Pros:**
- Reduces cold-start latency
- Better VRAM efficiency

**Cons:**
- May create load imbalance if clients cluster on same model

### 3. Round Robin

**Algorithm:** Cycle through workers sequentially.

**Configuration:**

```yaml
routing:
  load_balancing_strategy: "round_robin"
```

**Use Case:** Simple, deterministic distribution.

**Pros:**
- Predictable distribution
- Minimal orchestrator overhead

**Cons:**
- Ignores actual worker load
- No model affinity

### 4. Capability-Aware Routing

**Algorithm:** Filter by required capabilities (language, zero-shot, etc.), then apply load balancing.

**Configuration:**

```yaml
routing:
  load_balancing_strategy: "capability_aware"
  required_capabilities:
    - "streaming"
    - "zero_shot"
```

**Use Case:** Multi-adapter deployment with diverse capabilities.

**Example:**

```python
# Client requests zero-shot cloning (only XTTS supports)
session_start = {
    "type": "SessionStart",
    "model_id": "xtts-v2-en-demo",
    "capabilities": ["zero_shot"],
    "reference_audio": "<base64-encoded-reference>"
}

# Orchestrator routes only to workers advertising "zero_shot" capability
# (filters out CosyVoice workers)
```

---

## Redis-Based Discovery Across GPUs

### Worker Announcement Protocol

Each worker announces itself to Redis on startup and maintains a heartbeat.

**Announcement (on startup):**

```python
# In src/tts/worker.py
import redis.asyncio as redis
import json

async def announce_to_redis(config: TTSWorkerConfig, capabilities: dict):
    redis_client = redis.from_url(config.redis.url)

    worker_key = f"worker:{config.worker.name}"
    worker_data = {
        "name": config.worker.name,
        "addr": f"grpc://localhost:{config.worker.grpc_port}",
        "adapter_type": config.worker.adapter_type,
        "capabilities": json.dumps(capabilities),
        "resident_models": json.dumps([config.model_manager.default_model_id]),
        "metrics": json.dumps({"queue_depth": 0, "active_sessions": 0, "rtf": 0.0}),
        "last_heartbeat": time.time()
    }

    await redis_client.hset(worker_key, mapping=worker_data)
    await redis_client.expire(worker_key, config.redis.worker_ttl_ms // 1000)

    logger.info(f"Worker {config.worker.name} announced to Redis at {worker_data['addr']}")
```

**Heartbeat (every 5s):**

```python
async def heartbeat_loop(config: TTSWorkerConfig, get_metrics_fn):
    redis_client = redis.from_url(config.redis.url)
    interval = config.redis.heartbeat_interval_ms / 1000.0

    while True:
        await asyncio.sleep(interval)

        worker_key = f"worker:{config.worker.name}"
        metrics = get_metrics_fn()  # Get current queue_depth, active_sessions

        await redis_client.hset(worker_key, mapping={
            "metrics": json.dumps(metrics),
            "last_heartbeat": time.time()
        })
        await redis_client.expire(worker_key, config.redis.worker_ttl_ms // 1000)
```

**TTL Expiration:** If worker crashes or network partitions, Redis expires the key after `worker_ttl_ms` (default: 15s).

### Orchestrator Discovery

**Query all workers:**

```python
# In src/orchestrator/registry.py
async def discover_workers(redis_client: redis.Redis) -> List[WorkerInfo]:
    worker_keys = await redis_client.keys("worker:*")
    workers = []

    for key in worker_keys:
        worker_data = await redis_client.hgetall(key)
        if not worker_data:
            continue  # Expired or deleted

        workers.append(WorkerInfo(
            name=worker_data['name'],
            addr=worker_data['addr'],
            capabilities=json.loads(worker_data['capabilities']),
            resident_models=json.loads(worker_data['resident_models']),
            metrics=json.loads(worker_data['metrics'])
        ))

    return workers
```

**Filter and route:**

```python
async def route_session(
    model_id: str,
    language: str,
    redis_client: redis.Redis
) -> WorkerInfo:
    workers = await discover_workers(redis_client)

    # Filter by capability
    compatible = [
        w for w in workers
        if language in w.capabilities.get('languages', [])
        and w.capabilities.get('streaming', False)
    ]

    if not compatible:
        raise NoWorkerAvailableError(f"No worker supports language={language}")

    # Prefer workers with model already loaded
    resident = [w for w in compatible if model_id in w.resident_models]
    if resident:
        compatible = resident

    # Select least loaded
    compatible.sort(key=lambda w: w.metrics['queue_depth'])
    return compatible[0]
```

---

## Health Monitoring per GPU

### Per-Worker Health Checks

Each worker exposes a gRPC health endpoint:

```python
# In src/tts/worker.py
from grpc_health.v1 import health_pb2, health_pb2_grpc

class HealthServicer(health_pb2_grpc.HealthServicer):
    def Check(self, request, context):
        # Check GPU availability
        if not torch.cuda.is_available():
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.NOT_SERVING
            )

        # Check model loaded
        if not self.model_manager.is_loaded(self.default_model_id):
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.NOT_SERVING
            )

        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING
        )

# Register in gRPC server
health_servicer = HealthServicer()
health_pb2_grpc.add_HealthServicer_to_server(health_servicer, grpc_server)
```

**Check health:**

```bash
# Using grpcurl
grpcurl -plaintext localhost:7001 grpc.health.v1.Health/Check

# Expected output:
# {
#   "status": "SERVING"
# }
```

### GPU Metrics Monitoring

**Monitor GPU utilization per worker:**

```bash
# Real-time monitoring
nvidia-smi dmon -i 0,1 -s ucm

# Expected output:
# gpu   pwr  temp    sm   mem   enc   dec
#   0    45    62    55    40     0     0
#   1    48    65    60    45     0     0
```

**Prometheus Metrics (M11+):**

```python
# Export per-GPU metrics
from prometheus_client import Gauge

gpu_utilization = Gauge('worker_gpu_utilization_percent', 'GPU utilization', ['gpu_id', 'worker'])
gpu_memory_used = Gauge('worker_gpu_memory_used_bytes', 'GPU memory used', ['gpu_id', 'worker'])

# Update periodically
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

gpu_utilization.labels(gpu_id='0', worker='tts-cosy@0').set(utilization)
gpu_memory_used.labels(gpu_id='0', worker='tts-cosy@0').set(memory_info.used)
```

---

## Scaling Patterns

### Horizontal Scaling: Add More GPUs

**When to scale:**
- Concurrent sessions > 3 per GPU
- p95 latency exceeds 400ms
- GPU utilization > 90% sustained

**How to scale:**

```bash
# Add GPU 2 with same adapter
CUDA_VISIBLE_DEVICES=2 \
  python -m src.tts.worker --config config/worker-gpu2.yaml &

# Orchestrator automatically discovers new worker
# No restart required
```

**Capacity increase:** Linear (+3 sessions per GPU)

### Vertical Scaling: Upgrade GPU

**When to upgrade:**
- RTF > 0.4 (inference too slow)
- VRAM < 8GB (can't fit model)
- Multiple models needed per GPU

**Options:**
- RTX 3090 → RTX 4090: 2x VRAM (24GB → 48GB), 1.5x faster
- A100 40GB → A100 80GB: 2x VRAM, same speed
- Consumer → Datacenter GPU: Better FP16/BF16 performance

**Impact:** 2-4x model capacity, 1.5-2x throughput

### Hybrid Scaling: CPU + GPU Workers

**Use Case:** Handle overflow traffic with CPU workers.

**Architecture:**
- GPU workers (priority routing)
- CPU workers (Piper adapter, fallback)
- Orchestrator routes to GPU first, falls back to CPU if overloaded

**Configuration:**

```yaml
# In orchestrator.yaml
routing:
  load_balancing_strategy: "priority"
  worker_priority:
    - adapter_type: "cosyvoice2"  # GPU workers
      priority: 1
    - adapter_type: "piper"       # CPU workers
      priority: 2
  overflow_threshold: 5  # Route to priority 2 if queue_depth > 5
```

---

## Common Pitfalls

### Pitfall 1: GPU Memory Fragmentation

**Symptom:** OOM errors despite low reported memory usage.

**Cause:** Frequent model loading/unloading causes fragmentation.

**Solution:**

```bash
# Restart workers periodically (daily cron job)
0 3 * * * systemctl restart tts-worker@0 tts-worker@1
```

Or implement memory defragmentation:

```python
# In worker shutdown/restart logic
import torch
torch.cuda.empty_cache()
```

### Pitfall 2: Imbalanced GPU Utilization

**Symptom:** GPU 0 at 90%, GPU 1 at 20%.

**Cause:** Orchestrator not using metrics-based routing.

**Solution:** Enable `least_loaded` strategy and verify heartbeat metrics:

```bash
# Check worker metrics in Redis
redis-cli HGET worker:tts-cosy@0 metrics
redis-cli HGET worker:tts-cosy@1 metrics
```

### Pitfall 3: Worker Heartbeat Failures

**Symptom:** Workers disappear from Redis intermittently.

**Cause:** Redis connection timeout or network issues.

**Solution:** Increase heartbeat frequency and connection pooling:

```yaml
redis:
  heartbeat_interval_ms: 3000  # More frequent
  worker_ttl_ms: 12000         # 4x heartbeat interval
  connection_pool_size: 10
```

### Pitfall 4: PCIe Bandwidth Bottleneck

**Symptom:** GPU utilization < 50% despite high load.

**Cause:** Slow PCIe lanes (x4 instead of x16).

**Diagnosis:**

```bash
# Check PCIe link speed
nvidia-smi -q -d PCIE

# Expected: PCIe Gen3 x16 or Gen4 x16
# Bad: PCIe Gen3 x4 (bottleneck)
```

**Solution:** Move GPU to x16 slot or upgrade motherboard.

---

## Production Deployment Checklist

- [ ] **GPU verification:** All GPUs visible (`nvidia-smi -L`)
- [ ] **Worker configs:** Unique names and ports per worker
- [ ] **CUDA pinning:** `CUDA_VISIBLE_DEVICES` set for each worker
- [ ] **Redis running:** Accessible to all workers
- [ ] **Health checks:** All workers respond to gRPC health endpoint
- [ ] **Worker discovery:** All workers visible in Redis (`redis-cli KEYS "worker:*"`)
- [ ] **Load balancing:** Metrics-based routing enabled (`least_loaded` or `prefer_resident`)
- [ ] **Monitoring:** GPU utilization and memory tracked
- [ ] **Alerting:** Configured for OOM, high latency, worker failures
- [ ] **Failover:** Workers restart automatically on crash (systemd or Docker restart policy)
- [ ] **Logging:** Centralized log aggregation (e.g., ELK stack)
- [ ] **Capacity test:** Sustained 3+ sessions per GPU without degradation

---

## Further Reading

- **Performance Tuning:** `/home/gerald/git/full-duplex-voice-chat/docs/PERFORMANCE.md`
- **Observability:** `/home/gerald/git/full-duplex-voice-chat/docs/OBSERVABILITY.md`
- **Configuration Reference:** `/home/gerald/git/full-duplex-voice-chat/docs/CONFIGURATION_REFERENCE.md`
- **Runbook: Worker Connectivity:** `/home/gerald/git/full-duplex-voice-chat/docs/runbooks/GRPC_WORKER.md`
- **Architecture Diagrams:** `/home/gerald/git/full-duplex-voice-chat/docs/architecture/ARCHITECTURE.md`

---

**Last Updated:** 2025-10-05
**Target Milestone:** M9 (Multi-worker routing), M13 (Multi-host deployment)
