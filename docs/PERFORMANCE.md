# Performance Tuning Guide

This guide covers optimization strategies for achieving and maintaining production-grade performance targets in the Realtime Duplex Voice Demo system.

---

## Performance Targets

The system is designed to meet the following Service Level Objectives (SLOs):

**Latency SLAs:**
- **First Audio Latency (FAL):** p95 < 300ms for GPU adapters, < 500ms for Piper CPU
- **Barge-in pause latency:** p95 < 50ms (time from VAD detection to worker pause acknowledgment)
- **Frame jitter:** p95 < 10ms under 3 concurrent sessions
- **End-to-end latency:** p95 < 400ms (speech input to audio playback start)

**Throughput:**
- **Concurrent sessions:** 3+ sessions per GPU (model-dependent)
- **Real-time factor (RTF):** < 0.3 for GPU models (30% of real-time = 3x faster than realtime)

---

## 1. First Audio Latency (FAL) Optimization

First Audio Latency is the time from receiving the first text chunk to emitting the first audio frame. This is critical for perceived responsiveness.

### Target: p95 < 300ms (GPU adapters)

### Optimization Strategies

#### 1.1 Model Preloading

**Problem:** Cold-start model loading adds 2-10 seconds to first request.

**Solution:** Configure `preload_model_ids` in `worker.yaml`:

```yaml
model_manager:
  default_model_id: "cosyvoice2-en-base"
  preload_model_ids:
    - "cosyvoice2-en-base"
    - "xtts-v2-en-demo"
  resident_cap: 3
```

**Impact:** Eliminates cold-start penalty, ensures FAL < 300ms for preloaded models.

#### 1.2 CUDA Graph Optimization

**Problem:** PyTorch kernel launches add overhead on each inference call.

**Solution:** Enable CUDA graphs for static computation graphs (future M8+ feature):

```python
# In adapter implementation (example for future use)
if torch.cuda.is_available():
    # Warmup to record graph
    for _ in range(3):
        _ = model.generate(warmup_input)

    # Capture graph for inference
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model.generate(input_tensor)
```

**Impact:** 10-20% reduction in inference latency for models with static graphs.

#### 1.3 Half-Precision Inference

**Problem:** Full FP32 inference is slower than necessary.

**Solution:** Use FP16 or BF16 for inference where supported:

```python
# In adapter implementation
model = model.half()  # Convert to FP16
# Or use torch.autocast
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model.generate(input)
```

**Impact:** 30-50% reduction in inference time on modern GPUs (Ampere, Hopper).

**Compatibility:** Most TTS models support FP16 without quality degradation. Test audio quality before deploying.

#### 1.4 Batch Size = 1 for Streaming

**Problem:** Batching adds latency in streaming scenarios.

**Solution:** Keep batch size at 1 for realtime inference:

```yaml
# In adapter-specific config
inference:
  batch_size: 1
  streaming: true
```

**Impact:** Minimizes queuing delay, prioritizes low latency over throughput.

---

## 2. Frame Jitter Reduction

Frame jitter is variation in frame emission timing. High jitter causes audio glitches and buffer underruns.

### Target: p95 < 10ms

### Optimization Strategies

#### 2.1 Consistent Frame Duration

**Problem:** Variable-length audio chunks from model cause irregular frame timing.

**Solution:** Implement strict 20ms repacketization in adapters:

```python
# In adapter audio framing logic
from src.tts.audio.framing import repacketize_to_frames

async def synthesize_stream(self, text_chunks):
    for text_chunk in text_chunks:
        raw_audio = self.model.generate(text_chunk.text)

        # Repacketize to exact 20ms frames at 48kHz
        for frame in repacketize_to_frames(raw_audio, target_duration_ms=20, sample_rate=48000):
            yield AudioFrame(data=frame, duration_ms=20)
```

**Impact:** Eliminates jitter from model-generated chunk boundaries.

#### 2.2 Priority Thread/Process Scheduling

**Problem:** OS scheduler preempts audio emission thread, causing gaps.

**Solution:** Set high priority for gRPC streaming thread (Linux):

```python
import os
import psutil

# Set high priority for worker process
proc = psutil.Process(os.getpid())
proc.nice(-10)  # Requires CAP_SYS_NICE or root

# Or use real-time scheduling (requires CAP_SYS_NICE)
import ctypes
SCHED_FIFO = 1
sched_param = ctypes.c_int(50)  # Priority 50
ctypes.CDLL('libc.so.6').sched_setscheduler(0, SCHED_FIFO, ctypes.byref(sched_param))
```

**Impact:** Reduces jitter from 15-30ms to < 5ms under load.

**Warning:** Use with caution; can starve other processes. Test thoroughly.

#### 2.3 Dedicated CPU Cores (Pinning)

**Problem:** Thread migration between CPU cores causes cache misses and latency spikes.

**Solution:** Pin worker to specific CPU cores:

```bash
# Run worker on cores 0-3
taskset -c 0-3 just run-tts-cosy
```

Or in Docker Compose:

```yaml
services:
  tts-worker:
    cpuset: "0-3"
    mem_limit: 8g
```

**Impact:** Reduces p99 jitter by 20-30%.

#### 2.4 Disable CPU Frequency Scaling

**Problem:** CPU frequency transitions cause latency spikes.

**Solution:** Set CPU governor to `performance` mode:

```bash
# Set all CPUs to performance mode
sudo cpupower frequency-set -g performance

# Verify
cpupower frequency-info
```

**Impact:** Eliminates frequency-transition spikes (5-15ms).

---

## 3. Barge-in Latency Optimization

Barge-in latency is the time from detecting user speech to pausing TTS audio output.

### Target: p95 < 50ms

### Optimization Strategies

#### 3.1 VAD Aggressiveness Tuning

**Problem:** Conservative VAD settings add detection latency.

**Solution:** Increase VAD aggressiveness in `orchestrator.yaml`:

```yaml
vad:
  aggressiveness: 3  # Range: 0-3 (3 = most aggressive)
  min_speech_duration_ms: 60   # Lower = faster detection
  min_silence_duration_ms: 200
```

**Impact:** Reduces VAD detection latency from 100-200ms to 30-60ms.

**Tradeoff:** Higher false-positive rate (may trigger on background noise).

#### 3.2 Control Message Priority

**Problem:** PAUSE message queued behind audio frames in gRPC stream.

**Solution:** Use separate high-priority control channel (M3+ feature):

```protobuf
// Separate control RPC for low-latency commands
rpc Control(ControlRequest) returns (ControlResponse);

message ControlRequest {
  string session_id = 1;
  ControlCommand command = 2;  // PAUSE, RESUME, STOP
}
```

**Current Workaround:** Keep audio frame queue shallow (< 3 frames) to minimize queueing delay.

**Impact:** Reduces control latency from 20-60ms to < 10ms.

#### 3.3 Polling vs Streaming Control

**Problem:** Streaming control messages can lag during heavy audio emission.

**Solution:** Poll for control state in tight loop (every 5ms):

```python
# In adapter synthesize loop
last_control_check = time.monotonic()

for audio_chunk in model.generate_stream(text):
    # Check control state every 5ms
    if time.monotonic() - last_control_check > 0.005:
        if self.control_state == ControlState.PAUSED:
            # Stop emitting immediately
            break
        last_control_check = time.monotonic()

    yield audio_chunk
```

**Impact:** Reduces pause latency from 50-100ms to 10-20ms.

---

## 4. Concurrent Session Scaling

### Target: 3+ sessions per GPU

### Optimization Strategies

#### 4.1 GPU Memory Management

**Problem:** Multiple loaded models exhaust GPU memory.

**Solution:** Configure TTL-based eviction in `worker.yaml`:

```yaml
model_manager:
  ttl_ms: 600000           # Unload after 10 minutes idle
  min_residency_ms: 120000 # Keep at least 2 minutes
  resident_cap: 3          # Max 3 models in VRAM simultaneously
  evict_check_interval_ms: 30000
```

**Impact:** Allows dynamic model loading/unloading based on demand.

**Monitor:** Track `model_manager.resident_count` metric to ensure no thrashing.

#### 4.2 Batch Inference for Non-Streaming

**Problem:** Sequential inference underutilizes GPU.

**Solution:** For non-streaming use cases (M10+ ASR), batch requests:

```python
# In ASR module (Whisper)
class ASRBatcher:
    def __init__(self, max_batch_size=4, max_wait_ms=50):
        self.pending = []
        self.max_batch = max_batch_size
        self.max_wait = max_wait_ms

    async def transcribe(self, audio):
        self.pending.append(audio)

        if len(self.pending) >= self.max_batch or await_timeout(self.max_wait):
            batch_results = self.model.transcribe_batch(self.pending)
            self.pending.clear()
            return batch_results
```

**Impact:** 2-3x throughput increase for ASR, reduces CPU idle time.

**Tradeoff:** Adds up to `max_wait_ms` latency for first request in batch.

#### 4.3 Worker Pool Sizing

**Problem:** Single worker bottlenecks under high load.

**Solution:** Run multiple workers per GPU with shared model cache (M9+ feature):

```bash
# Run 2 workers on GPU 0, sharing Redis registry
CUDA_VISIBLE_DEVICES=0 just run-tts-cosy &
CUDA_VISIBLE_DEVICES=0 just run-tts-cosy --port 7002 &
```

**Impact:** 1.5-2x session capacity per GPU (limited by VRAM).

**Coordination:** Ensure both workers register with different names in Redis.

---

## 5. GPU Memory Optimization

### Optimization Strategies

#### 5.1 Model Quantization

**Problem:** FP32 models consume excessive VRAM.

**Solution:** Use INT8 or INT4 quantization where supported:

```python
# Example with bitsandbytes (future integration)
from bitsandbytes import quantize_model

model = quantize_model(model, bits=8)  # INT8 quantization
```

**Impact:** 2-4x reduction in VRAM usage with minimal quality loss.

**Compatibility:** Not all TTS models support quantization; test audio quality.

#### 5.2 Flash Attention (For Transformer Models)

**Problem:** Standard attention has O(n²) memory complexity.

**Solution:** Use Flash Attention 2 for models with attention layers:

```python
# Install: pip install flash-attn
from flash_attn import flash_attn_func

# In model config
model = Model(use_flash_attention=True)
```

**Impact:** 30-50% reduction in VRAM for long-context models.

#### 5.3 Gradient Checkpointing (Training Only)

Not applicable for inference-only workers. Relevant for future LoRA fine-tuning.

#### 5.4 Monitor GPU Memory Usage

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Detailed memory breakdown
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv -l 1

# Or use Python profiler
import torch
print(torch.cuda.memory_summary())
```

**Alerting:** Set up monitoring for `gpu.memory.used / gpu.memory.total > 0.9` to prevent OOM.

---

## 6. Redis Optimization

### Optimization Strategies

#### 6.1 Connection Pooling

**Problem:** Creating new Redis connections adds latency (5-20ms).

**Solution:** Use connection pool in orchestrator:

```python
import redis.asyncio as redis

# In orchestrator/registry.py
redis_pool = redis.ConnectionPool.from_url(
    config.redis.url,
    max_connections=20,
    decode_responses=True
)
redis_client = redis.Redis(connection_pool=redis_pool)
```

**Impact:** Reduces worker discovery latency from 10-20ms to < 1ms.

#### 6.2 Worker Heartbeat Optimization

**Problem:** Frequent heartbeats create Redis write contention.

**Solution:** Adjust heartbeat interval vs TTL:

```yaml
# In worker.yaml
redis:
  heartbeat_interval_ms: 5000  # Update every 5s
  worker_ttl_ms: 15000         # Expire after 15s (3x heartbeat)
```

**Impact:** Reduces Redis write load by 50-80% with acceptable staleness.

#### 6.3 Pub/Sub for Control Messages (M9+)

**Problem:** Polling Redis for control state adds latency.

**Solution:** Use Redis pub/sub for control events:

```python
# In orchestrator
await redis_client.publish(f"control:{session_id}", "PAUSE")

# In worker
async for message in redis_client.subscribe(f"control:{session_id}"):
    if message['data'] == 'PAUSE':
        self.pause_audio()
```

**Impact:** Reduces control latency from 10-50ms to < 5ms.

---

## 7. Network Tuning

### Optimization Strategies

#### 7.1 gRPC Keepalive Settings

**Problem:** Idle connections time out, causing reconnect overhead.

**Solution:** Configure gRPC keepalive in worker:

```python
import grpc

server = grpc.aio.server(
    options=[
        ('grpc.keepalive_time_ms', 10000),          # Send keepalive every 10s
        ('grpc.keepalive_timeout_ms', 5000),        # Wait 5s for keepalive ack
        ('grpc.keepalive_permit_without_calls', 1), # Allow keepalive when idle
        ('grpc.http2.max_pings_without_data', 0),   # Unlimited pings
    ]
)
```

**Impact:** Eliminates reconnect overhead (50-200ms) during bursty traffic.

#### 7.2 TCP Buffer Tuning (Linux)

**Problem:** Small TCP buffers cause backpressure under high throughput.

**Solution:** Increase TCP buffer sizes:

```bash
# Temporary (until reboot)
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"

# Permanent (add to /etc/sysctl.conf)
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
```

**Impact:** Reduces frame drops under high load (> 5 concurrent sessions).

#### 7.3 WebSocket Message Compression

**Problem:** Large audio frames consume bandwidth.

**Solution:** Enable WebSocket permessage-deflate compression:

```python
# In orchestrator WebSocket server
import websockets

async with websockets.serve(
    handler,
    host="0.0.0.0",
    port=8080,
    compression="deflate",  # Enable compression
    compression_level=6,    # Balance between CPU and size
):
    await asyncio.Future()
```

**Impact:** 40-60% reduction in bandwidth for audio frames (PCM is highly compressible).

**Tradeoff:** 1-5ms added CPU overhead for compression/decompression.

---

## 8. Profiling Tools

### 8.1 CPU Profiling with py-spy

**Use Case:** Identify Python-level bottlenecks.

**Installation:**

```bash
pip install py-spy
```

**Live Profiling:**

```bash
# Top-like view of function call time
just spy-top $(pgrep -f "tts/worker.py")

# Or manually
py-spy top --pid $(pgrep -f "tts/worker.py")
```

**Flame Graph Generation:**

```bash
# Record for 30 seconds
just spy-record $(pgrep -f "tts/worker.py") OUT="profile.svg"

# Or manually
py-spy record -o profile.svg --pid $(pgrep -f "tts/worker.py") --duration 30
```

**Analysis:**
- Open `profile.svg` in browser
- Look for wide bars (high CPU time)
- Focus on hot paths in synthesis loop

**Example Output:**

```
%CPU  Function
45%   model.generate() - Main TTS inference
15%   repacketize_to_frames() - Audio framing
10%   torch.cuda.synchronize() - GPU sync overhead
8%    grpc.aio.send() - Network transmission
```

**Optimization Target:** `model.generate()` should be > 70% of CPU time; if lower, optimize surrounding code.

### 8.2 GPU Profiling with Nsight Systems

**Use Case:** Identify GPU kernel bottlenecks and CPU-GPU sync issues.

**Installation:**

```bash
# Download from https://developer.nvidia.com/nsight-systems
# Or use CUDA Toolkit version
nsys --version
```

**Profiling:**

```bash
# Profile for 30 seconds with CUDA, Python, and gRPC traces
just nsys-tts

# Or manually
nsys profile \
  --trace=cuda,nvtx,python-gil \
  --sample=cpu \
  --duration=30 \
  --output=tts-worker-profile \
  python -m src.tts.worker
```

**Analysis:**

```bash
# Open in Nsight Systems GUI
nsys-ui tts-worker-profile.nsys-rep
```

**Look For:**
- **GPU idle time:** Should be < 10% during active synthesis
- **CPU-GPU sync gaps:** Large gaps indicate poor overlap
- **Kernel launch overhead:** Should be < 5% of inference time
- **NVTX ranges:** Custom markers for synthesis phases

**Example Timeline:**

```
Timeline:
|---Text Received---|
    |---Model Inference (GPU)---|
                      |---CPU Processing---|
                              |---Frame Emit---|
```

**Optimization Target:** GPU utilization > 80% during synthesis, minimal CPU-GPU sync gaps.

### 8.3 Kernel-Level Profiling with Nsight Compute

**Use Case:** Optimize specific CUDA kernels.

**Installation:**

```bash
# Included with CUDA Toolkit
ncu --version
```

**Profiling:**

```bash
# Profile specific kernel
just ncu-tts

# Or manually with kernel filter
ncu --kernel-name "regex:.*attention.*" \
    --set full \
    --target-processes all \
    python -m src.tts.worker
```

**Analysis:**

```bash
# Open in Nsight Compute GUI
ncu-ui tts-worker-kernel-profile.ncu-rep
```

**Metrics to Check:**
- **Compute throughput:** Should be > 60% of peak
- **Memory throughput:** Should be > 70% of peak
- **Occupancy:** Should be > 50% for most kernels
- **Warp execution efficiency:** Should be > 80%

**Optimization Target:** Roofline analysis shows kernels near compute or memory bound limit.

### 8.4 PyTorch Profiler with NVTX

**Use Case:** Integrated profiling of PyTorch operations.

**Code Instrumentation:**

```python
import torch.profiler as profiler

# In adapter synthesize method
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=profiler.tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    with_stack=True,
) as prof:
    for text_chunk in text_chunks:
        output = model.generate(text_chunk.text)
        prof.step()
```

**Visualization:**

```bash
# View in TensorBoard
tensorboard --logdir=./profiler_logs
```

**Analysis:**
- **Operator time:** Which PyTorch ops take longest
- **CUDA kernel time:** GPU execution breakdown
- **Memory usage:** Peak memory and allocations

### 8.5 Distributed Tracing with OpenTelemetry (M11+)

**Use Case:** End-to-end latency breakdown across services.

**Setup:** See `docs/OBSERVABILITY.md` for full configuration.

**Example Trace:**

```
WebSocket Receive (5ms)
  └─ Orchestrator Process (10ms)
      └─ gRPC Call to Worker (250ms)
          ├─ Network (5ms)
          ├─ Model Inference (200ms)
          ├─ Audio Framing (20ms)
          └─ gRPC Response (25ms)
      └─ WebSocket Send (5ms)
Total: 270ms (FAL within target)
```

---

## 9. Benchmarking Methodology

### 9.1 Latency Benchmarking

**Tool:** `scripts/benchmark-latency.py` (M11+)

**Metrics:**
- **First Audio Latency (FAL):** Time from first text chunk to first audio frame
- **Barge-in latency:** Time from VAD trigger to worker pause
- **End-to-end latency:** Full round-trip time

**Test Setup:**

```bash
# Run 100 iterations with varying text lengths
python scripts/benchmark-latency.py \
  --iterations 100 \
  --worker grpc://localhost:7001 \
  --percentiles 50,95,99

# Output:
# FAL p50: 180ms, p95: 250ms, p99: 310ms
# Barge-in p50: 25ms, p95: 42ms, p99: 58ms
```

**Pass Criteria:**
- FAL p95 < 300ms (GPU) or < 500ms (CPU)
- Barge-in p95 < 50ms

### 9.2 Frame Jitter Benchmarking

**Tool:** `scripts/benchmark-jitter.py` (M11+)

**Metrics:**
- **Inter-frame interval variance:** Stdev of time between frames
- **Frame drop rate:** Percentage of expected frames not received

**Test Setup:**

```bash
# Synthesize 30 seconds of audio, measure jitter
python scripts/benchmark-jitter.py \
  --duration 30 \
  --expected-interval 20ms \
  --worker grpc://localhost:7001

# Output:
# Mean interval: 20.1ms
# Jitter (stdev): 2.3ms
# p95 jitter: 8.5ms
# Frame drops: 0.1%
```

**Pass Criteria:**
- Jitter p95 < 10ms
- Frame drop rate < 0.5%

### 9.3 Concurrent Session Benchmarking

**Tool:** `scripts/benchmark-concurrency.py` (M11+)

**Metrics:**
- **Max concurrent sessions:** Number before p95 latency degrades > 20%
- **Throughput:** Total audio hours generated per wall-clock hour

**Test Setup:**

```bash
# Ramp up concurrent sessions, measure latency degradation
python scripts/benchmark-concurrency.py \
  --ramp-sessions 1,3,5,8,10 \
  --duration-per-level 60 \
  --worker grpc://localhost:7001

# Output:
# 1 session:  FAL p95 = 230ms, RTF = 0.18
# 3 sessions: FAL p95 = 260ms, RTF = 0.22
# 5 sessions: FAL p95 = 310ms, RTF = 0.28
# 8 sessions: FAL p95 = 420ms, RTF = 0.35 (DEGRADED)
# Max concurrent: 5 sessions (before 20% degradation)
```

**Pass Criteria:**
- 3+ concurrent sessions with p95 < 300ms (GPU)

---

## 10. Performance Monitoring in Production

### 10.1 Key Metrics to Track

**Orchestrator Metrics:**
- `orchestrator.sessions.active` - Current active sessions
- `orchestrator.latency.fal_ms` - First Audio Latency (histogram)
- `orchestrator.latency.barge_in_ms` - Barge-in pause latency (histogram)
- `orchestrator.errors.total` - Error count by type

**Worker Metrics:**
- `worker.inference.latency_ms` - Model inference time (histogram)
- `worker.inference.rtf` - Real-time factor (histogram)
- `worker.frames.jitter_ms` - Frame emission jitter (histogram)
- `worker.sessions.concurrent` - Active sessions per worker
- `worker.gpu.memory_used_bytes` - GPU memory usage
- `worker.gpu.utilization_percent` - GPU utilization

**Redis Metrics:**
- `redis.commands.latency_ms` - Redis command latency
- `redis.connections.active` - Active connections

See `docs/OBSERVABILITY.md` for full metrics catalog and Prometheus configuration.

### 10.2 Alerting Rules

**Critical Alerts:**

```yaml
# Prometheus alerting rules
groups:
  - name: voice_demo_slos
    rules:
      - alert: HighFirstAudioLatency
        expr: histogram_quantile(0.95, orchestrator_latency_fal_ms_bucket) > 300
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p95 FAL exceeds 300ms target"
          description: "Current p95 FAL: {{ $value }}ms"

      - alert: HighBargeInLatency
        expr: histogram_quantile(0.95, orchestrator_latency_barge_in_ms_bucket) > 50
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p95 barge-in latency exceeds 50ms target"

      - alert: HighFrameJitter
        expr: histogram_quantile(0.95, worker_frames_jitter_ms_bucket) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p95 frame jitter exceeds 10ms target"

      - alert: GPUMemoryExhaustion
        expr: worker_gpu_memory_used_bytes / worker_gpu_memory_total_bytes > 0.95
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory usage > 95%"
```

### 10.3 Performance Dashboards

**Recommended Grafana Dashboards:**

1. **Realtime Performance:**
   - Current sessions (gauge)
   - FAL p50/p95/p99 (time series)
   - Barge-in latency p95 (time series)
   - Frame jitter p95 (time series)

2. **Resource Utilization:**
   - GPU memory usage per worker (stacked area)
   - GPU utilization per worker (line chart)
   - CPU usage per process (line chart)
   - Redis command latency (heatmap)

3. **Session Overview:**
   - Sessions started/ended rate (line chart)
   - Session duration distribution (histogram)
   - Error rate by type (stacked area)

See `docs/OBSERVABILITY.md` for example Grafana dashboard JSON.

---

## 11. Common Performance Issues

### Issue: High FAL (> 500ms)

**Symptoms:**
- Slow response to text input
- Users perceive lag

**Diagnosis:**

```bash
# Check model loading time
grep "Model loaded" /var/log/tts-worker.log | tail -n 10

# Profile inference
just spy-record $(pgrep -f tts/worker.py)
```

**Likely Causes:**
1. **Cold start:** Model not preloaded → Configure `preload_model_ids`
2. **Slow inference:** Check RTF metric → Use FP16, optimize model
3. **Network latency:** Check gRPC latency → Reduce network hops, use local deployment

### Issue: High Frame Jitter (> 20ms p95)

**Symptoms:**
- Audio glitches or stuttering
- Buffer underruns in client

**Diagnosis:**

```bash
# Check CPU scheduler
chrt -p $(pgrep -f tts/worker.py)

# Monitor frame timing
python scripts/benchmark-jitter.py --duration 10
```

**Likely Causes:**
1. **CPU scheduling:** Low priority → Set process priority with `nice -n -10`
2. **Variable chunk sizes:** Model outputs irregular chunks → Implement strict repacketization
3. **GPU sync overhead:** Excessive `torch.cuda.synchronize()` → Remove unnecessary syncs

### Issue: Barge-in Latency > 100ms

**Symptoms:**
- Slow interruption response
- Audio continues after user starts speaking

**Diagnosis:**

```bash
# Check VAD configuration
grep "vad.aggressiveness" config/orchestrator.yaml

# Check control message latency
grep "PAUSE received" /var/log/tts-worker.log | grep -oP "latency: \K\d+"
```

**Likely Causes:**
1. **Slow VAD detection:** Aggressiveness too low → Set to 3
2. **Deep audio queue:** Too many buffered frames → Reduce queue depth
3. **Slow control path:** Polling interval too long → Reduce to 5ms

### Issue: Low Concurrent Session Capacity (< 3 per GPU)

**Symptoms:**
- OOM errors under load
- Session rejections

**Diagnosis:**

```bash
# Check GPU memory
nvidia-smi dmon -i 0 -s m

# Check resident models
redis-cli GET worker:tts-cosy@0:resident_models
```

**Likely Causes:**
1. **Too many resident models:** Exceeds VRAM → Lower `resident_cap`
2. **No model eviction:** Idle models not unloaded → Configure TTL eviction
3. **Memory leaks:** Memory usage grows over time → Profile with `torch.cuda.memory_summary()`

---

## 12. Performance Testing Checklist

Before deploying to production, validate performance:

- [ ] **FAL:** p95 < 300ms for GPU adapters (run `benchmark-latency.py`)
- [ ] **Barge-in:** p95 < 50ms (run `benchmark-latency.py --barge-in`)
- [ ] **Frame jitter:** p95 < 10ms (run `benchmark-jitter.py`)
- [ ] **Concurrent sessions:** 3+ per GPU without degradation (run `benchmark-concurrency.py`)
- [ ] **GPU memory:** < 90% usage under max load (`nvidia-smi dmon`)
- [ ] **Model preloading:** Default model loads in < 5s (`grep "Model loaded" logs`)
- [ ] **Control latency:** PAUSE acknowledged in < 20ms (`grep "PAUSE" logs`)
- [ ] **Network:** gRPC keepalive configured (`grep keepalive logs`)
- [ ] **Monitoring:** Prometheus metrics exported (`curl localhost:9090/metrics`)
- [ ] **Alerting:** SLO alerts configured in Prometheus

---

## 13. Further Reading

- **Configuration Reference:** `/home/gerald/git/full-duplex-voice-chat/docs/CONFIGURATION_REFERENCE.md`
- **Observability Guide:** `/home/gerald/git/full-duplex-voice-chat/docs/OBSERVABILITY.md`
- **Multi-GPU Deployment:** `/home/gerald/git/full-duplex-voice-chat/docs/MULTI_GPU.md`
- **Architecture:** `/home/gerald/git/full-duplex-voice-chat/docs/architecture/ARCHITECTURE.md`
- **gRPC Performance Best Practices:** https://grpc.io/docs/guides/performance/
- **PyTorch Performance Tuning:** https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- **CUDA Best Practices:** https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

---

**Last Updated:** 2025-10-05
**Target Milestone:** M11+ (Performance optimization features)
