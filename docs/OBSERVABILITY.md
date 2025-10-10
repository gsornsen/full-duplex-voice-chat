# Observability & Metrics Guide

This guide covers monitoring, logging, and distributed tracing for production deployments of the Realtime Duplex Voice Demo system.

---

## Overview

Production observability requires three pillars:

1. **Metrics:** Time-series data for performance and resource monitoring (Prometheus)
2. **Logs:** Structured event records for debugging and auditing (JSON logs)
3. **Traces:** Distributed request flows for latency analysis (OpenTelemetry)

**Target Milestones:**
- **M2 (Current):** Basic structured logging
- **M11:** Full Prometheus metrics and OpenTelemetry tracing
- **M12:** Production-ready dashboards and alerting

---

## Metrics (Prometheus)

### Metrics Catalog

#### Orchestrator Metrics

**Session Metrics:**

```python
# Counter: Total sessions started
orchestrator_sessions_started_total{
    transport="websocket",     # "websocket" or "livekit"
    model_id="cosyvoice2-en-base"
}

# Counter: Total sessions ended
orchestrator_sessions_ended_total{
    transport="websocket",
    model_id="cosyvoice2-en-base",
    reason="normal" | "error" | "client_disconnect" | "timeout"
}

# Gauge: Currently active sessions
orchestrator_sessions_active{
    transport="websocket",
    model_id="cosyvoice2-en-base"
}

# Counter: Session errors
orchestrator_session_errors_total{
    error_type="worker_unavailable" | "model_not_found" | "timeout" | "internal"
}
```

**Latency Metrics (Histograms):**

```python
# Histogram: First Audio Latency (ms)
orchestrator_latency_fal_ms_bucket{
    model_id="cosyvoice2-en-base",
    le="50" | "100" | "200" | "300" | "500" | "1000" | "+Inf"
}

# Histogram: Barge-in pause latency (ms)
orchestrator_latency_barge_in_ms_bucket{
    model_id="cosyvoice2-en-base",
    le="10" | "25" | "50" | "75" | "100" | "+Inf"
}

# Histogram: End-to-end latency (ms)
orchestrator_latency_e2e_ms_bucket{
    transport="websocket",
    le="100" | "200" | "400" | "600" | "1000" | "+Inf"
}
```

**Transport Metrics:**

```python
# Counter: Messages sent
orchestrator_websocket_messages_sent_total{
    message_type="SessionStarted" | "AudioFrame" | "SessionEnded" | "Error"
}

# Counter: Messages received
orchestrator_websocket_messages_received_total{
    message_type="SessionStart" | "TextChunk" | "Control"
}

# Histogram: Message processing time (ms)
orchestrator_message_processing_ms_bucket{
    message_type="TextChunk",
    le="1" | "5" | "10" | "50" | "100" | "+Inf"
}

# Gauge: Active WebSocket connections
orchestrator_websocket_connections_active{}
```

**VAD Metrics:**

```python
# Counter: VAD events
orchestrator_vad_events_total{
    event_type="speech_start" | "speech_end"
}

# Histogram: VAD detection latency (ms)
orchestrator_vad_detection_latency_ms_bucket{
    le="10" | "25" | "50" | "100" | "+Inf"
}
```

**ASR Metrics (M10+):**

```python
# Counter: ASR requests
orchestrator_asr_requests_total{
    model="openai/whisper-small"
}

# Histogram: ASR latency (ms)
orchestrator_asr_latency_ms_bucket{
    model="openai/whisper-small",
    le="50" | "100" | "200" | "500" | "+Inf"
}

# Counter: ASR errors
orchestrator_asr_errors_total{
    error_type="timeout" | "model_error"
}
```

#### Worker Metrics

**Session Metrics:**

```python
# Gauge: Active sessions per worker
worker_sessions_active{
    worker="tts-cosy@0",
    adapter_type="cosyvoice2"
}

# Counter: Synthesis requests
worker_synthesis_requests_total{
    worker="tts-cosy@0",
    model_id="cosyvoice2-en-base"
}

# Histogram: Synthesis duration (ms)
worker_synthesis_duration_ms_bucket{
    worker="tts-cosy@0",
    model_id="cosyvoice2-en-base",
    le="100" | "200" | "500" | "1000" | "2000" | "+Inf"
}
```

**Inference Metrics:**

```python
# Histogram: Inference latency (ms)
worker_inference_latency_ms_bucket{
    worker="tts-cosy@0",
    model_id="cosyvoice2-en-base",
    le="50" | "100" | "200" | "300" | "500" | "+Inf"
}

# Histogram: Real-time factor (RTF)
worker_inference_rtf_bucket{
    worker="tts-cosy@0",
    model_id="cosyvoice2-en-base",
    le="0.1" | "0.2" | "0.3" | "0.5" | "1.0" | "+Inf"
}

# Gauge: Inference queue depth
worker_inference_queue_depth{
    worker="tts-cosy@0"
}
```

**Audio Metrics:**

```python
# Counter: Audio frames emitted
worker_audio_frames_emitted_total{
    worker="tts-cosy@0",
    session_id="session-abc123"
}

# Histogram: Frame jitter (ms)
worker_audio_frame_jitter_ms_bucket{
    worker="tts-cosy@0",
    le="1" | "5" | "10" | "20" | "50" | "+Inf"
}

# Counter: Frame drops
worker_audio_frame_drops_total{
    worker="tts-cosy@0",
    reason="backpressure" | "pause" | "error"
}
```

**Model Manager Metrics:**

```python
# Gauge: Resident models count
worker_model_manager_resident_count{
    worker="tts-cosy@0"
}

# Counter: Model loads
worker_model_manager_loads_total{
    worker="tts-cosy@0",
    model_id="cosyvoice2-en-base"
}

# Counter: Model unloads
worker_model_manager_unloads_total{
    worker="tts-cosy@0",
    model_id="cosyvoice2-en-base",
    reason="ttl_eviction" | "lru_eviction" | "manual"
}

# Histogram: Model load duration (ms)
worker_model_manager_load_duration_ms_bucket{
    worker="tts-cosy@0",
    model_id="cosyvoice2-en-base",
    le="1000" | "2000" | "5000" | "10000" | "+Inf"
}

# Histogram: Model warmup duration (ms)
worker_model_manager_warmup_duration_ms_bucket{
    worker="tts-cosy@0",
    model_id="cosyvoice2-en-base",
    le="100" | "300" | "500" | "1000" | "+Inf"
}
```

**GPU Metrics:**

```python
# Gauge: GPU memory used (bytes)
worker_gpu_memory_used_bytes{
    worker="tts-cosy@0",
    gpu_id="0"
}

# Gauge: GPU memory total (bytes)
worker_gpu_memory_total_bytes{
    gpu_id="0"
}

# Gauge: GPU utilization (percent)
worker_gpu_utilization_percent{
    worker="tts-cosy@0",
    gpu_id="0"
}

# Gauge: GPU temperature (Celsius)
worker_gpu_temperature_celsius{
    gpu_id="0"
}

# Gauge: GPU power draw (watts)
worker_gpu_power_watts{
    gpu_id="0"
}
```

**Control Metrics:**

```python
# Counter: Control commands received
worker_control_commands_total{
    worker="tts-cosy@0",
    command="PAUSE" | "RESUME" | "STOP"
}

# Histogram: Control command latency (ms)
worker_control_latency_ms_bucket{
    worker="tts-cosy@0",
    command="PAUSE",
    le="5" | "10" | "20" | "50" | "+Inf"
}
```

#### Redis Metrics

```python
# Gauge: Active Redis connections
redis_connections_active{}

# Histogram: Redis command latency (ms)
redis_command_latency_ms_bucket{
    command="GET" | "SET" | "HGETALL" | "KEYS",
    le="1" | "5" | "10" | "50" | "100" | "+Inf"
}

# Counter: Redis command errors
redis_command_errors_total{
    command="GET",
    error_type="timeout" | "connection_error"
}
```

---

### Metrics Instrumentation

#### Orchestrator Instrumentation

**File:** `src/orchestrator/metrics.py`

```python
from prometheus_client import Counter, Histogram, Gauge, Info

# Session metrics
sessions_started = Counter(
    'orchestrator_sessions_started_total',
    'Total sessions started',
    ['transport', 'model_id']
)

sessions_ended = Counter(
    'orchestrator_sessions_ended_total',
    'Total sessions ended',
    ['transport', 'model_id', 'reason']
)

sessions_active = Gauge(
    'orchestrator_sessions_active',
    'Currently active sessions',
    ['transport', 'model_id']
)

# Latency metrics
fal_histogram = Histogram(
    'orchestrator_latency_fal_ms',
    'First Audio Latency in milliseconds',
    ['model_id'],
    buckets=[50, 100, 200, 300, 500, 1000, float('inf')]
)

barge_in_histogram = Histogram(
    'orchestrator_latency_barge_in_ms',
    'Barge-in pause latency in milliseconds',
    ['model_id'],
    buckets=[10, 25, 50, 75, 100, float('inf')]
)

# Usage in server.py
async def handle_session_start(session_id: str, model_id: str):
    sessions_started.labels(transport='websocket', model_id=model_id).inc()
    sessions_active.labels(transport='websocket', model_id=model_id).inc()

    start_time = time.monotonic()

    # ... synthesis logic ...

    # Record FAL
    fal_ms = (first_audio_time - start_time) * 1000
    fal_histogram.labels(model_id=model_id).observe(fal_ms)

async def handle_session_end(session_id: str, model_id: str, reason: str):
    sessions_ended.labels(transport='websocket', model_id=model_id, reason=reason).inc()
    sessions_active.labels(transport='websocket', model_id=model_id).dec()
```

#### Worker Instrumentation

**File:** `src/tts/metrics.py`

```python
from prometheus_client import Counter, Histogram, Gauge
import pynvml

# Inference metrics
inference_latency = Histogram(
    'worker_inference_latency_ms',
    'Model inference latency in milliseconds',
    ['worker', 'model_id'],
    buckets=[50, 100, 200, 300, 500, float('inf')]
)

inference_rtf = Histogram(
    'worker_inference_rtf',
    'Real-time factor (inference_time / audio_duration)',
    ['worker', 'model_id'],
    buckets=[0.1, 0.2, 0.3, 0.5, 1.0, float('inf')]
)

# GPU metrics
gpu_memory_used = Gauge(
    'worker_gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['worker', 'gpu_id']
)

gpu_utilization = Gauge(
    'worker_gpu_utilization_percent',
    'GPU utilization percentage',
    ['worker', 'gpu_id']
)

# Usage in worker.py
async def synthesize(text: str, model_id: str):
    start_time = time.monotonic()

    # Inference
    audio = model.generate(text)

    # Record latency
    latency_ms = (time.monotonic() - start_time) * 1000
    inference_latency.labels(worker=worker_name, model_id=model_id).observe(latency_ms)

    # Calculate RTF
    audio_duration = len(audio) / sample_rate
    rtf = latency_ms / 1000.0 / audio_duration
    inference_rtf.labels(worker=worker_name, model_id=model_id).observe(rtf)

# GPU metrics collection (background task)
async def collect_gpu_metrics(worker_name: str, gpu_id: int):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

    while True:
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

        gpu_memory_used.labels(worker=worker_name, gpu_id=str(gpu_id)).set(memory_info.used)
        gpu_utilization.labels(worker=worker_name, gpu_id=str(gpu_id)).set(utilization)

        await asyncio.sleep(5)  # Update every 5s
```

---

### Metrics Exporter

**Prometheus HTTP Exporter:**

```python
# In src/orchestrator/server.py
from prometheus_client import start_http_server

# Start Prometheus metrics server on port 9090
start_http_server(9090)

logger.info("Prometheus metrics available at http://localhost:9090/metrics")
```

**Access metrics:**

```bash
curl http://localhost:9090/metrics

# Sample output:
# # HELP orchestrator_sessions_started_total Total sessions started
# # TYPE orchestrator_sessions_started_total counter
# orchestrator_sessions_started_total{model_id="cosyvoice2-en-base",transport="websocket"} 142.0
#
# # HELP orchestrator_latency_fal_ms First Audio Latency in milliseconds
# # TYPE orchestrator_latency_fal_ms histogram
# orchestrator_latency_fal_ms_bucket{le="50.0",model_id="cosyvoice2-en-base"} 5.0
# orchestrator_latency_fal_ms_bucket{le="100.0",model_id="cosyvoice2-en-base"} 28.0
# orchestrator_latency_fal_ms_bucket{le="200.0",model_id="cosyvoice2-en-base"} 95.0
# orchestrator_latency_fal_ms_bucket{le="300.0",model_id="cosyvoice2-en-base"} 135.0
# orchestrator_latency_fal_ms_bucket{le="+Inf",model_id="cosyvoice2-en-base"} 142.0
# orchestrator_latency_fal_ms_sum{model_id="cosyvoice2-en-base"} 28450.0
# orchestrator_latency_fal_ms_count{model_id="cosyvoice2-en-base"} 142.0
```

---

### Prometheus Configuration

**File:** `config/prometheus.yml`

```yaml
global:
  scrape_interval: 15s      # Scrape metrics every 15s
  evaluation_interval: 15s  # Evaluate alerting rules every 15s

scrape_configs:
  # Orchestrator metrics
  - job_name: 'orchestrator'
    static_configs:
      - targets: ['localhost:9090']
        labels:
          service: 'orchestrator'
          environment: 'production'

  # Worker metrics (multi-GPU)
  - job_name: 'tts-workers'
    static_configs:
      - targets:
          - 'localhost:9091'  # Worker GPU 0
          - 'localhost:9092'  # Worker GPU 1
        labels:
          service: 'tts-worker'
          environment: 'production'

  # Redis exporter
  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
        labels:
          service: 'redis'

# Alerting rules (see Alerting section below)
rule_files:
  - 'alerts.yml'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']
```

**Run Prometheus:**

```bash
docker run -d --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/config/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

---

## Logging

### Structured Logging Format

All logs use JSON format for machine parsing and centralized aggregation.

**Log Schema:**

```json
{
  "timestamp": "2025-10-05T14:32:15.123456Z",
  "level": "INFO",
  "logger": "orchestrator.server",
  "message": "Session started",
  "session_id": "session-abc123",
  "model_id": "cosyvoice2-en-base",
  "client_ip": "192.168.1.100",
  "transport": "websocket",
  "trace_id": "trace-xyz789",
  "span_id": "span-001"
}
```

**Error Log Schema:**

```json
{
  "timestamp": "2025-10-05T14:35:42.987654Z",
  "level": "ERROR",
  "logger": "orchestrator.grpc_client",
  "message": "Failed to connect to worker",
  "session_id": "session-def456",
  "error_type": "ConnectionError",
  "error_message": "Worker tts-cosy@0 not reachable at grpc://localhost:7001",
  "traceback": "Traceback (most recent call last)...",
  "resolution": "See docs/runbooks/GRPC_WORKER.md",
  "trace_id": "trace-abc456"
}
```

### Logging Configuration

**File:** `src/utils/logging.py`

```python
import logging
import json
import sys
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    """Format logs as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add session_id if present
        if hasattr(record, 'session_id'):
            log_data['session_id'] = record.session_id

        # Add trace context if present (OpenTelemetry)
        if hasattr(record, 'trace_id'):
            log_data['trace_id'] = record.trace_id
            log_data['span_id'] = record.span_id

        # Add exception info
        if record.exc_info:
            log_data['error_type'] = record.exc_info[0].__name__
            log_data['error_message'] = str(record.exc_info[1])
            log_data['traceback'] = self.formatException(record.exc_info)

        # Add custom fields
        for key in ['model_id', 'client_ip', 'transport', 'worker', 'resolution']:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        return json.dumps(log_data)

def setup_logging(log_level: str = "INFO"):
    """Configure structured JSON logging."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)

# Usage in server.py
from src.utils.logging import setup_logging

setup_logging(log_level="INFO")
logger = logging.getLogger("orchestrator.server")

logger.info(
    "Session started",
    extra={
        'session_id': session_id,
        'model_id': model_id,
        'client_ip': client_ip,
        'transport': 'websocket'
    }
)
```

### Log Levels

**DEBUG:**
- Detailed debugging information (frame timings, model outputs)
- Enabled only in development

**INFO:**
- Session lifecycle events (start, end)
- Model loading/unloading
- Health check results

**WARNING:**
- Retryable errors (worker temporary unavailable)
- Performance degradation (high latency, jitter)
- Non-critical configuration issues

**ERROR:**
- Failed requests (model not found, timeout)
- Connection failures (gRPC, Redis)
- Recoverable exceptions

**CRITICAL:**
- System failures (worker crash, Redis down)
- Data corruption
- Unrecoverable errors

### Log Redaction (PII Protection)

```python
import re

class RedactingFormatter(JSONFormatter):
    """Redact sensitive information from logs."""

    REDACT_PATTERNS = [
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL]'),
        (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN]'),
        (re.compile(r'/voicepacks/[^/]+/[^/]+/ref/'), '/voicepacks/[REDACTED]/'),
    ]

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        for pattern, replacement in self.REDACT_PATTERNS:
            message = pattern.sub(replacement, message)

        record.msg = message
        return super().format(record)
```

---

## Distributed Tracing (OpenTelemetry)

### Tracing Overview

Distributed tracing tracks requests across services to identify latency bottlenecks.

**Example Trace:**

```
[Client Request] → [Orchestrator] → [Worker] → [Model Inference] → [Audio Framing] → [Response]
    |                  |                |              |                    |
    10ms              15ms            200ms          20ms                5ms
Total: 250ms (FAL)
```

### OpenTelemetry Setup

**Installation:**

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-grpc
```

**Configuration:**

```python
# src/utils/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient, GrpcInstrumentorServer

def setup_tracing(service_name: str):
    """Configure OpenTelemetry tracing with Jaeger exporter."""

    # Create tracer provider
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()

    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )

    # Add span processor
    tracer_provider.add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )

    # Instrument gRPC
    GrpcInstrumentorClient().instrument()
    GrpcInstrumentorServer().instrument()

    return trace.get_tracer(service_name)

# Usage in orchestrator
from src.utils.tracing import setup_tracing

tracer = setup_tracing("orchestrator")

async def handle_synthesize_request(text: str, model_id: str):
    with tracer.start_as_current_span("synthesize_request") as span:
        span.set_attribute("model_id", model_id)
        span.set_attribute("text_length", len(text))

        # Call worker
        with tracer.start_as_current_span("grpc_call_worker"):
            audio = await worker_client.synthesize(text, model_id)

        # Process audio
        with tracer.start_as_current_span("audio_processing"):
            frames = repacketize_to_frames(audio)

        span.set_attribute("frames_count", len(frames))
        return frames
```

### Trace Visualization

**Run Jaeger:**

```bash
docker run -d --name jaeger \
  -p 6831:6831/udp \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

**Access UI:**

```
http://localhost:16686
```

**Example Trace View:**

```
Service: orchestrator
  Span: synthesize_request (250ms)
    ├─ Span: grpc_call_worker (220ms)
    │   └─ Service: tts-worker
    │       Span: Synthesize (215ms)
    │         ├─ Span: model_inference (200ms)
    │         └─ Span: audio_framing (15ms)
    └─ Span: audio_processing (10ms)
```

---

## Dashboards (Grafana)

### Grafana Setup

```bash
docker run -d --name grafana \
  -p 3000:3000 \
  -v grafana-storage:/var/lib/grafana \
  grafana/grafana-oss
```

Access: `http://localhost:3000` (default: admin/admin)

### Dashboard: Realtime Performance

**Panels:**

1. **Current Active Sessions** (Gauge)
   - Query: `orchestrator_sessions_active`
   - Threshold: Yellow > 5, Red > 10

2. **First Audio Latency (p50/p95/p99)** (Graph)
   - Query:
     ```promql
     histogram_quantile(0.50, rate(orchestrator_latency_fal_ms_bucket[5m]))
     histogram_quantile(0.95, rate(orchestrator_latency_fal_ms_bucket[5m]))
     histogram_quantile(0.99, rate(orchestrator_latency_fal_ms_bucket[5m]))
     ```
   - Target line: 300ms (p95 SLO)

3. **Barge-in Latency (p95)** (Graph)
   - Query:
     ```promql
     histogram_quantile(0.95, rate(orchestrator_latency_barge_in_ms_bucket[5m]))
     ```
   - Target line: 50ms (p95 SLO)

4. **Frame Jitter (p95)** (Graph)
   - Query:
     ```promql
     histogram_quantile(0.95, rate(worker_audio_frame_jitter_ms_bucket[5m]))
     ```
   - Target line: 10ms (p95 SLO)

5. **Session Start Rate** (Graph)
   - Query: `rate(orchestrator_sessions_started_total[1m])`

6. **Error Rate** (Graph)
   - Query: `rate(orchestrator_session_errors_total[1m])`

### Dashboard: Resource Utilization

**Panels:**

1. **GPU Memory Usage** (Graph, stacked area)
   - Query: `worker_gpu_memory_used_bytes / worker_gpu_memory_total_bytes * 100`
   - Per worker, stacked

2. **GPU Utilization** (Graph)
   - Query: `worker_gpu_utilization_percent`
   - Per GPU

3. **Inference Queue Depth** (Graph)
   - Query: `worker_inference_queue_depth`
   - Per worker

4. **Model Manager Resident Count** (Graph)
   - Query: `worker_model_manager_resident_count`
   - Per worker

5. **Redis Command Latency (p95)** (Graph)
   - Query: `histogram_quantile(0.95, rate(redis_command_latency_ms_bucket[5m]))`

### Dashboard JSON Export

See `/home/gerald/git/full-duplex-voice-chat/config/grafana/dashboards/realtime-performance.json` (M11+)

---

## Alerting Rules

### Prometheus Alerting

**File:** `config/prometheus/alerts.yml`

```yaml
groups:
  - name: voice_demo_slos
    interval: 30s
    rules:
      # Critical: FAL exceeds SLO
      - alert: HighFirstAudioLatency
        expr: histogram_quantile(0.95, rate(orchestrator_latency_fal_ms_bucket[5m])) > 300
        for: 5m
        labels:
          severity: warning
          slo: fal
        annotations:
          summary: "p95 First Audio Latency exceeds 300ms SLO"
          description: "Current p95 FAL: {{ $value }}ms (target: <300ms)"
          runbook: "docs/runbooks/PERFORMANCE.md"

      # Critical: Barge-in latency exceeds SLO
      - alert: HighBargeInLatency
        expr: histogram_quantile(0.95, rate(orchestrator_latency_barge_in_ms_bucket[5m])) > 50
        for: 5m
        labels:
          severity: warning
          slo: barge_in
        annotations:
          summary: "p95 Barge-in latency exceeds 50ms SLO"
          description: "Current p95 barge-in latency: {{ $value }}ms (target: <50ms)"
          runbook: "docs/PERFORMANCE.md#3-barge-in-latency-optimization"

      # Critical: Frame jitter exceeds SLO
      - alert: HighFrameJitter
        expr: histogram_quantile(0.95, rate(worker_audio_frame_jitter_ms_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
          slo: jitter
        annotations:
          summary: "p95 Frame jitter exceeds 10ms SLO"
          description: "Current p95 jitter: {{ $value }}ms (target: <10ms)"
          runbook: "docs/PERFORMANCE.md#2-frame-jitter-reduction"

      # Critical: GPU memory exhaustion
      - alert: GPUMemoryExhaustion
        expr: worker_gpu_memory_used_bytes / worker_gpu_memory_total_bytes > 0.95
        for: 2m
        labels:
          severity: critical
          component: worker
        annotations:
          summary: "GPU memory usage > 95%"
          description: "Worker {{ $labels.worker }} GPU {{ $labels.gpu_id }}: {{ $value | humanizePercentage }} used"
          runbook: "docs/PERFORMANCE.md#5-gpu-memory-optimization"

      # Warning: High error rate
      - alert: HighSessionErrorRate
        expr: rate(orchestrator_session_errors_total[5m]) > 0.1
        for: 3m
        labels:
          severity: warning
          component: orchestrator
        annotations:
          summary: "Session error rate > 10%"
          description: "Current error rate: {{ $value | humanize }} errors/sec"
          runbook: "docs/runbooks/ADVANCED_TROUBLESHOOTING.md"

      # Warning: Worker unavailable
      - alert: WorkerUnavailable
        expr: up{job="tts-workers"} == 0
        for: 1m
        labels:
          severity: critical
          component: worker
        annotations:
          summary: "TTS Worker down"
          description: "Worker {{ $labels.instance }} is not responding to health checks"
          runbook: "docs/runbooks/GRPC_WORKER.md"

      # Warning: Redis connection issues
      - alert: RedisConnectionErrors
        expr: rate(redis_command_errors_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
          component: redis
        annotations:
          summary: "Redis connection errors > 5%"
          description: "Current error rate: {{ $value | humanize }} errors/sec"
          runbook: "docs/runbooks/REDIS.md"
```

### Alertmanager Configuration

**File:** `config/prometheus/alertmanager.yml`

```yaml
global:
  resolve_timeout: 5m

route:
  receiver: 'default'
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h

  routes:
    # Critical alerts: immediate notification
    - receiver: 'pagerduty'
      match:
        severity: critical
      continue: true

    # Warning alerts: Slack notification
    - receiver: 'slack'
      match:
        severity: warning

receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://localhost:5001/alerts'

  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#voice-demo-alerts'
        title: '{{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
```

---

## Performance SLIs/SLOs

### Service Level Indicators (SLIs)

| SLI | Measurement | Target (SLO) |
|-----|-------------|--------------|
| First Audio Latency (FAL) | p95 latency from first text chunk to first audio frame | < 300ms (GPU), < 500ms (CPU) |
| Barge-in Latency | p95 latency from VAD trigger to worker pause | < 50ms |
| Frame Jitter | p95 variance in frame emission timing | < 10ms |
| Availability | % of requests successfully processed | > 99.5% |
| Error Rate | % of requests resulting in error | < 0.5% |

### Service Level Objectives (SLOs)

**Availability SLO:** 99.5% (3.6 hours downtime/month)

**Latency SLO:**
- 95% of requests complete with FAL < 300ms (GPU)
- 95% of barge-in events pause within 50ms

**Error Budget:**
- 0.5% of requests may fail (≈ 216 requests/month @ 1 req/min)

### SLO Monitoring

```promql
# Availability (last 30 days)
(
  sum(rate(orchestrator_sessions_started_total[30d]))
  - sum(rate(orchestrator_session_errors_total[30d]))
) / sum(rate(orchestrator_sessions_started_total[30d])) * 100

# Target: > 99.5%

# FAL SLO compliance (last 7 days)
histogram_quantile(0.95, rate(orchestrator_latency_fal_ms_bucket[7d])) < 300

# Barge-in SLO compliance (last 7 days)
histogram_quantile(0.95, rate(orchestrator_latency_barge_in_ms_bucket[7d])) < 50
```

---

## Troubleshooting Workflows

### High Latency Investigation

**Step 1: Check metrics**

```bash
# Query p95 FAL
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(orchestrator_latency_fal_ms_bucket[5m]))' | jq

# Expected: < 300ms
```

**Step 2: Identify bottleneck via traces**

Open Jaeger UI → Search for slow traces → Identify longest span

**Step 3: Review logs**

```bash
# Find slow requests
grep '"level":"WARNING"' /var/log/orchestrator.log | grep "high_latency"

# Example log:
# {"level":"WARNING","message":"High FAL detected","session_id":"session-abc","fal_ms":520,"model_id":"cosyvoice2-en-base"}
```

**Step 4: Apply fix**

See `/home/gerald/git/full-duplex-voice-chat/docs/PERFORMANCE.md`

### High Error Rate Investigation

**Step 1: Check error types**

```promql
sum by (error_type) (rate(orchestrator_session_errors_total[5m]))
```

**Step 2: Review error logs**

```bash
grep '"level":"ERROR"' /var/log/orchestrator.log | tail -n 20
```

**Step 3: Apply resolution**

See error-specific runbooks in `/home/gerald/git/full-duplex-voice-chat/docs/runbooks/`

---

## Further Reading

- **Performance Tuning:** `/home/gerald/git/full-duplex-voice-chat/docs/PERFORMANCE.md`
- **Runbooks:** `/home/gerald/git/full-duplex-voice-chat/docs/runbooks/`
- **Prometheus Documentation:** https://prometheus.io/docs/
- **Grafana Documentation:** https://grafana.com/docs/
- **OpenTelemetry Documentation:** https://opentelemetry.io/docs/
- **Monitoring Runbook:** Reference @incident-responder's P2-1 (when available)

---

**Last Updated:** 2025-10-05
**Target Milestone:** M11 (Full observability), M12 (Production dashboards)
