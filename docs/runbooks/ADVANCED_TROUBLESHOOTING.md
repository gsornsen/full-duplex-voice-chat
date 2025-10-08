# Runbook: Advanced Troubleshooting

**Time to Resolution:** 15-60 minutes (depending on complexity)
**Severity:** Medium-High (complex issues)
**Related:** [Log Debugging](LOG_DEBUGGING.md), [Monitoring](MONITORING.md), [Audio Quality](AUDIO_QUALITY.md)

---

## Overview

This runbook covers advanced diagnostic techniques for complex, multi-component failures in the M2 TTS system. Use these techniques when standard runbooks don't resolve the issue.

**Prerequisites:**
- Familiarity with basic runbooks
- Understanding of system architecture
- Access to all system components

---

## System-Wide Diagnostics

### Full Health Check

**Comprehensive system validation:**

```bash
#!/bin/bash
# full-health-check.sh

echo "=== Full System Health Check ==="

# 1. Container status
echo -e "\n[1] Container Status:"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 2. Resource usage
echo -e "\n[2] Resource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# 3. Network connectivity
echo -e "\n[3] Network Connectivity:"
docker network ls
docker network inspect tts-network | jq -r '.[].Containers | to_entries[] | "\(.value.Name): \(.value.IPv4Address)"'

# 4. Service health endpoints
echo -e "\n[4] Service Health:"
echo -n "Orchestrator: "
curl -sf http://localhost:8081/health && echo "✅" || echo "❌"
echo -n "Worker metrics: "
curl -sf http://localhost:9090/metrics > /dev/null && echo "✅" || echo "❌"
echo -n "Redis: "
redis-cli -u redis://localhost:6379 ping && echo "✅" || echo "❌"
echo -n "Prometheus: "
curl -sf http://localhost:9090/-/healthy > /dev/null && echo "✅" || echo "❌"

# 5. Disk space
echo -e "\n[5] Disk Space:"
df -h | grep -E "Filesystem|/dev/|overlay"

# 6. Recent errors (last 5 minutes)
echo -e "\n[6] Recent Errors:"
for container in orchestrator tts-worker; do
  echo -e "\n  $container:"
  docker logs $container --since 5m 2>&1 | grep -i error | tail -5
done

# 7. Active connections
echo -e "\n[7] Active Connections:"
netstat -tuln | grep -E "8080|7001|6379|9090"

echo -e "\n=== Health Check Complete ==="
```

**Save and run:**
```bash
chmod +x scripts/full-health-check.sh
./scripts/full-health-check.sh
```

---

### Distributed Tracing

**Trace request across components:**

**1. Client → Orchestrator → Worker → Client**

```bash
# Enable trace logging
export LOG_LEVEL=DEBUG

# Start services with tracing
docker compose restart orchestrator tts-worker
```

**2. Generate trace ID:**
```python
# Client sends trace ID
import uuid

trace_id = str(uuid.uuid4())
ws.send(json.dumps({
    'text': 'Hello world',
    'trace_id': trace_id
}))
```

**3. Extract trace from logs:**
```bash
TRACE_ID="abc-123-def"

# Orchestrator logs
docker logs orchestrator | jq "select(.trace_id == \"$TRACE_ID\") | {timestamp, component, message}"

# Worker logs
docker logs tts-worker | jq "select(.trace_id == \"$TRACE_ID\") | {timestamp, component, message}"
```

**4. Reconstruct timeline:**
```bash
# Combined timeline
(docker logs orchestrator && docker logs tts-worker) | \
  jq "select(.trace_id == \"$TRACE_ID\") | {timestamp, service: .container_name, event: .message}" | \
  jq -s 'sort_by(.timestamp)'
```

---

## Performance Profiling

### CPU Profiling

**Using py-spy:**

```bash
# Install py-spy
pip install py-spy

# Find worker PID
WORKER_PID=$(docker exec tts-worker ps aux | grep "worker.py" | awk '{print $2}' | head -1)

# Top-like live view
docker exec tts-worker py-spy top --pid $WORKER_PID

# Record flame graph (30 seconds)
docker exec tts-worker py-spy record -o profile.svg --pid $WORKER_PID --duration 30

# Copy out of container
docker cp tts-worker:/app/profile.svg ./profile.svg
```

**Analyze hotspots:**
- Open `profile.svg` in browser
- Look for wide bars (high CPU time)
- Identify bottleneck functions

---

### Memory Profiling

**Using memory_profiler:**

```python
# Add to code temporarily
from memory_profiler import profile

@profile
def synthesize_text(text: str):
    # ... synthesis code ...
    pass
```

**Run and analyze:**
```bash
# Run with memory profiling
uv run python -m memory_profiler src/tts/worker.py

# Output shows memory usage per line
Line #    Mem usage    Increment   Line Contents
================================================
    10    100.5 MiB      0.0 MiB   def synthesize_text(text):
    11    200.3 MiB     99.8 MiB       model = load_model()  # <-- High allocation
```

**Docker container memory:**
```bash
# Monitor memory over time
docker stats tts-worker --no-stream --format "{{.MemUsage}}"

# Memory limit hit
docker inspect tts-worker | jq '.[0].HostConfig.Memory'
docker inspect tts-worker | jq '.[0].State.OOMKilled'
```

---

### GPU Profiling

**Using nvidia-smi:**

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log GPU usage
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 1 > gpu-usage.csv

# GPU process info
nvidia-smi pmon -c 10 -s u
```

**Using PyTorch Profiler:**

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output = model.synthesize(text)

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export trace for Chrome
prof.export_chrome_trace("trace.json")
# Open in chrome://tracing
```

**NVIDIA Nsight Systems:**

```bash
# Profile worker
nsys profile -o worker-profile \
  docker exec tts-worker python src/tts/worker.py

# View in Nsight Systems GUI
nsys-ui worker-profile.nsys-rep
```

---

## Network Debugging

### Packet Capture

**Capture WebSocket traffic:**

```bash
# Install tcpdump in container
docker exec orchestrator apt-get update && apt-get install -y tcpdump

# Capture on port 8080
docker exec orchestrator tcpdump -i any -w /tmp/capture.pcap port 8080

# Transfer out
docker cp orchestrator:/tmp/capture.pcap ./capture.pcap

# Analyze with Wireshark
wireshark capture.pcap
```

**Filter WebSocket frames:**
- Wireshark filter: `websocket`
- Look for: connection establishment, frame timing, errors

---

### gRPC Traffic Analysis

**Enable gRPC tracing:**

```python
# orchestrator/grpc_client.py
import grpc
import logging

# Enable gRPC debug logging
logging.basicConfig()
logging.getLogger('grpc').setLevel(logging.DEBUG)

# Or environment variable
export GRPC_TRACE=all
export GRPC_VERBOSITY=DEBUG
```

**Capture gRPC traffic:**

```bash
# Capture on gRPC port
docker exec orchestrator tcpdump -i any -w /tmp/grpc.pcap port 7001

# Analyze with gRPC tools
grpcurl -plaintext localhost:7001 list
grpcurl -plaintext localhost:7001 describe TTSService
```

---

### Network Latency Measurement

**Measure RTT (round-trip time):**

```python
# Client-side measurement
import time
import asyncio

async def measure_latency():
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        await ws.send(json.dumps({'type': 'ping'}))
        response = await ws.recv()
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        await asyncio.sleep(0.1)

    print(f"Average: {sum(latencies)/len(latencies):.2f}ms")
    print(f"p95: {sorted(latencies)[95]:.2f}ms")
    print(f"p99: {sorted(latencies)[99]:.2f}ms")
```

**Docker network inspection:**

```bash
# Check network mode
docker inspect orchestrator | jq '.[0].HostConfig.NetworkMode'

# Check network settings
docker network inspect tts-network | jq '.[0].IPAM.Config'

# Test inter-container latency
docker exec orchestrator ping -c 10 tts-worker
```

---

## Database Debugging (Redis)

### Redis Slow Log

**Check slow queries:**

```bash
# Get slow log
redis-cli -u redis://localhost:6379 slowlog get 10

# Configure slow log threshold (microseconds)
redis-cli -u redis://localhost:6379 config set slowlog-log-slower-than 10000

# Reset slow log
redis-cli -u redis://localhost:6379 slowlog reset
```

---

### Redis Memory Analysis

**Memory usage:**

```bash
# Total memory
redis-cli -u redis://localhost:6379 info memory

# Memory by key pattern
redis-cli -u redis://localhost:6379 --bigkeys

# Scan for specific pattern
redis-cli -u redis://localhost:6379 --scan --pattern "worker:*" | wc -l
```

**Memory leaks:**

```bash
# Monitor memory over time
while true; do
  redis-cli -u redis://localhost:6379 info memory | grep used_memory_human
  sleep 60
done
```

---

### Redis Connection Pool

**Check active connections:**

```bash
# Connected clients
redis-cli -u redis://localhost:6379 client list

# Connection count
redis-cli -u redis://localhost:6379 info clients

# Monitor connections in real-time
redis-cli -u redis://localhost:6379 monitor
```

---

## Concurrency Issues

### Race Conditions

**Detect race conditions:**

```python
# Add assertions to detect races
import threading

_lock = threading.Lock()
_counter = 0

def increment():
    global _counter
    local = _counter
    # Simulate race window
    import time; time.sleep(0.001)
    _counter = local + 1

# Run multiple threads
threads = [threading.Thread(target=increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Expected: 10, Actual: may be less (race condition)
print(_counter)
```

**Fix with proper locking:**

```python
import asyncio

# Async lock
_lock = asyncio.Lock()

async def safe_increment():
    async with _lock:
        global _counter
        _counter += 1
```

---

### Deadlock Detection

**Identify deadlocks:**

```bash
# Python deadlock detection
pip install deadlock-detection

# Add to code
import deadlock_detection
deadlock_detection.enable()

# Deadlock detected → prints stack traces
```

**Analyze thread dumps:**

```python
import sys
import traceback

def dump_threads():
    """Print all thread stack traces."""
    for thread_id, frame in sys._current_frames().items():
        print(f"\n--- Thread {thread_id} ---")
        traceback.print_stack(frame)

# Call on SIGUSR1
import signal
signal.signal(signal.SIGUSR1, lambda sig, frame: dump_threads())
```

**Trigger dump:**

```bash
# Find process PID
PID=$(docker exec tts-worker ps aux | grep worker.py | awk '{print $2}')

# Send signal
docker exec tts-worker kill -SIGUSR1 $PID

# Check logs
docker logs tts-worker --tail=100
```

---

## Data Corruption Issues

### Audio Frame Validation

**Validate frame integrity:**

```python
def validate_audio_frame(frame: bytes, expected_size: int = 1920):
    """Validate audio frame data."""
    # Check size (960 samples * 2 bytes/sample)
    if len(frame) != expected_size:
        raise ValueError(f"Frame size {len(frame)} != {expected_size}")

    # Convert to samples
    import struct
    samples = struct.unpack(f'{len(frame)//2}h', frame)

    # Check for clipping
    clipped = sum(1 for s in samples if abs(s) >= 32767)
    if clipped > len(samples) * 0.01:  # > 1% clipped
        print(f"Warning: {clipped}/{len(samples)} samples clipped")

    # Check for silence
    if all(abs(s) < 100 for s in samples):
        print("Warning: Frame appears to be silent")

    # Check for DC offset
    mean = sum(samples) / len(samples)
    if abs(mean) > 100:
        print(f"Warning: DC offset detected: {mean}")

    return True
```

---

### Checksum Validation

**Add checksums to frames:**

```python
import hashlib

def frame_with_checksum(audio_data: bytes) -> dict:
    """Add checksum to audio frame."""
    checksum = hashlib.md5(audio_data).hexdigest()
    return {
        'data': audio_data,
        'checksum': checksum,
        'size': len(audio_data)
    }

def verify_frame(frame: dict) -> bool:
    """Verify frame checksum."""
    expected = hashlib.md5(frame['data']).hexdigest()
    if expected != frame['checksum']:
        raise ValueError(f"Checksum mismatch: {expected} != {frame['checksum']}")
    return True
```

---

## System Resource Exhaustion

### File Descriptor Leaks

**Check open files:**

```bash
# List open files
docker exec orchestrator lsof -p $(docker exec orchestrator ps aux | grep server.py | awk '{print $2}')

# Count by type
docker exec orchestrator lsof -p $(docker exec orchestrator ps aux | grep server.py | awk '{print $2}') | \
  awk '{print $5}' | sort | uniq -c

# Monitor over time
watch -n 5 'docker exec orchestrator lsof -p $(docker exec orchestrator ps aux | grep server.py | awk "{print \$2}") | wc -l'
```

**System limits:**

```bash
# Check ulimit
docker exec orchestrator sh -c 'ulimit -n'

# Increase limit
docker run --ulimit nofile=65536:65536 ...
```

---

### Connection Pool Exhaustion

**Monitor connection pools:**

```python
# Redis pool monitoring
async def monitor_redis_pool(pool):
    while True:
        print(f"Pool size: {pool.size}")
        print(f"Available: {pool.available}")
        print(f"In use: {pool.size - pool.available}")
        await asyncio.sleep(10)
```

**Fix pool leaks:**

```python
# Bad: Connection not returned
async def leaky_function():
    conn = await pool.acquire()
    result = await conn.get('key')
    # Missing: await pool.release(conn)
    return result

# Good: Use context manager
async def safe_function():
    async with pool.acquire() as conn:
        result = await conn.get('key')
        return result  # Connection auto-released
```

---

## Advanced Log Analysis

### Log Correlation

**Correlate logs across services:**

```bash
#!/bin/bash
# correlate-logs.sh

SESSION_ID="$1"

echo "=== Orchestrator Events ==="
docker logs orchestrator | jq "select(.session_id == \"$SESSION_ID\") | {timestamp, event: .message}"

echo -e "\n=== Worker Events ==="
docker logs tts-worker | jq "select(.session_id == \"$SESSION_ID\") | {timestamp, event: .message}"

echo -e "\n=== Timeline ==="
(docker logs orchestrator && docker logs tts-worker) | \
  jq "select(.session_id == \"$SESSION_ID\") | {timestamp, service: .container_name, event: .message}" | \
  jq -s 'sort_by(.timestamp)'
```

---

### Anomaly Detection

**Detect statistical anomalies:**

```python
import numpy as np

def detect_anomalies(values: list[float], threshold: float = 3.0) -> list[int]:
    """Detect anomalies using standard deviation.

    Args:
        values: List of metric values
        threshold: Number of standard deviations for anomaly (default: 3)

    Returns:
        List of anomaly indices
    """
    mean = np.mean(values)
    std = np.std(values)

    anomalies = []
    for i, value in enumerate(values):
        z_score = abs((value - mean) / std) if std > 0 else 0
        if z_score > threshold:
            anomalies.append(i)

    return anomalies

# Example: Detect latency spikes
latencies = [100, 105, 98, 102, 500, 103, 99]  # 500ms is anomaly
anomaly_indices = detect_anomalies(latencies)
print(f"Anomalies at indices: {anomaly_indices}")  # [4]
```

---

## Chaos Engineering

### Fault Injection

**Inject network delays:**

```bash
# Add 100ms latency
sudo tc qdisc add dev lo root netem delay 100ms

# Add jitter (±20ms)
sudo tc qdisc add dev lo root netem delay 100ms 20ms

# Remove
sudo tc qdisc del dev lo root
```

**Inject packet loss:**

```bash
# 5% packet loss
sudo tc qdisc add dev lo root netem loss 5%

# Remove
sudo tc qdisc del dev lo root
```

**Kill random processes:**

```bash
# Chaos monkey for containers
while true; do
  sleep $((RANDOM % 300))  # Random 0-5 minutes
  docker restart $(docker ps --format '{{.Names}}' | shuf | head -1)
done
```

---

### Resilience Testing

**Test graceful degradation:**

```bash
# Stop worker
docker stop tts-worker

# Check orchestrator behavior
curl http://localhost:8081/health
# Should show worker=false but orchestrator still running

# Restart worker
docker start tts-worker
```

**Test recovery:**

```bash
# Kill Redis
docker stop redis

# Observe failures
docker logs orchestrator -f

# Restart Redis
docker start redis

# Verify automatic reconnection
docker logs orchestrator -f | grep -i "redis.*connect"
```

---

## Root Cause Analysis Framework

### 5 Whys Method

**Example:**

**Problem:** Audio playback is choppy

1. **Why?** Frame queue is filling up
   - **Check:** `docker logs orchestrator | grep "queue.*full"`

2. **Why?** Client not consuming frames fast enough
   - **Check:** Browser console frame processing time

3. **Why?** Client CPU at 100%
   - **Check:** Browser DevTools → Performance tab

4. **Why?** Audio decoding in main thread
   - **Check:** JavaScript profiler shows audio decoding hotspot

5. **Why?** Not using Web Audio API efficiently
   - **Root cause:** Need to use Web Workers for audio processing

**Solution:** Offload audio processing to Web Worker

---

### Timeline Reconstruction

**Build incident timeline:**

```bash
#!/bin/bash
# timeline.sh - Reconstruct incident timeline

START_TIME="2025-10-05T14:00:00"
END_TIME="2025-10-05T14:30:00"

echo "=== Incident Timeline ==="

# Orchestrator events
docker logs orchestrator --since "$START_TIME" --until "$END_TIME" | \
  jq '{timestamp, level, message}' > /tmp/orch-events.json

# Worker events
docker logs tts-worker --since "$START_TIME" --until "$END_TIME" | \
  jq '{timestamp, level, message}' > /tmp/worker-events.json

# Merge and sort
jq -s 'add | sort_by(.timestamp)' /tmp/orch-events.json /tmp/worker-events.json | \
  jq -r '.[] | "\(.timestamp) [\(.level)] \(.message)"'
```

---

## Emergency Procedures

### Immediate Mitigation

**High CPU:**

```bash
# Identify hot container
docker stats --no-stream | sort -k3 -h

# Restart container
docker restart <container-name>

# If persistent, scale down
docker compose scale tts-worker=0
```

**Out of Memory:**

```bash
# Check memory usage
docker stats --no-stream

# Find OOM-killed containers
docker inspect --format='{{.State.OOMKilled}}' $(docker ps -aq)

# Increase memory limit
# Edit docker-compose.yml:
# mem_limit: 4g

# Restart with new limit
docker compose up -d
```

**Disk Full:**

```bash
# Check disk usage
df -h

# Clean Docker
docker system prune -a -f --volumes

# Clean logs
truncate -s 0 /var/lib/docker/containers/*/*-json.log
```

---

## Prevention Strategies

### Circuit Breakers

**Implement circuit breaker:**

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, threshold: int = 5, timeout: float = 60.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure = None
        self.state = 'closed'  # closed, open, half-open

    async def call(self, func, *args, **kwargs):
        if self.state == 'open':
            if datetime.now() - self.last_failure > timedelta(seconds=self.timeout):
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            if self.state == 'half-open':
                self.state = 'closed'
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = datetime.now()
            if self.failures >= self.threshold:
                self.state = 'open'
            raise e
```

---

### Health Checks

**Comprehensive health check:**

```python
async def comprehensive_health_check() -> dict:
    """Check all system components."""
    checks = {}

    # Redis
    try:
        await redis_client.ping()
        checks['redis'] = {'status': 'healthy'}
    except Exception as e:
        checks['redis'] = {'status': 'unhealthy', 'error': str(e)}

    # Worker gRPC
    try:
        response = await grpc_client.health_check()
        checks['worker'] = {'status': 'healthy'}
    except Exception as e:
        checks['worker'] = {'status': 'unhealthy', 'error': str(e)}

    # Disk space
    import shutil
    total, used, free = shutil.disk_usage('/')
    checks['disk'] = {
        'status': 'healthy' if free / total > 0.1 else 'warning',
        'free_percent': free / total * 100
    }

    # Memory
    import psutil
    mem = psutil.virtual_memory()
    checks['memory'] = {
        'status': 'healthy' if mem.percent < 90 else 'warning',
        'used_percent': mem.percent
    }

    # Overall status
    overall = 'healthy' if all(c.get('status') == 'healthy' for c in checks.values()) else 'degraded'

    return {'status': overall, 'checks': checks}
```

---

## Related Runbooks

- **[Log Debugging](LOG_DEBUGGING.md)** - Log analysis techniques
- **[Monitoring](MONITORING.md)** - Metrics and alerting
- **[Audio Quality](AUDIO_QUALITY.md)** - Audio-specific issues
- **[Audio Backpressure](AUDIO_BACKPRESSURE.md)** - Frame delivery
- **[Test Debugging](TEST_DEBUGGING.md)** - Test failures

---

## Further Help

**Advanced diagnostic tools:**

```bash
# System-wide health check
./scripts/full-health-check.sh

# Distributed trace
./scripts/correlate-logs.sh <session-id>

# Performance profiling
py-spy record -o profile.svg --pid <pid> --duration 30

# Network capture
docker exec orchestrator tcpdump -i any -w /tmp/capture.pcap

# GPU monitoring
nvidia-smi dmon -s u
```

**Still stuck?**

1. Run full health check
2. Collect logs from all services
3. Generate performance profiles
4. Reconstruct timeline
5. File detailed issue with all diagnostics
