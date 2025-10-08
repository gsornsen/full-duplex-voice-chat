# Runbook: Audio Frame Loss & Backpressure

**Time to Resolution:** 10-20 minutes
**Severity:** High (affects audio quality)
**Related:** [Log Debugging](LOG_DEBUGGING.md), [Monitoring](MONITORING.md), [Audio Quality](AUDIO_QUALITY.md)

---

## Overview

Audio backpressure occurs when the TTS worker produces audio frames faster than the client can consume them, leading to frame drops, choppy audio, or playback interruptions. This runbook covers detection, diagnosis, and resolution of backpressure issues.

**Key Concepts:**
- **Frame Rate:** 20ms frames at 48kHz = 50 frames/second
- **Frame Queue:** Buffer between worker and client (default: 50 frames = 1 second)
- **Backpressure:** Queue full condition causing frame drops

---

## Symptoms

**Audio Issues:**
- Choppy or stuttering audio playback
- Audio cuts out intermittently
- Delayed or missing speech segments
- Robot-like or distorted audio

**Log Indicators:**
```json
{"level": "WARNING", "message": "Frame queue full, dropping frames", "queue_size": 50}
{"level": "WARNING", "message": "Frame sequence gap detected", "expected": 42, "received": 45}
{"level": "ERROR", "message": "Client not consuming frames fast enough"}
```

**Client Symptoms:**
- Browser console: "Audio buffer underrun"
- Network tab: WebSocket frames backing up
- Jittery playback
- Audio-visual desync

---

## Quick Diagnostic Checklist

```bash
# 1. Check for frame drop warnings
docker logs orchestrator | grep -i "frame.*drop"

# 2. Monitor frame queue depth
docker logs orchestrator | jq 'select(.message | contains("queue")) | {timestamp, session_id, queue_size}'

# 3. Check network latency
# In client browser console:
# Measure round-trip time for WebSocket messages

# 4. Verify worker RTF (Real-Time Factor)
docker logs tts-worker | jq 'select(.rtf?) | {timestamp, rtf}'
# RTF should be < 1.0 (e.g., 0.3 = synthesizing 3x faster than real-time)

# 5. Check client playback stats
# In browser console:
# audioContext.currentTime - lastFrameTime
```

---

## Common Causes

### 1. Slow Network Connection

**Symptom:** Frames arrive slower than playback rate

**Diagnostic:**
```bash
# Test network latency
ping -c 10 localhost  # For local testing

# Check WebSocket frame timing
# Browser DevTools → Network → WS → Frames tab
# Look for irregular timing

# Monitor network stats
docker stats orchestrator --no-stream
# Check TX (transmitted bytes)
```

**Resolution:**
```yaml
# Reduce frame queue size for faster failure detection
# configs/orchestrator.yaml
transport:
  websocket:
    frame_queue_size: 25  # Reduce from 50
```

```javascript
// Client: Add buffering
const audioBuffer = [];
const MIN_BUFFER_MS = 200;  // Buffer 200ms before playback
```

---

### 2. Client CPU Overload

**Symptom:** Client too slow to process frames

**Diagnostic:**
```javascript
// Browser console: Check frame processing time
const start = performance.now();
await processAudioFrame(frame);
const duration = performance.now() - start;
console.log(`Frame processing: ${duration}ms`);
// Should be < 20ms
```

**Resolution:**

**Option A: Reduce frame rate (if possible in future)**
```yaml
# configs/worker.yaml (M4+ feature)
audio:
  frame_duration_ms: 40  # Reduce to 25 fps (from 50)
```

**Option B: Optimize client processing**
```javascript
// Use Web Audio API for efficient playback
const audioContext = new AudioContext({sampleRate: 48000});
const source = audioContext.createBufferSource();
// Avoid blocking operations in frame handler
```

**Option C: Use Web Workers**
```javascript
// Offload frame processing to worker thread
const worker = new Worker('audio-processor.js');
worker.postMessage({type: 'frame', data: audioFrame});
```

---

### 3. Worker Producing Too Fast (RTF Issue)

**Symptom:** Worker synthesizes faster than real-time playback

**Diagnostic:**
```bash
# Check real-time factor
docker logs tts-worker | jq 'select(.rtf?) | {timestamp, model_id, rtf}'

# RTF examples:
# 0.3 = 300ms to synthesize 1s of audio (too fast, causes backpressure)
# 1.0 = 1s to synthesize 1s (ideal)
# 2.0 = 2s to synthesize 1s (too slow, latency issue)
```

**Explanation:** Low RTF (< 0.5) means worker outputs frames much faster than playback rate, filling queue quickly.

**Resolution:**

**Option A: Rate limiting (M4+ feature)**
```python
# Worker: Add frame pacing
import asyncio

async def send_frame_with_pacing(frame: AudioFrame):
    await stub.SendFrame(frame)
    await asyncio.sleep(0.02)  # 20ms pacing for 50fps
```

**Option B: Increase client buffer**
```yaml
# configs/orchestrator.yaml
transport:
  websocket:
    frame_queue_size: 100  # Increase buffer (2 seconds)
```

**Option C: Client-side flow control**
```javascript
// Implement backpressure signaling
ws.send(JSON.stringify({
  type: 'flow_control',
  action: 'pause'
}));

// Resume when buffer drains
setTimeout(() => {
  ws.send(JSON.stringify({
    type: 'flow_control',
    action: 'resume'
  }));
}, 500);
```

---

### 4. WebSocket Congestion

**Symptom:** Network buffers full

**Diagnostic:**
```bash
# Check WebSocket send buffer
netstat -an | grep 8080
# Look for large Recv-Q or Send-Q values

# Docker container network stats
docker stats orchestrator --no-stream
# Check NET I/O

# Client browser DevTools
# Network tab → WS connection → Timing
# Look for "Stalled" time
```

**Resolution:**

**Option A: Enable WebSocket compression (if not already)**
```python
# orchestrator/transport/websocket_transport.py
app.router.add_route('GET', '/ws', websocket_handler, compress=True)
```

**Option B: Reduce frame size**
```yaml
# configs/worker.yaml
audio:
  output_sample_rate: 24000  # Half the bandwidth (M4+ feature)
  # Note: May affect quality
```

**Option C: Binary encoding**
```javascript
// Client: Use binary frames instead of JSON
const buffer = new ArrayBuffer(frame.length * 2);
const view = new Int16Array(buffer);
// ... fill buffer ...
ws.send(buffer);  // Binary frame
```

---

### 5. Client Playback Stall

**Symptom:** Client audio buffer not draining

**Diagnostic:**
```javascript
// Check AudioContext state
console.log(audioContext.state);  // Should be 'running'

// Check buffer queue
console.log(audioBufferQueue.length);
// Should be < 10 frames typically

// Measure playback lag
const lag = audioContext.currentTime - scheduledPlayTime;
console.log(`Playback lag: ${lag * 1000}ms`);
```

**Resolution:**

**Resume AudioContext:**
```javascript
// Ensure AudioContext is running
if (audioContext.state === 'suspended') {
  await audioContext.resume();
}
```

**Flush stale buffers:**
```javascript
// Clear old frames if lag > 500ms
if (lag > 0.5) {
  audioBufferQueue = [];
  console.warn('Flushed stale audio buffers');
}
```

---

## Diagnostic Steps

### 1. Measure Frame Timing

**Server-side (Orchestrator logs):**
```bash
# Extract frame send times
docker logs orchestrator | jq 'select(.message | contains("frame sent")) | {timestamp, session_id, sequence}'

# Calculate inter-frame intervals
docker logs orchestrator | jq -r 'select(.message | contains("frame sent")) | .timestamp' | \
  awk '{if(p){print ($1-p)*1000"ms"} p=$1}'
# Should be ~20ms
```

**Client-side (Browser console):**
```javascript
let lastFrameTime = null;

ws.onmessage = (event) => {
  const now = performance.now();
  if (lastFrameTime) {
    const interval = now - lastFrameTime;
    console.log(`Frame interval: ${interval.toFixed(1)}ms`);
    if (interval > 30) {
      console.warn(`Frame jitter: ${interval.toFixed(1)}ms > 30ms`);
    }
  }
  lastFrameTime = now;
};
```

---

### 2. Monitor Queue Depth

**Log-based monitoring:**
```bash
# Watch queue size in real-time
docker logs orchestrator -f | jq 'select(.queue_size?) | {timestamp, session_id, queue_size, message}'

# Alert on high queue
docker logs orchestrator -f | jq 'select(.queue_size? > 40) | "WARNING: Queue at \(.queue_size)/50"'
```

**Metrics-based (if Prometheus enabled):**
```bash
# Query Prometheus
curl -s 'http://localhost:9090/api/v1/query?query=tts_frame_queue_depth' | jq .

# Or view in Grafana dashboard
```

---

### 3. Check Frame Sequence Gaps

**Detect dropped frames:**
```bash
# Find sequence gaps in logs
docker logs orchestrator | jq 'select(.frame_sequence?) | {sequence, timestamp}' | \
  awk 'BEGIN{prev=0} {if(prev>0 && $1!=prev+1){print "GAP: "prev" -> "$1} prev=$1}'
```

**Client-side detection:**
```javascript
let expectedSequence = 0;

ws.onmessage = (event) => {
  const frame = JSON.parse(event.data);
  if (frame.sequence !== expectedSequence) {
    console.error(`Frame gap: expected ${expectedSequence}, got ${frame.sequence}`);
  }
  expectedSequence = frame.sequence + 1;
};
```

---

### 4. Measure End-to-End Latency

**Full pipeline latency:**
```bash
# Worker synthesis time
docker logs tts-worker | jq 'select(.synthesis_duration_ms?) | {timestamp, duration_ms: .synthesis_duration_ms}'

# Orchestrator queueing time
docker logs orchestrator | jq 'select(.queue_wait_ms?) | {timestamp, wait_ms: .queue_wait_ms}'

# Network delivery time (client-side)
# measure: server_timestamp - client_received_timestamp
```

**Target latencies:**
- Synthesis: < 100ms (RTF < 0.5)
- Queueing: < 20ms (queue not full)
- Network: < 50ms (local), < 200ms (remote)
- Total: < 300ms (p95 FAL target)

---

## Resolution Strategies

### Strategy 1: Increase Client Buffer

**When to use:** Occasional network jitter, but overall network is good

**Implementation:**
```yaml
# configs/orchestrator.yaml
transport:
  websocket:
    frame_queue_size: 100  # Increase from 50 (2 seconds buffer)
```

**Pros:**
- Absorbs transient network issues
- Simple configuration change

**Cons:**
- Increased latency (2s buffer)
- More memory usage

---

### Strategy 2: Client-Side Buffering

**When to use:** Variable network latency

**Implementation:**
```javascript
class AudioFrameBuffer {
  constructor(minBufferMs = 200) {
    this.buffer = [];
    this.minBufferMs = minBufferMs;
    this.playing = false;
  }

  addFrame(frame) {
    this.buffer.push(frame);
    if (!this.playing && this.bufferDuration() >= this.minBufferMs) {
      this.startPlayback();
    }
  }

  bufferDuration() {
    return this.buffer.length * 20;  // 20ms per frame
  }

  startPlayback() {
    this.playing = true;
    this.playNextFrame();
  }

  playNextFrame() {
    if (this.buffer.length === 0) {
      console.warn('Buffer underrun!');
      this.playing = false;
      return;
    }

    const frame = this.buffer.shift();
    playAudioFrame(frame);

    setTimeout(() => this.playNextFrame(), 20);
  }
}

const audioBuffer = new AudioFrameBuffer(200);  // 200ms min buffer
```

---

### Strategy 3: Adaptive Bitrate/Quality

**When to use:** Sustained network congestion (M4+ feature)

**Implementation:**
```yaml
# configs/worker.yaml (future feature)
audio:
  adaptive_quality: true
  quality_levels:
    - sample_rate: 48000
      frame_duration_ms: 20
    - sample_rate: 24000  # Fallback for congestion
      frame_duration_ms: 40
```

---

### Strategy 4: Flow Control Protocol

**When to use:** Client wants explicit control (M3+ feature)

**Implementation:**

**Client sends backpressure signal:**
```javascript
// Client detects buffer overrun
if (audioBuffer.length > 50) {
  ws.send(JSON.stringify({
    type: 'control',
    action: 'pause'
  }));
}

// Resume when buffer drains
if (audioBuffer.length < 10) {
  ws.send(JSON.stringify({
    type: 'control',
    action: 'resume'
  }));
}
```

**Server respects pause/resume:**
```python
# orchestrator: handle control messages
if msg['type'] == 'control':
    if msg['action'] == 'pause':
        await session.pause_audio()
    elif msg['action'] == 'resume':
        await session.resume_audio()
```

---

### Strategy 5: Frame Pacing (Worker-Side)

**When to use:** Worker RTF too low (< 0.3)

**Implementation:**
```python
# tts/worker.py
import asyncio
import time

async def synthesize_with_pacing(text: str):
    frame_interval = 0.02  # 20ms
    last_frame_time = time.time()

    for frame in model.synthesize_streaming(text):
        # Pace frame emission
        elapsed = time.time() - last_frame_time
        if elapsed < frame_interval:
            await asyncio.sleep(frame_interval - elapsed)

        yield frame
        last_frame_time = time.time()
```

---

## Monitoring & Alerting

### Key Metrics to Track

**Frame Queue Metrics:**
```bash
# Prometheus metrics (if enabled)
tts_frame_queue_depth{session_id="abc123"}
tts_frame_queue_full_count{session_id="abc123"}
tts_frames_dropped_total{session_id="abc123"}
```

**Latency Metrics:**
```bash
tts_frame_interval_ms{session_id="abc123"}
tts_frame_jitter_ms{session_id="abc123"}
tts_end_to_end_latency_ms{session_id="abc123"}
```

**Alert Rules (Prometheus):**
```yaml
# alerts.yml
groups:
  - name: audio_backpressure
    rules:
      - alert: FrameQueueHigh
        expr: tts_frame_queue_depth > 40
        for: 30s
        annotations:
          summary: "Frame queue depth > 80%"

      - alert: FramesDropped
        expr: rate(tts_frames_dropped_total[1m]) > 0
        annotations:
          summary: "Frames being dropped"

      - alert: HighJitter
        expr: tts_frame_jitter_ms > 10
        for: 1m
        annotations:
          summary: "Frame jitter > 10ms"
```

---

### Dashboard Visualization

**Grafana panels:**

1. **Queue Depth Over Time:**
   ```promql
   tts_frame_queue_depth{job="orchestrator"}
   ```

2. **Frame Drop Rate:**
   ```promql
   rate(tts_frames_dropped_total[1m])
   ```

3. **Jitter Distribution:**
   ```promql
   histogram_quantile(0.95, tts_frame_jitter_ms_bucket)
   ```

4. **RTF by Model:**
   ```promql
   tts_rtf{job="tts-worker"}
   ```

---

## Testing & Validation

### Load Testing

**Simulate slow client:**
```python
# test-slow-client.py
import asyncio
import websockets

async def slow_client():
    async with websockets.connect('ws://localhost:8080') as ws:
        while True:
            frame = await ws.recv()
            # Simulate slow processing
            await asyncio.sleep(0.025)  # 25ms (slower than 20ms rate)
            print(f"Processed frame (slow)")

asyncio.run(slow_client())
```

**Expected:** Queue should fill, trigger warnings

---

### Network Simulation

**Introduce latency:**
```bash
# Linux: add 100ms delay
sudo tc qdisc add dev lo root netem delay 100ms

# Test with delay
just cli

# Remove delay
sudo tc qdisc del dev lo root netem
```

**Introduce packet loss:**
```bash
sudo tc qdisc add dev lo root netem loss 5%
# Test
just cli
# Clean up
sudo tc qdisc del dev lo root netem
```

---

## Prevention

### Client Best Practices

1. **Use Web Audio API** (not HTML5 audio element)
2. **Implement buffering** (200ms minimum)
3. **Monitor buffer health** (alert on underrun)
4. **Handle backpressure gracefully** (pause/resume)
5. **Measure and log latency**

### Server Best Practices

1. **Monitor queue depth** (alert at 80%)
2. **Track frame drops** (zero drops in normal operation)
3. **Measure RTF** (keep < 0.5 for headroom)
4. **Enable metrics** (Prometheus + Grafana)
5. **Test under load** (simulate slow clients)

### Configuration Tuning

```yaml
# configs/orchestrator.yaml
transport:
  websocket:
    frame_queue_size: 50  # Tune based on network latency
    # Lower for local, higher for WAN

# configs/worker.yaml
audio:
  frame_duration_ms: 20  # Fixed in M2
  # Future: adaptive frame rate
```

---

## Related Runbooks

- **[Log Debugging](LOG_DEBUGGING.md)** - Log analysis for frame issues
- **[Monitoring](MONITORING.md)** - Metrics setup and alerting
- **[Audio Quality](AUDIO_QUALITY.md)** - Quality troubleshooting
- **[WebSocket Errors](WEBSOCKET.md)** - Connection issues

---

## Further Help

**Quick diagnostics:**
```bash
# Check for frame warnings
docker logs orchestrator | grep -i "frame.*drop\|queue.*full"

# Monitor queue depth
docker logs orchestrator -f | jq 'select(.queue_size?) | {queue_size, message}'

# Check worker RTF
docker logs tts-worker | jq 'select(.rtf?) | {rtf}'
```

**Still experiencing issues?**

1. Validate frame timing: `docker logs orchestrator | grep frame`
2. Check network: `docker stats orchestrator`
3. Review client logs (browser console)
4. Enable DEBUG logging: `LOG_LEVEL=DEBUG just run-orch`
5. File issue with: queue depth logs, RTF metrics, client timing data
