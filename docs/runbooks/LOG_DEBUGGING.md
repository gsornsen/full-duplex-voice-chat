# Runbook: Log Parsing & Debugging

**Time to Resolution:** 5-15 minutes
**Severity:** Medium (diagnostic tool for other issues)
**Related:** [WebSocket Errors](WEBSOCKET.md), [Environment Setup](ENVIRONMENT.md)

---

## Overview

This runbook covers log analysis techniques for the M2 Realtime Duplex Voice Demo system. Logs are your primary diagnostic tool for understanding system behavior and identifying issues.

**Log Format:** JSON structured logs (default)
**Log Levels:** DEBUG, INFO, WARNING, ERROR, CRITICAL

---

## Quick Reference

```bash
# View orchestrator logs
docker logs orchestrator --tail=50
# Or: tail -f logs/orchestrator.log

# View worker logs
docker logs tts-worker --tail=50
# Or: tail -f logs/worker.log

# View Redis logs
docker logs redis --tail=50

# Follow all logs (Docker Compose)
docker compose logs -f

# Search for errors
docker logs orchestrator | grep -i error

# Filter by log level (JSON logs)
docker logs orchestrator | jq 'select(.level == "ERROR")'

# Filter by session ID
docker logs orchestrator | jq 'select(.session_id == "abc123")'

# Watch for specific pattern
docker logs orchestrator -f | grep "WebSocket"
```

---

## Log Locations

### Docker Deployment

**Container logs (default):**
```bash
# Orchestrator
docker logs orchestrator

# TTS Worker
docker logs tts-worker

# Redis
docker logs redis

# All services
docker compose logs
```

**Persistent logs (if volume mounted):**
```bash
# Check docker-compose.yml for volume mounts
docker compose config | grep -A 3 volumes

# Example: logs mounted to ./logs/
ls -la logs/
# orchestrator.log
# worker.log
```

### Local Development

**File locations:**
```bash
logs/
├── orchestrator.log       # Main orchestrator log
├── worker.log             # TTS worker log
├── orchestrator.json      # JSON formatted orchestrator log
└── worker.json            # JSON formatted worker log
```

**Live tail:**
```bash
tail -f logs/orchestrator.log
tail -f logs/worker.log

# Or both
tail -f logs/*.log
```

---

## Log Levels

### Level Descriptions

| Level | Usage | Example |
|-------|-------|---------|
| **DEBUG** | Detailed diagnostic info | "Received audio frame: 960 samples" |
| **INFO** | Normal operation events | "Session started: session_id=abc123" |
| **WARNING** | Unexpected but handled | "Frame queue 80% full, may drop frames" |
| **ERROR** | Serious issues | "gRPC connection failed: connection refused" |
| **CRITICAL** | System failure | "Out of memory, shutting down" |

### Filtering by Level

**Docker logs:**
```bash
# Only errors and critical
docker logs orchestrator | grep -E "ERROR|CRITICAL"

# Only warnings and above
docker logs orchestrator | grep -E "WARNING|ERROR|CRITICAL"

# JSON filtering with jq
docker logs orchestrator | jq 'select(.level == "ERROR" or .level == "CRITICAL")'
```

**Local logs:**
```bash
# grep patterns
grep "ERROR" logs/orchestrator.log
grep -E "ERROR|CRITICAL" logs/orchestrator.log

# JSON logs with jq
cat logs/orchestrator.json | jq 'select(.level == "ERROR")'
```

---

## Log Structure

### JSON Format (Default)

**Example log entry:**
```json
{
  "timestamp": "2025-10-05T14:23:45.123Z",
  "level": "INFO",
  "logger": "orchestrator.websocket",
  "message": "WebSocket connection established",
  "session_id": "session-abc123",
  "client_ip": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "context": {
    "connection_id": "conn-456",
    "protocol": "websocket"
  }
}
```

**Key fields:**
- `timestamp`: ISO 8601 timestamp with milliseconds
- `level`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logger`: Component/module name
- `message`: Human-readable message
- `session_id`: Session identifier (if applicable)
- `context`: Additional structured data

### Text Format (Legacy)

**Example:**
```
2025-10-05 14:23:45,123 INFO [orchestrator.websocket] session=session-abc123 - WebSocket connection established
```

**Format:** `timestamp level [logger] session=session_id - message`

---

## Common Log Patterns

### Session Lifecycle

**Connection established:**
```json
{"level": "INFO", "message": "WebSocket connection established", "session_id": "session-abc123"}
```

**Session started:**
```json
{"level": "INFO", "message": "TTS session started", "session_id": "session-abc123", "model_id": "cosyvoice2-en-base"}
```

**Session ended:**
```json
{"level": "INFO", "message": "Session ended gracefully", "session_id": "session-abc123", "duration_ms": 12345}
```

**Connection error:**
```json
{"level": "ERROR", "message": "WebSocket connection error", "session_id": "session-abc123", "error": "Connection reset by peer"}
```

### gRPC Communication

**Successful synthesis request:**
```json
{"level": "INFO", "message": "Synthesis request sent", "session_id": "session-abc123", "text_length": 128}
```

**gRPC connection failure:**
```json
{"level": "ERROR", "message": "gRPC worker connection failed", "worker_addr": "grpc://localhost:7001", "error": "Connection refused"}
```

**Frame received:**
```json
{"level": "DEBUG", "message": "Audio frame received", "session_id": "session-abc123", "frame_size": 960, "sequence": 42}
```

### Model Management

**Model load started:**
```json
{"level": "INFO", "message": "Loading model", "model_id": "cosyvoice2-en-base"}
```

**Model loaded:**
```json
{"level": "INFO", "message": "Model loaded successfully", "model_id": "cosyvoice2-en-base", "load_time_ms": 2345}
```

**Model load failed:**
```json
{"level": "ERROR", "message": "Model load failed", "model_id": "cosyvoice2-en-base", "error": "Model file not found"}
```

### Resource Issues

**High memory usage:**
```json
{"level": "WARNING", "message": "High memory usage", "memory_percent": 85.2, "resident_models": 3}
```

**Frame queue full:**
```json
{"level": "WARNING", "message": "Frame queue full, dropping frames", "session_id": "session-abc123", "queue_size": 50}
```

**Out of memory:**
```json
{"level": "CRITICAL", "message": "Out of memory, cannot load model", "model_id": "cosyvoice2-en-base"}
```

---

## Search Techniques

### grep Patterns

**Find all errors in last 24 hours:**
```bash
docker logs orchestrator --since 24h | grep ERROR
```

**Find specific error type:**
```bash
# Connection errors
docker logs orchestrator | grep -i "connection.*failed"

# Timeout errors
docker logs orchestrator | grep -i "timeout"

# Configuration errors
docker logs orchestrator | grep -i "config.*error"
```

**Case-insensitive search:**
```bash
docker logs orchestrator | grep -i "websocket"
```

**Multiple patterns (OR):**
```bash
docker logs orchestrator | grep -E "ERROR|CRITICAL|FATAL"
```

**Context lines (show surrounding lines):**
```bash
# 3 lines before and after match
docker logs orchestrator | grep -C 3 "ERROR"

# 5 lines before
docker logs orchestrator | grep -B 5 "ERROR"

# 5 lines after
docker logs orchestrator | grep -A 5 "ERROR"
```

### jq Filtering (JSON Logs)

**Filter by level:**
```bash
docker logs orchestrator | jq 'select(.level == "ERROR")'
```

**Filter by session:**
```bash
docker logs orchestrator | jq 'select(.session_id == "session-abc123")'
```

**Filter by time range:**
```bash
# After specific timestamp
docker logs orchestrator | jq 'select(.timestamp > "2025-10-05T14:00:00Z")'

# Between timestamps
docker logs orchestrator | jq 'select(.timestamp > "2025-10-05T14:00:00Z" and .timestamp < "2025-10-05T15:00:00Z")'
```

**Multiple conditions (AND):**
```bash
docker logs orchestrator | jq 'select(.level == "ERROR" and .session_id == "session-abc123")'
```

**Multiple conditions (OR):**
```bash
docker logs orchestrator | jq 'select(.level == "ERROR" or .level == "CRITICAL")'
```

**Extract specific fields:**
```bash
# Only timestamp and message
docker logs orchestrator | jq '{timestamp, message}'

# Only errors with session ID
docker logs orchestrator | jq 'select(.level == "ERROR") | {timestamp, session_id, message}'
```

**Count occurrences:**
```bash
# Count errors by type
docker logs orchestrator | jq 'select(.level == "ERROR") | .message' | sort | uniq -c

# Count sessions
docker logs orchestrator | jq -r '.session_id' | sort | uniq -c
```

### Advanced Filtering

**Combine grep and jq:**
```bash
# First grep for speed, then jq for structure
docker logs orchestrator | grep ERROR | jq .
```

**Time-based filtering (Docker):**
```bash
# Last hour
docker logs orchestrator --since 1h

# Since specific time
docker logs orchestrator --since "2025-10-05T14:00:00"

# Last N lines
docker logs orchestrator --tail 100
```

**Follow logs in real-time:**
```bash
# Follow all logs
docker logs orchestrator -f

# Follow and filter
docker logs orchestrator -f | grep ERROR

# Follow with jq
docker logs orchestrator -f | jq 'select(.level == "ERROR")'
```

---

## Debugging Workflows

### Workflow 1: Session Failure Investigation

**Scenario:** Client reports session failure

```bash
# 1. Find session ID from client error message or logs
SESSION_ID="session-abc123"

# 2. Extract all logs for that session
docker logs orchestrator | jq "select(.session_id == \"$SESSION_ID\")"

# 3. Look for errors
docker logs orchestrator | jq "select(.session_id == \"$SESSION_ID\" and .level == \"ERROR\")"

# 4. Get session timeline
docker logs orchestrator | jq "select(.session_id == \"$SESSION_ID\") | {timestamp, level, message}" | less

# 5. Check worker logs for same session
docker logs tts-worker | jq "select(.session_id == \"$SESSION_ID\")"
```

### Workflow 2: Connection Issue Diagnosis

**Scenario:** Intermittent connection failures

```bash
# 1. Find all connection errors in last hour
docker logs orchestrator --since 1h | grep -i "connection.*failed"

# 2. Count frequency
docker logs orchestrator --since 1h | grep -i "connection.*failed" | wc -l

# 3. Identify pattern (time-based? specific client?)
docker logs orchestrator --since 1h | jq 'select(.message | contains("connection") and contains("failed")) | {timestamp, client_ip, error}'

# 4. Check if worker-related
docker logs tts-worker --since 1h | grep -i "connection"

# 5. Verify network (Redis, gRPC)
docker logs orchestrator | grep -E "redis|grpc" | grep -i error
```

### Workflow 3: Performance Investigation

**Scenario:** Slow response times

```bash
# 1. Find high latency events
docker logs orchestrator | jq 'select(.latency_ms? > 500) | {timestamp, session_id, latency_ms, message}'

# 2. Check frame timing
docker logs orchestrator | jq 'select(.message | contains("frame")) | {timestamp, session_id, frame_sequence, jitter_ms}'

# 3. Look for queue warnings
docker logs orchestrator | grep -i "queue"

# 4. Check worker RTF (real-time factor)
docker logs tts-worker | jq 'select(.rtf?) | {timestamp, model_id, rtf}'

# 5. Identify bottleneck
docker logs orchestrator | jq 'select(.level == "WARNING") | {timestamp, message}' | less
```

### Workflow 4: Crash Root Cause Analysis

**Scenario:** Service crashed unexpectedly

```bash
# 1. Check last logs before crash
docker logs orchestrator --tail 100

# 2. Look for critical errors
docker logs orchestrator | grep CRITICAL

# 3. Check for OOM or resource exhaustion
docker logs orchestrator | grep -E "memory|OOM|resource"

# 4. Review stack trace (if present)
docker logs orchestrator | grep -A 50 "Traceback"

# 5. Check container exit code
docker inspect orchestrator --format='{{.State.ExitCode}}'
# 0: Clean exit
# 1: Error exit
# 137: OOM killed
# 139: Segfault

# 6. Review system resources at crash time
docker stats orchestrator --no-stream
```

---

## Common Error Patterns

### Error: "Connection refused"

**Log pattern:**
```json
{"level": "ERROR", "message": "gRPC worker connection failed", "error": "Connection refused"}
```

**Cause:** Worker not running or wrong address

**Diagnostic:**
```bash
# Check worker status
docker ps | grep tts-worker

# Check worker address in config
grep static_worker_addr configs/orchestrator.yaml

# Test gRPC connection
grpc-health-probe -addr=localhost:7001
```

**Resolution:** See [gRPC Worker Runbook](GRPC_WORKER.md)

---

### Error: "Port already in use"

**Log pattern:**
```
ERROR: Failed to bind to port 8080: Address already in use
```

**Cause:** Another process using the port

**Diagnostic:**
```bash
lsof -i :8080
```

**Resolution:** See [Port Conflicts Runbook](PORTS.md)

---

### Error: "Redis connection failed"

**Log pattern:**
```json
{"level": "ERROR", "message": "Redis connection failed", "url": "redis://localhost:6379", "error": "Connection refused"}
```

**Cause:** Redis not running

**Diagnostic:**
```bash
docker ps | grep redis
redis-cli -u redis://localhost:6379 ping
```

**Resolution:** See [Redis Runbook](REDIS.md)

---

### Error: "Model not found"

**Log pattern:**
```json
{"level": "ERROR", "message": "Model load failed", "model_id": "cosyvoice2-en-base", "error": "Model file not found"}
```

**Cause:** Model files missing from voicepacks/

**Diagnostic:**
```bash
ls -la voicepacks/cosyvoice2/en-base/
```

**Resolution:**
```bash
# Download model or check path
# Verify config points to correct model_id
grep default_model_id configs/worker.yaml
```

---

### Warning: "Frame queue full"

**Log pattern:**
```json
{"level": "WARNING", "message": "Frame queue full, dropping frames", "session_id": "session-abc123", "queue_size": 50}
```

**Cause:** Consumer too slow or network congestion

**Resolution:** See [Audio Backpressure Runbook](AUDIO_BACKPRESSURE.md)

---

## Log Aggregation & Analysis

### Multi-Service Log Correlation

**View all services together:**
```bash
# Docker Compose: interleaved logs
docker compose logs -f

# Separate terminal per service
tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  send-keys 'docker logs orchestrator -f' C-m \; \
  select-pane -t 1 \; \
  send-keys 'docker logs tts-worker -f' C-m \; \
  select-pane -t 2 \; \
  send-keys 'docker logs redis -f' C-m
```

**Correlate by session ID:**
```bash
SESSION="session-abc123"

# Extract from all services
echo "=== Orchestrator ===" && \
  docker logs orchestrator | jq "select(.session_id == \"$SESSION\")" && \
echo "=== Worker ===" && \
  docker logs tts-worker | jq "select(.session_id == \"$SESSION\")"
```

### Log Export for Analysis

**Export to file:**
```bash
# Export last hour
docker logs orchestrator --since 1h > /tmp/orch-last-hour.log

# Export with timestamps
docker logs orchestrator --timestamps > /tmp/orch-full.log

# Export as JSON
docker logs orchestrator | jq -s '.' > /tmp/orch-logs.json
```

**Analyze with tools:**
```bash
# Count error types
cat /tmp/orch-logs.json | jq -r 'select(.level == "ERROR") | .message' | sort | uniq -c

# Create timeline
cat /tmp/orch-logs.json | jq -r '{timestamp, level, message} | @csv' > timeline.csv

# Import to spreadsheet or analysis tool
```

---

## Log Level Configuration

### Change Log Level (Runtime)

**Orchestrator:**
```yaml
# configs/orchestrator.yaml
log_level: "DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

**Worker:**
```yaml
# configs/worker.yaml
logging:
  level: "DEBUG"
```

**Restart service to apply:**
```bash
docker compose restart orchestrator
docker compose restart tts-worker
```

### Temporary Debug Logging

**Enable DEBUG for specific component:**

Edit code (temporary):
```python
import logging
logging.getLogger('orchestrator.websocket').setLevel(logging.DEBUG)
```

Or use environment variable:
```bash
# Set log level via env
LOG_LEVEL=DEBUG just run-orch
```

---

## Performance Tips

### Reduce Log Overhead

**Production settings:**
```yaml
logging:
  level: "INFO"  # Don't use DEBUG in production
  format: "json"  # Structured for parsing
  include_session_id: true
```

**Disable verbose logs:**
```python
# Silence noisy libraries
logging.getLogger('grpc').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
```

### Log Rotation

**Docker log rotation:**
```json
// /etc/docker/daemon.json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

**Restart Docker daemon:**
```bash
sudo systemctl restart docker
```

---

## Related Runbooks

- **[WebSocket Errors](WEBSOCKET.md)** - Connection diagnostics
- **[Environment Setup](ENVIRONMENT.md)** - Setup validation
- **[Audio Backpressure](AUDIO_BACKPRESSURE.md)** - Frame delivery issues
- **[Monitoring](MONITORING.md)** - Metrics and alerting

---

## Further Help

**Useful log analysis commands cheat sheet:**

```bash
# Quick error check
docker logs orchestrator --tail=100 | grep ERROR

# Session timeline
docker logs orchestrator | jq "select(.session_id == \"$SID\") | {timestamp, message}"

# Error summary
docker logs orchestrator | jq -r 'select(.level == "ERROR") | .message' | sort | uniq -c

# Live error watch
docker logs orchestrator -f | grep --color=always ERROR

# Export for analysis
docker logs orchestrator --since 1h > analysis.log
```

**Still stuck?**

1. Check recent errors: `docker logs orchestrator --tail=100 | grep ERROR`
2. Validate config: `uv run python scripts/validate-config.py`
3. Test connections: `uv run python scripts/test-connection.py`
4. Review related runbooks above
5. File issue with relevant log excerpts
