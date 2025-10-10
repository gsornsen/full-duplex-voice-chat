# Runbook: WebSocket Connection Errors

**Time to Resolution:** < 10 minutes
**Severity:** High (blocks client connections)
**Related:** [Environment Setup](ENVIRONMENT.md), [Log Debugging](LOG_DEBUGGING.md)

---

## Symptom

Client cannot connect to WebSocket endpoint. Common errors include:

- "WebSocket connection failed"
- "Connection refused on ws://localhost:8080"
- "Connection timeout"
- "Unexpected close frame"
- "Error 1006: Abnormal Closure"

---

## Common Causes

1. **Orchestrator not running** - Service process not started
2. **Wrong WebSocket URL** - Incorrect host, port, or protocol (ws vs wss)
3. **Port already in use** - Another process using port 8080
4. **Firewall blocking connection** - System or network firewall rules
5. **Orchestrator crashed** - Service started but encountered error

---

## Quick Diagnostic Checklist

```bash
# 1. Check orchestrator is running
curl -f http://localhost:8081/health
# Expected: {"status": "healthy", ...}

# 2. Verify port 8080 is listening
lsof -i :8080
# Expected: Python process listening on port 8080

# 3. Test WebSocket connection
wscat -c ws://localhost:8080
# Or use: websocat ws://localhost:8080

# 4. Check orchestrator logs
docker logs orchestrator --tail=50
# Or: tail -f logs/orchestrator.log

# 5. Verify configuration
grep -A 3 "websocket:" configs/orchestrator.yaml
# Expected: enabled: true, port: 8080
```

---

## Diagnostic Steps

### 1. Verify Orchestrator is Running

**Docker deployment:**
```bash
docker ps | grep orchestrator
```

**Expected output:**
```
CONTAINER ID   IMAGE              STATUS          PORTS
abc123def456   orchestrator:latest   Up 5 minutes   0.0.0.0:8080->8080/tcp
```

**Local process:**
```bash
ps aux | grep "orchestrator/server.py"
```

**If not running:**
```bash
# Start orchestrator
just run-orch

# Or with Docker
docker compose up orchestrator
```

---

### 2. Check Health Endpoint

```bash
curl -f http://localhost:8081/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "redis": true,
  "worker": true,
  "uptime_seconds": 123.45
}
```

**If health check fails:**
- Status 503: Service unhealthy (Redis or worker connection issue)
- Connection refused: Orchestrator not running
- Timeout: Service hung or unresponsive

**Resolution:**
```bash
# Check logs for startup errors
docker logs orchestrator

# Verify Redis is running
docker ps | grep redis
just redis  # If not running

# Verify worker is accessible
grpc-health-probe -addr=localhost:7001
```

---

### 3. Verify Port Availability

```bash
# Check what's using port 8080
lsof -i :8080

# Or with netstat
netstat -tuln | grep 8080

# Or with ss
ss -tuln | grep 8080
```

**Expected output (orchestrator running):**
```
python3   12345  user   10u  IPv4  0x123456  0t0  TCP *:8080 (LISTEN)
```

**If port in use by different process:**
```bash
# Find process ID
lsof -t -i :8080

# Kill conflicting process
kill -9 $(lsof -t -i :8080)

# Or change orchestrator port in configs/orchestrator.yaml
```

---

### 4. Test WebSocket Connection

**Using wscat (JavaScript):**
```bash
# Install wscat if needed
npm install -g wscat

# Test connection
wscat -c ws://localhost:8080

# Expected: Connected (press Ctrl+C to exit)
```

**Using websocat (Rust):**
```bash
# Install websocat
cargo install websocat

# Test connection
websocat ws://localhost:8080
```

**Using Python:**
```python
import asyncio
import aiohttp

async def test_ws():
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect('ws://localhost:8080') as ws:
            print("✅ Connected!")
            await ws.close()

asyncio.run(test_ws())
```

**Using JavaScript (browser):**
```javascript
const ws = new WebSocket('ws://localhost:8080');

ws.onopen = () => {
    console.log('✅ Connected!');
    ws.close();
};

ws.onerror = (error) => {
    console.error('❌ Connection failed:', error);
};
```

---

### 5. Check Orchestrator Logs

**Docker:**
```bash
# Last 50 lines
docker logs orchestrator --tail=50

# Follow logs
docker logs orchestrator -f

# With timestamps
docker logs orchestrator --timestamps
```

**Local:**
```bash
# Tail log file
tail -f logs/orchestrator.log

# Search for errors
grep -i "error" logs/orchestrator.log

# JSON log filtering (if using jq)
tail -f logs/orchestrator.log | jq 'select(.level == "ERROR")'
```

**Look for:**
- WebSocket server startup messages
- Port binding errors: "Address already in use"
- Configuration errors: "Invalid config"
- Crash stack traces

---

### 6. Verify Configuration

```bash
# Check WebSocket config
cat configs/orchestrator.yaml | grep -A 5 "websocket:"
```

**Expected:**
```yaml
websocket:
  enabled: true
  host: "0.0.0.0"
  port: 8080
  max_connections: 100
  frame_queue_size: 50
```

**Validate configuration:**
```bash
uv run python scripts/validate-config.py --orchestrator configs/orchestrator.yaml
```

---

## Resolution Strategies

### Orchestrator Not Running

**Start orchestrator:**
```bash
# Local development
just run-orch

# Docker Compose
docker compose up orchestrator

# Manual start
uv run python src/orchestrator/server.py --config configs/orchestrator.yaml
```

**Verify startup:**
```bash
# Wait a few seconds, then check health
sleep 5 && curl http://localhost:8081/health
```

---

### Wrong WebSocket URL

**Common URL mistakes:**

| Incorrect | Correct | Notes |
|-----------|---------|-------|
| `http://localhost:8080` | `ws://localhost:8080` | Use `ws://` protocol |
| `wss://localhost:8080` | `ws://localhost:8080` | Use `ws://` for local dev |
| `ws://localhost:8081` | `ws://localhost:8080` | Port 8081 is health endpoint |
| `ws://127.0.0.1:8080` | `ws://localhost:8080` | Both work, but use consistent |

**Update client connection URL:**
```python
# Python client
ws_url = "ws://localhost:8080"  # Correct

# JavaScript client
const ws = new WebSocket('ws://localhost:8080');
```

---

### Port Conflict

**Find and kill conflicting process:**
```bash
# Find process using port 8080
lsof -t -i :8080

# Kill process
kill -9 $(lsof -t -i :8080)

# Restart orchestrator
just run-orch
```

**Or change orchestrator port:**
```yaml
# configs/orchestrator.yaml
transport:
  websocket:
    port: 8090  # Use different port
```

Then update client URL: `ws://localhost:8090`

---

### Firewall Issues

**Linux (ufw):**
```bash
# Check firewall status
sudo ufw status

# Allow port 8080
sudo ufw allow 8080/tcp

# Reload firewall
sudo ufw reload
```

**macOS (pf):**
```bash
# Check if firewall is blocking
sudo pfctl -s rules | grep 8080

# Add rule (if needed)
# Edit /etc/pf.conf and add: pass in proto tcp to port 8080
sudo pfctl -f /etc/pf.conf
```

**Windows Firewall:**
```powershell
# Allow port 8080 inbound
New-NetFirewallRule -DisplayName "Orchestrator WS" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
```

**Docker network:**
```bash
# Ensure port is mapped
docker ps | grep orchestrator
# Should show: 0.0.0.0:8080->8080/tcp

# Recreate container if port not mapped
docker compose down && docker compose up orchestrator
```

---

### Orchestrator Crashed

**Check exit code:**
```bash
docker inspect orchestrator --format='{{.State.ExitCode}}'
```

**Common exit codes:**
- 0: Clean shutdown
- 1: Runtime error
- 137: OOM killed (out of memory)
- 139: Segmentation fault

**View crash logs:**
```bash
docker logs orchestrator --tail=100
```

**Common crash causes:**

1. **Redis connection failed:**
   ```bash
   # Start Redis first
   just redis

   # Verify Redis running
   redis-cli -u redis://localhost:6379 ping
   ```

2. **Worker unavailable:**
   ```bash
   # Check worker connection in config
   grep static_worker_addr configs/orchestrator.yaml

   # Start worker
   just run-tts-sesame
   ```

3. **Configuration error:**
   ```bash
   # Validate config
   uv run python scripts/validate-config.py
   ```

4. **Out of memory:**
   ```bash
   # Check memory usage
   docker stats orchestrator

   # Increase memory limit in docker-compose.yml
   services:
     orchestrator:
       mem_limit: 2g
   ```

---

## Prevention

### Pre-Flight Checks

**Before starting orchestrator:**
```bash
# 1. Run pre-flight check
./scripts/preflight-check.sh

# 2. Validate configuration
uv run python scripts/validate-config.py

# 3. Ensure dependencies running
docker ps | grep redis
docker ps | grep tts-worker

# 4. Check port availability
lsof -i :8080
```

---

### Health Monitoring

**Continuous monitoring:**
```bash
# Watch health endpoint
watch -n 5 'curl -s http://localhost:8081/health | jq'

# Or setup alert
while true; do
  if ! curl -f http://localhost:8081/health > /dev/null 2>&1; then
    echo "❌ Orchestrator unhealthy!"
  fi
  sleep 30
done
```

**Docker healthcheck:**
```yaml
# docker-compose.yml
services:
  orchestrator:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 10s
```

---

### Automated Testing

**Connection test script:**
```bash
# Test all connections
uv run python scripts/test-connection.py

# Expected output:
# ✅ Redis: Connected
# ✅ gRPC Worker: Healthy
# ✅ Orchestrator WebSocket: Connected
# ✅ Orchestrator Health: OK
```

---

## Example Debugging Session

**Scenario:** Client can't connect to WebSocket

```bash
# Step 1: Check health endpoint
$ curl http://localhost:8081/health
curl: (7) Failed to connect to localhost port 8081: Connection refused

# Diagnosis: Orchestrator not running

# Step 2: Check if container exists
$ docker ps -a | grep orchestrator
orchestrator   Exited (1) 2 minutes ago

# Step 3: View exit logs
$ docker logs orchestrator --tail=20
ERROR: Redis connection failed at redis://localhost:6379
Resolution: Start Redis with 'just redis'

# Step 4: Start Redis
$ just redis
Redis container started

# Step 5: Restart orchestrator
$ docker compose up orchestrator -d
Container orchestrator started

# Step 6: Verify health
$ curl http://localhost:8081/health
{"status":"healthy","redis":true,"worker":true}

# Step 7: Test WebSocket
$ wscat -c ws://localhost:8080
Connected (press CTRL+C to quit)

# ✅ Connection successful!
```

---

## Related Runbooks

- **[Environment Setup](ENVIRONMENT.md)** - Initial setup and prerequisites
- **[Redis Connection Failures](REDIS.md)** - Redis troubleshooting
- **[Port Conflicts](PORTS.md)** - Port conflict resolution
- **[Log Debugging](LOG_DEBUGGING.md)** - Log analysis techniques
- **[Audio Backpressure](AUDIO_BACKPRESSURE.md)** - Frame delivery issues

---

## Further Help

**Still stuck?**

1. Check logs with: `docker logs orchestrator --tail=100`
2. Validate configs: `uv run python scripts/validate-config.py`
3. Test connections: `uv run python scripts/test-connection.py`
4. Review setup: [Environment Setup Guide](ENVIRONMENT.md)
5. File issue: Include logs, config, and error messages

**Useful commands:**
```bash
# Full system check
./scripts/preflight-check.sh && \
  uv run python scripts/validate-config.py && \
  uv run python scripts/test-connection.py

# Clean restart
docker compose down && \
  just redis && \
  sleep 5 && \
  docker compose up orchestrator tts-worker
```
