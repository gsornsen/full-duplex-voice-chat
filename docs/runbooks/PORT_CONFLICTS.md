# Runbook: Port Conflicts Resolution

**Time to Resolution:** < 5 minutes
**Severity:** Medium (prevents service startup)

## Symptom

Service fails to start with error messages:
- `Address already in use`
- `bind: address already in use`
- `Port XXX is already allocated`
- `Cannot assign requested address`

## Required Ports

| Port | Service | Protocol | Required |
|------|---------|----------|----------|
| 6379 | Redis | TCP | Yes |
| 7001 | TTS Worker (gRPC) | TCP | Yes |
| 8080 | Orchestrator (WebSocket) | TCP | Yes |
| 8081 | Orchestrator (Health) | TCP | Optional |
| 9090 | Metrics (Prometheus) | TCP | Optional |

## Diagnostic Steps

### 1. Check All Required Ports

```bash
# Quick check all ports
for port in 6379 7001 8080 8081 9090; do
  echo -n "Port $port: "
  if lsof -i :$port > /dev/null 2>&1; then
    echo "IN USE"
  else
    echo "available"
  fi
done
```

### 2. Identify Process Using Port

**Using lsof (Linux/macOS):**
```bash
# Check specific port
lsof -i :8080

# Example output:
# COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
# node    12345 user   21u  IPv4 123456      0t0  TCP *:8080 (LISTEN)
```

**Using ss (Linux):**
```bash
ss -tuln | grep :8080
```

**Using netstat (cross-platform):**
```bash
netstat -tuln | grep :8080
```

**For Docker:**
```bash
# List port mappings for all containers
docker ps --format "table {{.Names}}\t{{.Ports}}"
```

### 3. Determine if Process is Important

```bash
# Get process details
ps -p <PID> -f

# Example: PID 12345
ps -p 12345 -f
# Check if it's a development server, old container, etc.
```

## Resolution Strategies

### Strategy 1: Kill Conflicting Process

**If process is safe to kill:**
```bash
# Kill gracefully
kill <PID>

# Force kill if needed
kill -9 <PID>
```

**Example:**
```bash
# Find process on port 8080
PID=$(lsof -t -i :8080)

# Kill it
kill -9 $PID
```

### Strategy 2: Stop Conflicting Docker Container

```bash
# List containers using ports
docker ps

# Stop specific container
docker stop <container_name>

# Or remove if no longer needed
docker rm -f <container_name>
```

**Clean up old containers:**
```bash
# Remove all stopped containers
docker container prune

# Or force remove all M2 containers
docker ps -a | grep -E "redis|tts|orchestrator" | awk '{print $1}' | xargs docker rm -f
```

### Strategy 3: Change Service Port

**For Redis:**
```yaml
# docker-compose.yml
services:
  redis:
    ports:
      - "6380:6379"  # Map to different host port
```

Update orchestrator config:
```yaml
# configs/orchestrator.yaml
redis:
  url: "redis://localhost:6380"
```

**For TTS Worker:**
```yaml
# configs/worker.yaml
grpc:
  port: 7002  # Use different port
```

Update orchestrator routing:
```yaml
# configs/orchestrator.yaml
routing:
  static_worker_addr: "grpc://localhost:7002"
```

**For Orchestrator WebSocket:**
```yaml
# configs/orchestrator.yaml
transport:
  websocket:
    port: 8088  # Use different port
```

Update client connection:
```bash
# CLI client
just cli HOST="ws://localhost:8088"
```

### Strategy 4: Use Different Port Range

If all default ports are blocked, use alternative range:

```yaml
# docker-compose.yml - Example with custom ports
services:
  redis:
    ports:
      - "16379:6379"  # Redis on 16379

  tts-worker-0:
    ports:
      - "17001:7001"  # Worker on 17001

  orchestrator:
    ports:
      - "18080:8080"  # Orchestrator on 18080
      - "18081:8081"  # Health on 18081
```

**Update all configs accordingly.**

## Common Scenarios

### Scenario 1: Previous M2 Instance Still Running

```bash
# Stop all M2 containers
docker compose down

# Verify all stopped
docker ps | grep -E "redis|tts|orchestrator"

# Clean up networks
docker network prune
```

### Scenario 2: System Service Using Port

**Example: Another Redis instance**
```bash
# Check if Redis service is running
sudo systemctl status redis

# Stop if not needed
sudo systemctl stop redis

# Disable auto-start
sudo systemctl disable redis
```

### Scenario 3: Development Server on 8080

Many development tools use 8080 by default (webpack, rails, etc.)

```bash
# Find process
lsof -i :8080

# If it's a development server you need, change orchestrator port
# configs/orchestrator.yaml
transport:
  websocket:
    port: 8090
```

### Scenario 4: Port Held by Crashed Process

Sometimes OS doesn't release port immediately:

```bash
# Wait 30 seconds
sleep 30

# Or force TCP TIME_WAIT cleanup (Linux)
sudo sysctl -w net.ipv4.tcp_tw_reuse=1

# Retry service start
docker compose up
```

## Prevention

**1. Run pre-flight check:**
```bash
./scripts/preflight-check.sh
```

**2. Always use docker-compose down:**
```bash
# Instead of just stopping containers
docker compose down

# This releases ports and cleans up networks
```

**3. Use unique port ranges per project:**
```bash
# For M2 demo
export REDIS_PORT=6379
export WORKER_PORT=7001
export ORCH_PORT=8080

# For other projects, use different ranges
```

**4. Set up port exclusion ranges:**
```bash
# Linux: Reserve ports for M2
sudo sysctl -w net.ipv4.ip_local_reserved_ports=6379,7001,8080,8081
```

**5. Clean up regularly:**
```bash
# Weekly cleanup
docker container prune
docker network prune
docker volume prune
```

## Quick Resolution Script

Save as `scripts/free-ports.sh`:
```bash
#!/usr/bin/env bash
# Free up M2 required ports

PORTS=(6379 7001 8080 8081)

for port in "${PORTS[@]}"; do
  PID=$(lsof -t -i :$port 2>/dev/null)
  if [ -n "$PID" ]; then
    echo "Killing process $PID on port $port"
    kill -9 $PID
  fi
done

echo "All M2 ports freed"
```

Usage:
```bash
chmod +x scripts/free-ports.sh
./scripts/free-ports.sh
```

## Validation

After resolving conflicts:

```bash
# 1. Verify ports are free
./scripts/preflight-check.sh

# 2. Start services
docker compose up

# 3. Check all services are listening
ss -tuln | grep -E "6379|7001|8080|8081"

# Expected: All ports showing LISTEN state
```

## Quick Checklist

- [ ] Identified which port is in conflict?
- [ ] Found process using port? (`lsof -i :PORT`)
- [ ] Determined if process can be killed?
- [ ] Stopped old Docker containers? (`docker compose down`)
- [ ] Changed port config if needed?
- [ ] Updated client connection strings?
- [ ] Verified ports are free? (`./scripts/preflight-check.sh`)

## Platform-Specific Notes

### Linux
```bash
# Check port
sudo lsof -i :8080
# Or
sudo ss -tuln | grep :8080
```

### macOS
```bash
# Check port
lsof -i :8080
```

### Windows (WSL2)
```bash
# Check port in WSL
ss -tuln | grep :8080

# Check port from Windows PowerShell
netstat -ano | findstr :8080
```

**Note:** WSL2 port forwarding can sometimes conflict. Restart WSL if issues persist:
```powershell
# From PowerShell (admin)
wsl --shutdown
wsl
```

## Related Runbooks

- [gRPC Worker Connectivity](GRPC_WORKER.md)
- [Redis Connection Failures](REDIS.md)
- [Environment Setup](ENVIRONMENT.md)
- [WebSocket Connection Errors](WEBSOCKET.md)

## Still Having Issues?

1. Reboot system (releases all ports)
2. Check firewall rules: `sudo ufw status`
3. Review Docker port mappings: `docker inspect <container>`
4. Try binding to 0.0.0.0 instead of localhost
5. Check for IPv4/IPv6 conflicts

## Advanced: Dynamic Port Allocation

For CI/CD environments, use dynamic ports:

```bash
# Find available port
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Start service on available port
docker run -p $PORT:8080 orchestrator
```
