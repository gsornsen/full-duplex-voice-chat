# Runbook: Redis Connection Failures

**Time to Resolution:** < 5 minutes
**Severity:** High (blocks worker registration and discovery)

## Symptom

Services fail to connect to Redis with error messages:
- `Redis connection failed`
- `Connection refused` on port 6379
- `Worker registration failed`
- `Could not connect to Redis at localhost:6379`

## Common Causes

1. Redis server not running
2. Wrong Redis URL in configuration
3. Redis container stopped or crashed
4. Network connectivity issues
5. Redis authentication required but not provided
6. Port 6379 blocked or already in use

## Diagnostic Steps

### 1. Check Redis Container Status

```bash
# List running containers
docker ps | grep redis

# Expected: redis-tts container running
# If not present:
docker ps -a | grep redis  # Check if stopped
```

### 2. Test Redis Connectivity

```bash
# Using redis-cli (install if needed)
redis-cli -h localhost -p 6379 ping

# Expected: PONG
# Error: Connection refused → Redis not running
# Error: timeout → firewall or network issue
```

**Using Docker:**
```bash
docker exec redis-tts redis-cli ping

# Expected: PONG
```

### 3. Check Redis Logs

```bash
# Docker logs
docker logs redis-tts --tail=50

# Look for:
# - "Ready to accept connections" (success)
# - "Address already in use" (port conflict)
# - "Out of memory" (resource exhaustion)
```

### 4. Verify Configuration

```bash
# Check orchestrator config
grep -A 3 "redis:" configs/orchestrator.yaml

# Expected:
# redis:
#   url: "redis://localhost:6379"
#   db: 0
```

**Common mistakes:**
- Wrong hostname (e.g., `redis` vs `localhost`)
- Wrong port
- Missing `redis://` prefix
- Using Docker container name when connecting from host

### 5. Test Port Availability

```bash
# Check if port is listening
nc -zv localhost 6379

# Or:
telnet localhost 6379

# Expected: Connection successful
```

**Check for port conflicts:**
```bash
lsof -i :6379
# Or:
ss -tuln | grep :6379
```

## Resolution Strategies

### Redis Not Running

**Start Redis with Docker:**
```bash
# Using justfile
just redis

# Or manually:
docker run -d \
  --name redis-tts \
  -p 6379:6379 \
  redis:7-alpine

# Or with docker-compose:
docker compose up redis -d
```

**Verify Redis started:**
```bash
docker logs redis-tts
# Should see: "Ready to accept connections"
```

### Wrong Redis URL

**For orchestrator running locally with Redis in Docker:**
```yaml
# configs/orchestrator.yaml
redis:
  url: "redis://localhost:6379"
  db: 0
```

**For orchestrator in Docker container:**
```yaml
# configs/orchestrator.yaml
redis:
  url: "redis://redis:6379"  # Use container name
  db: 0
```

**Verify connection:**
```bash
# Test from orchestrator container
docker exec orchestrator \
  redis-cli -h redis -p 6379 ping
```

### Redis Container Stopped

**Start existing container:**
```bash
docker start redis-tts
```

**If container is corrupted, recreate:**
```bash
docker rm -f redis-tts
docker run -d --name redis-tts -p 6379:6379 redis:7-alpine
```

### Port Conflict

**Find conflicting process:**
```bash
lsof -i :6379
# Note the PID

# Kill if safe to do so:
kill <PID>
```

**Or use different port:**
```bash
# Start Redis on different port
docker run -d \
  --name redis-tts \
  -p 6380:6379 \
  redis:7-alpine

# Update config
# configs/orchestrator.yaml
redis:
  url: "redis://localhost:6380"
```

### Network Connectivity Issues

**Verify Docker network:**
```bash
# Check network exists
docker network ls | grep tts-network

# Inspect network
docker network inspect tts-network

# Ensure Redis is on network
docker inspect redis-tts | grep -A 10 Networks
```

**Reconnect container to network:**
```bash
docker network connect tts-network redis-tts
```

### Redis Authentication

If Redis requires authentication:
```yaml
# configs/orchestrator.yaml
redis:
  url: "redis://:password@localhost:6379"
  # Or:
  url: "redis://username:password@localhost:6379"
```

**Set password in Redis:**
```bash
docker exec redis-tts \
  redis-cli CONFIG SET requirepass "mypassword"
```

### Redis Out of Memory

```bash
# Check Redis memory usage
docker exec redis-tts redis-cli INFO memory

# Increase maxmemory (example: 256MB)
docker exec redis-tts \
  redis-cli CONFIG SET maxmemory 268435456

# Or set eviction policy
docker exec redis-tts \
  redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

## Prevention

**1. Run pre-flight check:**
```bash
./scripts/preflight-check.sh
```

**2. Use Docker healthchecks:**
```yaml
# docker-compose.yml
services:
  redis:
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
```

**3. Monitor Redis health:**
```bash
# Continuous monitoring
watch -n 5 "docker exec redis-tts redis-cli ping"
```

**4. Use connection pooling:**
```yaml
# configs/orchestrator.yaml
redis:
  connection_pool_size: 10  # Reuse connections
```

**5. Set up persistent storage (optional):**
```bash
docker run -d \
  --name redis-tts \
  -p 6379:6379 \
  -v redis-data:/data \
  redis:7-alpine redis-server --appendonly yes
```

## Validation

After applying fixes, verify:

```bash
# 1. Redis is running
docker ps | grep redis

# 2. Ping succeeds
redis-cli ping

# 3. Worker can register
# Check orchestrator logs for "Connected to Redis"

# 4. Workers are discoverable
redis-cli KEYS "worker:*"
# Should list registered workers
```

## Quick Checklist

- [ ] Redis container running? (`docker ps | grep redis`)
- [ ] Redis responds to ping? (`redis-cli ping`)
- [ ] Correct URL in config? (`grep redis: configs/orchestrator.yaml`)
- [ ] Port 6379 accessible? (`nc -zv localhost 6379`)
- [ ] No port conflicts? (`lsof -i :6379`)
- [ ] Docker network configured? (`docker network inspect tts-network`)
- [ ] Logs show no errors? (`docker logs redis-tts`)

## Testing Worker Registration

```bash
# Check if workers are registered
redis-cli KEYS "worker:*"

# View worker registration details
redis-cli GET "worker:tts-worker-0"

# Monitor registrations in real-time
redis-cli MONITOR | grep worker:
```

## Related Runbooks

- [gRPC Worker Connectivity](GRPC_WORKER.md)
- [Port Conflicts](PORT_CONFLICTS.md)
- [Environment Setup](ENVIRONMENT.md)
- [Docker Setup](../setup/DOCKER_SETUP.md)

## Advanced Troubleshooting

**Enable Redis slow log:**
```bash
docker exec redis-tts redis-cli CONFIG SET slowlog-log-slower-than 10000
docker exec redis-tts redis-cli SLOWLOG GET 10
```

**Check Redis stats:**
```bash
docker exec redis-tts redis-cli INFO stats
```

**Flush all data (CAUTION):**
```bash
# Only if you need to start fresh
docker exec redis-tts redis-cli FLUSHALL
```

## Still Having Issues?

1. Check Redis logs: `docker logs redis-tts`
2. Verify container networking: `docker network inspect tts-network`
3. Test connection from Python:
   ```python
   import redis
   r = redis.from_url("redis://localhost:6379")
   print(r.ping())  # Should print True
   ```
4. Review registry code: `src/orchestrator/registry.py`
5. Check Redis documentation: https://redis.io/docs/
