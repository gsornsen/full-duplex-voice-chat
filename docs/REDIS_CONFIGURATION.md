# Redis Configuration Guide

## Overview

The Realtime Duplex Voice Demo uses Redis for worker service discovery and registration. This guide explains how Redis is configured in different environments and how to troubleshoot common issues.

## Docker Environment (Recommended)

### Internal Networking

In Docker Compose, Redis runs on an internal network and is **not exposed** to the host by default. This design:

1. **Prevents port conflicts** with existing Redis/Valkey instances on the host
2. **Improves security** by not exposing Redis externally
3. **Simplifies configuration** using Docker's internal DNS

### Connection URLs

All services inside Docker use the internal DNS name:

```bash
REDIS_URL=redis://redis:6379
```

- `redis` is the Docker service name (resolved via internal DNS)
- `6379` is the standard Redis port inside the container
- No external port mapping needed

### Configuration Files

**docker-compose.yml:**
```yaml
services:
  redis:
    image: redis:7-alpine
    container_name: redis-tts
    # No ports section - internal only
    networks:
      - tts-network
    healthcheck:
      test: ["CMD", "redis-cli", "-h", "localhost", "-p", "6379", "ping"]
```

**Environment variables:**
```yaml
orchestrator:
  environment:
    - REDIS_URL=redis://redis:6379

tts0:
  environment:
    - REDIS_URL=redis://redis:6379
```

## Debugging Redis in Docker

### Accessing Redis CLI

From host machine:
```bash
# Execute redis-cli inside the running container
docker compose exec redis redis-cli

# Or with specific container name
docker exec -it redis-tts redis-cli
```

### Checking Redis Health

```bash
# Check if Redis is running
docker compose ps redis

# View Redis logs
docker compose logs redis

# Test Redis connectivity from orchestrator
docker compose exec orchestrator redis-cli -h redis -p 6379 ping
# Expected output: PONG
```

### Exposing Redis for External Tools (Optional)

If you need to access Redis from the host (e.g., for GUI tools like RedisInsight):

1. Copy the override example:
```bash
cp docker-compose.override.yml.example docker-compose.override.yml
```

2. Edit `docker-compose.override.yml` and uncomment the Redis port mapping:
```yaml
services:
  redis:
    ports:
      - "6380:6379"  # Host:Container
```

3. Restart services:
```bash
docker compose down
docker compose up -d
```

4. Connect from host:
```bash
redis-cli -h localhost -p 6380
```

**Note:** Even with external port exposure, internal services still use `redis:6379` (not `6380`).

## Local Development (Without Docker)

### Prerequisites

Run Redis locally:
```bash
# Option 1: System package
sudo systemctl start redis-server

# Option 2: Docker container
docker run -d --name redis-local -p 6379:6379 redis:7-alpine

# Option 3: Existing Redis/Valkey instance
# Ensure it's running on port 6379
```

### Configuration

Update config files to use localhost:

**configs/orchestrator.yaml:**
```yaml
redis:
  url: "redis://localhost:6379"
```

**configs/worker.yaml:**
```yaml
redis:
  url: "redis://localhost:6379"
```

Or set environment variable:
```bash
export REDIS_URL=redis://localhost:6379
```

## Port Conflict Resolution

### Problem: Port 6379 Already in Use

If you have an existing Redis/Valkey instance:

```bash
$ docker compose up
Error: Bind for 0.0.0.0:6379 failed: port is already allocated
```

### Solution 1: Use Internal Networking (Recommended)

Our default `docker-compose.yml` already handles this by **not exposing** port 6379 to the host. The Redis container runs isolated within the Docker network.

Verify the configuration:
```yaml
redis:
  # No ports section
  networks:
    - tts-network
```

### Solution 2: Change External Port (If Needed)

If you need external access and port 6380 is also taken:

Edit `docker-compose.override.yml`:
```yaml
services:
  redis:
    ports:
      - "16379:6379"  # Use any available port
```

Connect from host:
```bash
redis-cli -h localhost -p 16379
```

**Important:** Internal services still use `redis:6379` - they are unaffected by the external port choice.

### Solution 3: Use Existing Redis Instance

Point all services to your existing Redis:

1. Update `docker-compose.yml`:
```yaml
orchestrator:
  environment:
    - REDIS_URL=redis://host.docker.internal:6379

tts0:
  environment:
    - REDIS_URL=redis://host.docker.internal:6379
```

2. Remove the Redis service from `docker-compose.yml`

3. Ensure existing Redis is accessible from Docker:
```bash
# Test from container
docker run --rm redis:7-alpine redis-cli -h host.docker.internal -p 6379 ping
```

## Health Checks

Redis health is verified before starting dependent services:

```yaml
redis:
  healthcheck:
    test: ["CMD", "redis-cli", "-h", "localhost", "-p", "6379", "ping"]
    interval: 10s
    timeout: 3s
    retries: 5
    start_period: 5s
```

Services wait for Redis to be healthy:
```yaml
orchestrator:
  depends_on:
    redis:
      condition: service_healthy
```

## Troubleshooting

### Issue: "Connection refused" to Redis

**Symptoms:**
```
redis.exceptions.ConnectionError: Error connecting to redis:6379
```

**Diagnosis:**
```bash
# Check if Redis is running
docker compose ps redis

# Check Redis logs
docker compose logs redis

# Test connectivity
docker compose exec orchestrator ping -c 2 redis
```

**Solutions:**
1. Ensure Redis service is healthy: `docker compose ps`
2. Check network configuration in docker-compose.yml
3. Verify REDIS_URL uses correct service name: `redis:6379`

### Issue: "WRONGPASS Authentication failed"

**Symptoms:**
```
redis.exceptions.AuthenticationError: Authentication failed
```

**Solution:**
Redis in docker-compose.yml has no password by default. If you've added password protection, update connection URLs:

```yaml
environment:
  - REDIS_URL=redis://:password@redis:6379
```

### Issue: "No route to host"

**Symptoms:**
```
redis.exceptions.ConnectionError: Error 113 connecting to redis:6379
```

**Diagnosis:**
```bash
# Check if services are on the same network
docker network inspect full-duplex-voice-chat_tts-network

# Verify both orchestrator and redis are listed
```

**Solution:**
Ensure all services use the same network in docker-compose.yml:
```yaml
networks:
  - tts-network
```

### Issue: Existing Host Redis Conflict

**Symptoms:**
```
Error response from daemon: driver failed programming external connectivity
```

**Solution:**
This is already handled! Our default configuration doesn't expose Redis to the host, so there's no conflict with existing Redis instances.

If you manually added port mapping and see this error:
1. Remove the `ports:` section from Redis service
2. Or use a different external port: `- "6380:6379"`

## Performance Tuning

For production deployments:

### Redis Persistence

Add persistence configuration:
```yaml
redis:
  image: redis:7-alpine
  command: redis-server --appendonly yes
  volumes:
    - redis-data:/data

volumes:
  redis-data:
```

### Connection Pooling

Application-side configuration (worker.yaml, orchestrator.yaml):
```yaml
redis:
  connection_pool_size: 10  # Adjust based on load
  socket_timeout: 5         # Seconds
  socket_connect_timeout: 3 # Seconds
```

### Monitoring

Access Redis INFO command:
```bash
docker compose exec redis redis-cli INFO

# Key metrics:
# - connected_clients
# - used_memory
# - keyspace_hits/misses
```

## Multi-GPU and Multi-Host Setup

### Same Host, Multiple Workers

All workers connect to the same Redis instance:
```yaml
tts0:
  environment:
    - REDIS_URL=redis://redis:6379

tts1:
  environment:
    - REDIS_URL=redis://redis:6379
```

### Multi-Host Deployment

Run Redis on a dedicated host:

1. Expose Redis with authentication:
```yaml
redis:
  image: redis:7-alpine
  command: redis-server --requirepass ${REDIS_PASSWORD}
  ports:
    - "6379:6379"
```

2. Configure workers on other hosts:
```yaml
environment:
  - REDIS_URL=redis://:password@redis-host.lan:6379
```

3. Consider Redis Sentinel or Cluster for HA

## Security Best Practices

1. **No external exposure** (default configuration)
2. **Use authentication** for multi-host setups
3. **Enable TLS** for WAN deployments
4. **Use Docker secrets** for passwords:

```yaml
services:
  redis:
    secrets:
      - redis_password
    environment:
      - REDIS_PASSWORD_FILE=/run/secrets/redis_password

secrets:
  redis_password:
    file: ./secrets/redis_password.txt
```

## Summary

**Default Setup (Docker Compose):**
- Redis runs on internal network only
- No port conflicts with host Redis
- Services use `redis://redis:6379`
- No external access needed for normal operation

**Debugging:**
- Use `docker compose exec redis redis-cli`
- Check logs: `docker compose logs redis`
- Test connectivity from other services

**Optional External Access:**
- Uncomment port mapping in docker-compose.override.yml
- Connect from host: `redis-cli -h localhost -p 6380`
- Internal services unaffected (still use `redis:6379`)

For more troubleshooting, see:
- `docs/runbooks/ENVIRONMENT.md`
- `docs/runbooks/LOG_DEBUGGING.md`
- `docs/TESTING_GUIDE.md`
