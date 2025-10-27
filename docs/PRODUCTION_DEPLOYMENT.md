# Production Deployment Guide

**Last Updated**: 2025-10-26
**Milestone**: M12 Docker Polish

This guide provides step-by-step instructions for deploying the TTS pipeline in production with monitoring, health checks, and best practices.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Production Configuration](#production-configuration)
- [Deployment Steps](#deployment-steps)
- [Monitoring Setup](#monitoring-setup)
- [Health Checks](#health-checks)
- [Security Hardening](#security-hardening)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## Prerequisites

### Hardware Requirements

**Minimum (Piper CPU baseline)**:
- CPU: 4+ cores
- RAM: 8 GB
- GPU: Optional (NVIDIA GPU with 4+ GB VRAM)
- Storage: 20 GB

**Recommended (CosyVoice GPU)**:
- CPU: 8+ cores
- RAM: 16 GB
- GPU: NVIDIA GPU with 8+ GB VRAM (RTX 3090, A100, etc.)
- Storage: 50 GB

### Software Requirements

- **Docker**: 28.x+ with Docker Compose V2
- **NVIDIA Container Runtime**: For GPU support
- **Operating System**: Ubuntu 22.04+ / RHEL 8+ / Debian 11+
- **Network**: Open ports 7880-7882, 8080-8081, 8443-8444

### NVIDIA Docker Setup

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Architecture Overview

### Service Components

```
┌─────────────────────────────────────────────────────────────┐
│                        Load Balancer                         │
│                    (Caddy Reverse Proxy)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
        ┌───────────┐  ┌──────────┐  ┌──────────────┐
        │  LiveKit  │  │  Next.js │  │ Orchestrator │
        │  Server   │  │   Web    │  │   (Agent)    │
        └───────────┘  └──────────┘  └──────────────┘
                                            │
                                            ▼
                                      ┌──────────┐
                                      │  Redis   │
                                      └──────────┘
                                            │
                                            ▼
                                    ┌───────────────┐
                                    │  TTS Workers  │
                                    │ (Piper/Cosy)  │
                                    └───────────────┘
```

### Monitoring Stack

```
┌─────────────┐     ┌────────────┐     ┌──────────────┐
│ Orchestrator├────▶│ Prometheus ├────▶│   Grafana    │
└─────────────┘     └────────────┘     └──────────────┘
       │                  │                    │
       ▼                  ▼                    ▼
┌─────────────┐     ┌────────────┐     ┌──────────────┐
│ TTS Workers │     │   Alerts   │     │  Dashboards  │
└─────────────┘     └────────────┘     └──────────────┘
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-org/full-duplex-voice-chat.git
cd full-duplex-voice-chat
```

### 2. Configure Environment

```bash
# Copy production environment template
cp .env.example .env.prod

# Edit configuration
nano .env.prod
```

**Required Variables**:
```bash
# Model Selection
ADAPTER_TYPE=piper
DEFAULT_MODEL=piper-en-us-lessac-medium

# LiveKit Configuration
LIVEKIT_API_KEY=your-secure-api-key
LIVEKIT_API_SECRET=your-secure-secret-key

# OpenAI (for LLM integration)
OPENAI_API_KEY=sk-your-openai-key

# ASR Configuration
ASR_ENABLED=true
ASR_ADAPTER=whisperx
ASR_DEVICE=auto

# Monitoring
ENVIRONMENT=production
```

### 3. Start Production Stack

```bash
# Start with Piper (CPU baseline)
docker compose -f docker-compose.prod.yml --profile piper up -d

# OR start with CosyVoice (GPU)
docker compose -f docker-compose.prod.yml --profile cosyvoice up -d
```

### 4. Verify Deployment

```bash
# Check all services are healthy
docker compose -f docker-compose.prod.yml ps

# Test health endpoints
curl http://localhost:8081/health
curl http://localhost:8081/metrics

# Check logs
docker compose -f docker-compose.prod.yml logs -f orchestrator
```

## Production Configuration

### Resource Limits

All services have resource limits defined in `docker-compose.prod.yml`:

**Orchestrator**:
- Memory: 4-8 GB
- CPU: 4 cores
- GPU: 1x NVIDIA GPU

**TTS Workers**:
- Piper: 2-4 GB RAM, 2 CPU cores
- CosyVoice: 8-10 GB RAM, 4 CPU cores, 1 GPU

**Redis**:
- Memory: 128-256 MB
- CPU: 0.5 cores

**Caddy**:
- Memory: 128-256 MB
- CPU: 1 core

### Security Configuration

**Non-Root Containers**:
All production containers run as non-root users (UID 1000).

**Read-Only Mounts**:
Application code and configs are mounted read-only where possible.

**Network Isolation**:
Services communicate via isolated Docker network (`tts-network`).

**TLS/HTTPS**:
Caddy provides automatic HTTPS with Let's Encrypt or custom certificates.

### Model Cache Volumes

Persistent volumes eliminate runtime downloads:

- `huggingface-cache`: WhisperX models (~1-2 GB)
- `torch-cache`: PyTorch models (~500 MB)
- `mfa-cache`: Montreal Forced Aligner (~100 MB)
- `modelscope-cache`: WeText FST files (~50 MB)

## Deployment Steps

### Step 1: Prepare Environment

```bash
# Create directories
mkdir -p logs monitoring/grafana/{provisioning,dashboards}

# Generate TLS certificates (for HTTPS)
# Option A: Self-signed (development)
mkcert voicechat.local "*.voicechat.local"

# Option B: Let's Encrypt (production)
# Configure Caddy to use Let's Encrypt (see Caddyfile)
```

### Step 2: Configure Services

**Edit `configs/orchestrator.yaml`**:
```yaml
server:
  host: 0.0.0.0
  port: 8080
  health_check_port: 8081

tts:
  worker_address: "tts:7001"
  default_model: "piper-en-us-lessac-medium"

asr:
  enabled: true
  adapter: "whisperx"
  device: "auto"
```

**Edit `configs/livekit.yaml`**:
```yaml
port: 7880
rtc:
  port_range_start: 50000
  port_range_end: 50099
  use_external_ip: true
```

### Step 3: Build Images

```bash
# Build production images
docker compose -f docker-compose.prod.yml build

# Tag for registry (optional)
docker tag tts-orchestrator:prod registry.example.com/tts-orchestrator:v1.0
docker tag tts-worker:prod registry.example.com/tts-worker:v1.0
```

### Step 4: Start Services

```bash
# Start core services
docker compose -f docker-compose.prod.yml up -d redis livekit caddy

# Wait for health checks
sleep 30

# Start orchestrator and workers
docker compose -f docker-compose.prod.yml --profile piper up -d
```

### Step 5: Verify Health

```bash
# Check service status
docker compose -f docker-compose.prod.yml ps

# Expected output:
# NAME                    STATUS          HEALTH
# redis-tts-prod         Up 30s          healthy
# livekit-server-prod    Up 30s          healthy
# caddy-proxy-prod       Up 30s          healthy
# orchestrator-prod      Up 30s          healthy (after 150s)
# tts-worker-0-prod      Up 30s          healthy (after 60s)

# Test endpoints
curl http://localhost:8081/health
curl http://localhost:8081/metrics/summary
```

## Monitoring Setup

### Start Monitoring Stack

```bash
# Start Prometheus + Grafana
docker compose -f docker-compose.monitoring.yml up -d

# Verify monitoring services
docker compose -f docker-compose.monitoring.yml ps

# Access Grafana: http://localhost:3000
# Default credentials: admin/admin
```

### Configure Grafana

1. **Login**: Navigate to `http://localhost:3000` (admin/admin)
2. **Change Password**: Set secure admin password
3. **Verify Datasource**: Configuration > Datasources > Prometheus (should be green)
4. **Import Dashboard**: Dashboards > Browse > TTS Pipeline Metrics

### Key Metrics to Monitor

**Synthesis Performance**:
- p50/p95/p99 synthesis latency
- SLA compliance percentage (target: >99%)
- Error rate (target: <1%)

**Worker Health**:
- Worker utilization (target: 50-80%)
- Active vs idle workers
- Queue depth (target: <10)

**Session Metrics**:
- Active sessions
- Session duration
- Messages per session
- Barge-in latency (target: <50ms)

### Alerts Configuration

**Create Prometheus Alert Rules** (`monitoring/alerts.yml`):

```yaml
groups:
  - name: tts_sla
    interval: 30s
    rules:
      - alert: HighSynthesisLatency
        expr: histogram_quantile(0.95, rate(synthesis_latency_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Synthesis latency exceeds SLA (p95 > 500ms)"

      - alert: LowSLACompliance
        expr: sla_compliance_percent < 95
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "SLA compliance below 95%"

      - alert: HighErrorRate
        expr: rate(synthesis_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Synthesis error rate > 10%"
```

## Health Checks

### Service Health Endpoints

**Orchestrator**:
- `/health` - Overall health (Redis + Worker)
- `/readiness` - Ready to accept requests
- `/liveness` - Service is alive
- `/metrics` - Prometheus metrics
- `/metrics/summary` - JSON metrics summary

**Example Health Check**:
```bash
curl http://localhost:8081/health

# Response (200 OK):
{
  "status": "healthy",
  "uptime_seconds": 3600.5,
  "redis": true,
  "worker": true,
  "checks": {
    "redis": {"ok": true, "error": null},
    "worker": {"ok": true, "error": null}
  }
}
```

### Docker Health Checks

All services have built-in Docker health checks:

**Redis**: `redis-cli ping`
**LiveKit**: `wget http://localhost:7880/`
**Caddy**: `wget http://localhost:2019/config/`
**Orchestrator**: `curl http://localhost:8081/health`
**TTS Workers**: TCP connectivity test on gRPC port

### Monitoring Health Status

```bash
# Watch health status
watch -n 5 'docker compose -f docker-compose.prod.yml ps'

# View health check logs
docker inspect orchestrator-prod | jq '.[0].State.Health'

# Restart unhealthy service
docker compose -f docker-compose.prod.yml restart orchestrator
```

## Security Hardening

### Container Security

**1. Non-Root Users**:
All services run as non-root (UID 1000).

**2. Read-Only Filesystems**:
```yaml
security_opt:
  - no-new-privileges:true
read_only: true
tmpfs:
  - /tmp
```

**3. Resource Limits**:
Prevent resource exhaustion with CPU/memory limits.

**4. Network Isolation**:
Services communicate via internal Docker network.

### TLS/HTTPS Configuration

**Production Caddy Configuration**:
```
voicechat.example.com {
    tls your-email@example.com
    reverse_proxy livekit:7880
}

*.voicechat.example.com {
    tls your-email@example.com
    reverse_proxy web:3000
}
```

### Secrets Management

**Never commit secrets to git**. Use environment files:

```bash
# .env.prod (not in git)
LIVEKIT_API_KEY=$(openssl rand -hex 32)
LIVEKIT_API_SECRET=$(openssl rand -hex 32)
OPENAI_API_KEY=sk-your-key

# Load from environment file
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d
```

### Firewall Rules

```bash
# Allow only required ports
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw allow 8443/tcp    # Web client
sudo ufw allow 7880/tcp    # LiveKit
sudo ufw allow 7882/udp    # TURN/UDP
sudo ufw allow 50000:50099/udp  # RTC range

# Deny all other incoming
sudo ufw default deny incoming
sudo ufw enable
```

## Troubleshooting

### Common Issues

**Issue 1: Orchestrator fails to start**

```bash
# Check logs
docker compose -f docker-compose.prod.yml logs orchestrator

# Common causes:
# - WhisperX model download timeout (increase start_period)
# - GPU not available (check nvidia-smi)
# - Redis not ready (check redis health)
```

**Issue 2: High synthesis latency**

```bash
# Check Grafana dashboard: Synthesis Latency panel
# Check worker utilization
curl http://localhost:8081/metrics/summary | jq '.metrics.worker_utilization_percent'

# If utilization > 80%, scale workers:
docker compose -f docker-compose.prod.yml up -d --scale tts0=2
```

**Issue 3: SLA violations**

```bash
# Check metrics endpoint
curl http://localhost:8081/metrics/summary | jq '.metrics.sla_violations'

# Root causes:
# - Overloaded workers (scale up)
# - GPU memory exhaustion (reduce RESIDENT_CAP)
# - Network latency (check ping times)
```

### Debug Mode

Enable verbose logging:

```bash
# Edit docker-compose.prod.yml
environment:
  - LOG_LEVEL=DEBUG
  - GRPC_VERBOSITY=DEBUG

# Restart services
docker compose -f docker-compose.prod.yml restart orchestrator
```

### Performance Profiling

```bash
# Enable profiling
environment:
  - ENABLE_PROFILING=true
  - PROFILE_OUTPUT_DIR=/app/profiles
  - PROFILE_MIN_DURATION_MS=100

# Collect profiles (mounted volume)
volumes:
  - ./profiles:/app/profiles

# Analyze with pstats
python -m pstats profiles/synthesis_*.pstats
```

## Maintenance

### Backup Strategy

**Configuration Backup**:
```bash
# Backup configs and environment
tar -czf backup-config-$(date +%Y%m%d).tar.gz \
  configs/ .env.prod docker-compose.prod.yml
```

**Model Cache Backup**:
```bash
# Backup model caches (saves download time on restore)
docker run --rm -v full-duplex-voice-chat_huggingface-cache:/data \
  -v $(pwd):/backup ubuntu tar czf /backup/huggingface-cache.tar.gz /data
```

### Updates and Upgrades

```bash
# Pull latest images
git pull
docker compose -f docker-compose.prod.yml build --pull

# Rolling update (zero-downtime)
docker compose -f docker-compose.prod.yml up -d --no-deps orchestrator
docker compose -f docker-compose.prod.yml up -d --no-deps tts0
```

### Log Rotation

```bash
# Configure Docker log rotation
# /etc/docker/daemon.json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "10"
  }
}

# Restart Docker
sudo systemctl restart docker
```

### Monitoring Data Retention

**Prometheus**:
- Default retention: 30 days
- Adjust in `docker-compose.monitoring.yml`: `--storage.tsdb.retention.time=30d`

**Grafana**:
- Persistent volume: `grafana-data`
- Backup dashboards: Configuration > Import/Export

## Performance Tuning

### GPU Optimization

**CosyVoice GPU Settings**:
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  - RESIDENT_CAP=2  # Max models in VRAM
  - TTL_MS=600000   # Model eviction timeout (10 min)
```

### CPU Optimization

**Piper CPU Settings**:
```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
```

### Network Optimization

**LiveKit RTC Configuration**:
```yaml
rtc:
  port_range_start: 50000
  port_range_end: 50099
  use_external_ip: true
  tcp_port: 7881
  udp_port: 7882
```

## Next Steps

1. **Scale Workers**: Add more TTS workers for horizontal scaling
2. **Multi-Host Deployment**: Deploy across multiple hosts with Docker Swarm/K8s
3. **Load Balancing**: Add HAProxy/nginx for load balancing
4. **CDN Integration**: Use CloudFront/Cloudflare for static assets
5. **Disaster Recovery**: Implement automated backup and restore

## References

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Prometheus Configuration](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)
- [Grafana Dashboard Guide](https://grafana.com/docs/grafana/latest/dashboards/)
- [LiveKit Self-Hosting](https://docs.livekit.io/deploy/)

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-org/full-duplex-voice-chat/issues
- Documentation: docs/
- Monitoring: http://localhost:3000 (Grafana)
