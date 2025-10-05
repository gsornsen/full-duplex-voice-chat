# Docker Setup & Troubleshooting Guide

## Overview

This guide covers Docker installation, configuration, and troubleshooting for the Realtime Duplex Voice Demo system. Docker is the recommended deployment method for both development and production.

**What You'll Learn:**
- Installing Docker Engine and Docker Desktop
- Setting up NVIDIA Container Runtime for GPU support
- Verifying your Docker installation
- Common errors and resolutions
- Container cleanup and maintenance

---

## Prerequisites

### System Requirements

**Operating System:**
- Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+, Arch, etc.)
- macOS 11+ (Big Sur or later)
- Windows 10/11 with WSL2

**Hardware:**
- **CPU:** 2+ cores (4+ recommended)
- **RAM:** 4GB minimum (8GB+ recommended)
- **Disk:** 20GB free space minimum
- **GPU (optional):** NVIDIA GPU with 8GB+ VRAM for TTS workers

**Network:**
- Ports available: 6379 (Redis), 7001+ (TTS workers), 8080 (orchestrator)

---

## Installation

### Option 1: Docker Desktop (macOS, Windows)

**Recommended for:** Development on macOS and Windows

#### macOS Installation

1. **Download Docker Desktop:**
   - Visit: https://www.docker.com/products/docker-desktop/
   - Download "Docker Desktop for Mac" (Intel or Apple Silicon)

2. **Install:**
   ```bash
   # Open the downloaded .dmg file
   # Drag Docker.app to Applications folder
   # Launch Docker from Applications
   ```

3. **First Launch:**
   - Accept the service agreement
   - Provide admin password for privilege escalation
   - Wait for Docker daemon to start (whale icon in menu bar)

4. **Verify Installation:**
   ```bash
   docker --version
   # Expected: Docker version 24.0.0 or higher

   docker info
   # Should show Docker daemon info without errors
   ```

#### Windows Installation

1. **Enable WSL2:**
   ```powershell
   # Run as Administrator
   wsl --install
   wsl --set-default-version 2
   ```

2. **Download Docker Desktop:**
   - Visit: https://www.docker.com/products/docker-desktop/
   - Download "Docker Desktop for Windows"

3. **Install:**
   - Run the installer
   - Choose "Use WSL 2 instead of Hyper-V" when prompted
   - Restart computer when installation completes

4. **Configure WSL2 Integration:**
   - Open Docker Desktop settings
   - Navigate to Resources → WSL Integration
   - Enable integration for your WSL2 distributions

5. **Verify Installation:**
   ```powershell
   docker --version
   docker info
   ```

---

### Option 2: Docker Engine (Linux)

**Recommended for:** Production servers and Linux development

#### Ubuntu/Debian Installation

```bash
# 1. Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# 2. Update package index
sudo apt-get update

# 3. Install prerequisites
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# 4. Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 5. Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 6. Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 7. Verify installation
sudo docker --version
sudo docker run hello-world
```

#### RHEL/CentOS/Fedora Installation

```bash
# 1. Remove old versions
sudo yum remove docker \
    docker-client \
    docker-client-latest \
    docker-common \
    docker-latest \
    docker-latest-logrotate \
    docker-logrotate \
    docker-engine

# 2. Install yum-utils
sudo yum install -y yum-utils

# 3. Add Docker repository
sudo yum-config-manager \
    --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo

# 4. Install Docker Engine
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 5. Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# 6. Verify installation
sudo docker --version
sudo docker run hello-world
```

#### Arch Linux Installation

```bash
# Install Docker
sudo pacman -S docker docker-compose

# Start and enable Docker daemon
sudo systemctl start docker.service
sudo systemctl enable docker.service

# Verify installation
sudo docker --version
sudo docker run hello-world
```

---

## User Permissions (Linux)

**Problem:** Running `docker` commands requires `sudo` (inconvenient for development).

**Solution:** Add your user to the `docker` group.

### Add User to Docker Group

```bash
# 1. Create docker group (if it doesn't exist)
sudo groupadd docker

# 2. Add your user to the docker group
sudo usermod -aG docker $USER

# 3. Apply the new group membership (choose one):

# Option A: Log out and log back in

# Option B: Use newgrp (temporary for current shell)
newgrp docker

# Option C: Restart the session
# Close terminal and open a new one
```

### Verify Permissions

```bash
# Test without sudo
docker run hello-world

# If successful, you should see:
# "Hello from Docker!"
```

**If still getting permission denied:**
```bash
# Check if docker daemon is running
systemctl status docker

# Check your user is in docker group
groups | grep docker

# Ensure socket permissions are correct
ls -la /var/run/docker.sock
# Should show: srw-rw---- ... root docker
```

---

## NVIDIA Container Runtime (GPU Support)

**Required for:** Running TTS workers on GPU

**Prerequisites:**
- NVIDIA GPU with 8GB+ VRAM
- NVIDIA drivers installed (515+ recommended)
- Linux OS (Docker Desktop on macOS/Windows doesn't support GPU passthrough to containers)

### Installation

#### Ubuntu/Debian

```bash
# 1. Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. Configure Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker

# 4. Restart Docker daemon
sudo systemctl restart docker

# 5. Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

**Expected Output:**
```
+-------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx             Driver Version: 535.xx.xx   CUDA Version: 12.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name                     | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
...
```

#### RHEL/CentOS/Fedora

```bash
# 1. Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/nvidia-container-toolkit.repo | \
    sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# 2. Install nvidia-container-toolkit
sudo yum install -y nvidia-container-toolkit

# 3. Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker

# 4. Restart Docker
sudo systemctl restart docker

# 5. Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

## Docker Daemon Verification

### Check Daemon Status

```bash
# Systemd-based systems (Ubuntu, Debian, RHEL, etc.)
systemctl status docker

# Expected output:
# ● docker.service - Docker Application Container Engine
#    Loaded: loaded (/lib/systemd/system/docker.service; enabled; vendor preset: enabled)
#    Active: active (running) since ...
```

### Start/Stop/Restart Daemon

```bash
# Start
sudo systemctl start docker

# Stop
sudo systemctl stop docker

# Restart
sudo systemctl restart docker

# Enable on boot
sudo systemctl enable docker

# Disable on boot
sudo systemctl disable docker
```

### Check Docker Info

```bash
docker info

# Look for:
# Server Version: 24.0.0 or higher
# Storage Driver: overlay2 (recommended)
# Cgroup Driver: systemd (recommended)
# Runtimes: runc, nvidia (if GPU installed)
```

---

## Running the System

### Quick Start with Docker Compose

```bash
# 1. Navigate to project directory
cd /home/gerald/git/full-duplex-voice-chat

# 2. Start all services
docker compose up --build

# 3. View logs
docker compose logs -f

# 4. Stop services
docker compose down
```

**Services Started:**
- Redis (port 6379)
- Orchestrator (port 8080)
- TTS Worker (port 7001, GPU 0)

### Verify Services Running

```bash
# List running containers
docker ps

# Expected output:
# CONTAINER ID   IMAGE                    STATUS         PORTS
# abc123...      full-duplex-orch         Up 2 minutes   0.0.0.0:8080->8080/tcp
# def456...      full-duplex-tts-worker   Up 2 minutes   0.0.0.0:7001->7001/tcp
# ghi789...      redis:7-alpine           Up 2 minutes   0.0.0.0:6379->6379/tcp
```

### Test Connection

```bash
# Test orchestrator WebSocket
just cli

# Or manually
python -m src.client.cli_client --host ws://localhost:8080
```

---

## Common Errors and Resolutions

### 1. Cannot Connect to Docker Daemon

**Error Message:**
```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock.
Is the docker daemon running?
```

**Cause:** Docker daemon is not running.

**Resolution:**

```bash
# Check daemon status
systemctl status docker

# If inactive, start it
sudo systemctl start docker

# Enable on boot
sudo systemctl enable docker

# Verify
docker info
```

---

### 2. Permission Denied While Connecting

**Error Message:**
```
Got permission denied while trying to connect to the Docker daemon socket
at unix:///var/run/docker.sock: Get "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/containers/json":
dial unix /var/run/docker.sock: connect: permission denied
```

**Cause:** User not in `docker` group.

**Resolution:**

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and log back in, or use:
newgrp docker

# Verify
docker run hello-world
```

---

### 3. Error Response: Could Not Select Device Driver

**Error Message:**
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```

**Cause:** NVIDIA container runtime not installed.

**Resolution:**

```bash
# Install nvidia-container-toolkit (see GPU Support section above)
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

### 4. Container Name Conflict

**Error Message:**
```
docker: Error response from daemon: Conflict. The container name "/orchestrator" is already in use
by container "abc123...". You have to remove (or rename) that container to be able to reuse that name.
```

**Cause:** Previous container with same name not cleaned up.

**Resolution:**

```bash
# Option A: Remove specific container
docker rm -f orchestrator

# Option B: Remove all stopped containers
docker container prune

# Option C: Full compose cleanup
docker compose down

# Then restart
docker compose up --build
```

---

### 5. Port Already Allocated

**Error Message:**
```
Error starting userland proxy: listen tcp4 0.0.0.0:8080: bind: address already in use
```

**Cause:** Another process using the port.

**Resolution:**

```bash
# Find process using port 8080
sudo lsof -i :8080
# Or
sudo netstat -tuln | grep 8080

# Kill the process
sudo kill -9 <PID>

# Or change port in docker-compose.yml
services:
  orchestrator:
    ports:
      - "8081:8080"  # Changed host port to 8081
```

**Check All Ports:**
```bash
# Check required ports: 6379, 7001, 8080
sudo lsof -i :6379
sudo lsof -i :7001
sudo lsof -i :8080
```

---

### 6. No Space Left on Device

**Error Message:**
```
write /var/lib/docker/...: no space left on device
```

**Cause:** Docker storage full.

**Resolution:**

```bash
# Check Docker disk usage
docker system df

# Remove unused data
docker system prune -a

# Remove specific items:
docker container prune  # Stopped containers
docker image prune -a   # Unused images
docker volume prune     # Unused volumes
docker network prune    # Unused networks

# Check host disk space
df -h
```

---

### 7. Failed to Pull Image

**Error Message:**
```
Error response from daemon: manifest for nvidia/cuda:12.8.0-base-ubuntu22.04 not found:
manifest unknown: manifest unknown
```

**Cause:** Image tag doesn't exist or network issue.

**Resolution:**

```bash
# Check available tags at Docker Hub
# Visit: https://hub.docker.com/r/nvidia/cuda/tags

# Try alternative tag
docker pull nvidia/cuda:12.8.0-base-ubuntu22.04

# Check network connectivity
ping registry-1.docker.io

# Check Docker Hub login (if using private images)
docker login
```

---

### 8. Container Exits Immediately

**Error Message:**
```
orchestrator exited with code 1
```

**Cause:** Application error inside container.

**Resolution:**

```bash
# Check container logs
docker logs orchestrator

# Or with compose
docker compose logs orchestrator

# Run container in interactive mode for debugging
docker run -it --rm full-duplex-orch /bin/bash

# Check for common issues:
# - Missing environment variables
# - Configuration file errors
# - Port conflicts
# - Dependency issues
```

---

### 9. GPU Not Available in Container

**Error Message:**
```
RuntimeError: CUDA not available
```

**Cause:** GPU not passed to container.

**Resolution:**

```bash
# Verify nvidia-smi works on host
nvidia-smi

# Test GPU in container
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# If docker-compose.yml, ensure deploy.resources section:
services:
  tts-worker:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

# Or use --gpus flag
docker run --gpus all ...
```

---

### 10. Redis Connection Refused

**Error Message:**
```
redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379. Connection refused.
```

**Cause:** Redis not running or wrong host.

**Resolution:**

```bash
# Check if Redis container running
docker ps | grep redis

# If not, start it
docker compose up -d redis

# Verify Redis accessible
docker exec -it redis redis-cli ping
# Expected: PONG

# In compose, use service name, not localhost
redis:
  url: "redis://redis:6379"  # ✓ Use service name
  # NOT: "redis://localhost:6379"  # ✗ Wrong in Docker network
```

---

## Container Cleanup

### Stop All Containers

```bash
# Stop all running containers
docker stop $(docker ps -q)

# Or with compose
docker compose down
```

### Remove Containers

```bash
# Remove all stopped containers
docker container prune

# Remove specific container
docker rm <container_id_or_name>

# Force remove running container
docker rm -f <container_id_or_name>
```

### Remove Images

```bash
# Remove unused images
docker image prune -a

# Remove specific image
docker rmi <image_id_or_name>

# Remove with force
docker rmi -f <image_id_or_name>
```

### Remove Volumes

```bash
# WARNING: This deletes data!

# Remove unused volumes
docker volume prune

# Remove specific volume
docker volume rm <volume_name>

# With compose (removes named volumes)
docker compose down -v
```

### Complete System Cleanup

```bash
# WARNING: This removes EVERYTHING!

# Remove all stopped containers, unused networks, dangling images, and build cache
docker system prune -a

# Include volumes (deletes all data)
docker system prune -a --volumes

# Check reclaimed space
docker system df
```

---

## Troubleshooting Checklist

Before seeking help, verify:

### Basic Checks
- [ ] Docker daemon is running: `systemctl status docker`
- [ ] User in docker group: `groups | grep docker`
- [ ] Docker version ≥ 24.0: `docker --version`
- [ ] Can run hello-world: `docker run hello-world`

### Network Checks
- [ ] No port conflicts: `sudo lsof -i :6379,7001,8080`
- [ ] Firewall not blocking ports
- [ ] Can reach Docker Hub: `ping registry-1.docker.io`

### GPU Checks (if using GPU)
- [ ] NVIDIA drivers installed: `nvidia-smi`
- [ ] nvidia-container-toolkit installed
- [ ] GPU accessible in container: `docker run --rm --gpus all nvidia/cuda:12.8.0-base nvidia-smi`

### Service Checks
- [ ] Redis running: `docker ps | grep redis`
- [ ] Orchestrator running: `docker ps | grep orch`
- [ ] TTS worker running: `docker ps | grep tts`
- [ ] Logs show no errors: `docker compose logs`

### Resource Checks
- [ ] Sufficient disk space: `df -h`
- [ ] Sufficient RAM: `free -h`
- [ ] Docker storage not full: `docker system df`

---

## Docker Compose Reference

### Common Commands

```bash
# Start services (build if needed)
docker compose up --build

# Start in background (detached)
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs

# Follow logs (tail -f)
docker compose logs -f

# Logs for specific service
docker compose logs orchestrator

# Restart specific service
docker compose restart orchestrator

# Rebuild specific service
docker compose up -d --build orchestrator

# Scale service (multiple instances)
docker compose up -d --scale tts-worker=3

# Execute command in running container
docker compose exec orchestrator /bin/bash

# View resource usage
docker compose stats
```

### docker-compose.yml Structure

```yaml
version: '3.8'

services:
  # Redis service
  redis:
    image: redis:7-alpine
    container_name: redis
    ports:
      - "6379:6379"
    restart: unless-stopped

  # Orchestrator service
  orchestrator:
    build:
      context: .
      dockerfile: Dockerfile.orchestrator
    container_name: orchestrator
    ports:
      - "8080:8080"
    environment:
      - LOG_LEVEL=INFO
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    restart: unless-stopped
    volumes:
      - ./configs/orchestrator.yaml:/app/configs/orchestrator.yaml:ro

  # TTS Worker service
  tts-worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: tts-worker
    ports:
      - "7001:7001"
    environment:
      - LOG_LEVEL=INFO
      - WORKER_NAME=tts-worker-0
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    restart: unless-stopped
    volumes:
      - ./configs/worker.yaml:/app/configs/worker.yaml:ro
      - ./voicepacks:/app/voicepacks:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Performance Optimization

### Build Cache

```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker compose build

# Or export permanently
export DOCKER_BUILDKIT=1
```

### Resource Limits

```yaml
services:
  orchestrator:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Network Mode

```yaml
# For better performance on single host
services:
  orchestrator:
    network_mode: host  # Use host network (no NAT overhead)
    # Note: ports section ignored in host mode
```

---

## Related Documentation

- [Quick Start Guide](../QUICKSTART.md) - System setup after Docker is ready
- [Configuration Reference](../CONFIGURATION_REFERENCE.md) - Docker environment configuration
- [CLI Client Guide](../CLI_CLIENT_GUIDE.md) - Testing with CLI client
- [WebSocket Protocol](../WEBSOCKET_PROTOCOL.md) - Protocol for troubleshooting

---

## Additional Resources

**Official Docker Documentation:**
- Installation: https://docs.docker.com/engine/install/
- Docker Compose: https://docs.docker.com/compose/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

**Community Help:**
- Docker Forums: https://forums.docker.com/
- Stack Overflow: https://stackoverflow.com/questions/tagged/docker

---

## Changelog

**v0.2.0 (M2):**
- Initial Docker setup guide
- Installation instructions for major platforms
- NVIDIA GPU support setup
- Common error resolutions
- Container cleanup procedures

**Future:**
- Docker Swarm deployment (M9+)
- Kubernetes manifests (M10+)
- Multi-host networking guide
