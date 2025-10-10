# Runbook: Environment Setup & Validation

**Time to Resolution:** 15-30 minutes (first time), < 5 minutes (recurring)
**Severity:** Critical (blocks all development)
**Related:** [WebSocket Errors](WEBSOCKET.md), [Log Debugging](LOG_DEBUGGING.md)

---

## Overview

This runbook covers common environment setup and validation issues for the M2 Realtime Duplex Voice Demo system.

**Prerequisites:**
- Python 3.13.x
- Docker Engine 28.x with NVIDIA runtime (GPU workers)
- uv package manager
- Redis (via Docker)

---

## Quick Start Checklist

```bash
# 1. Check Python version
python3.13 --version
# Expected: Python 3.13.x

# 2. Check Docker
docker --version && docker ps
# Expected: Docker version 28.x, no permission errors

# 3. Check uv installed
uv --version
# Expected: uv x.x.x

# 4. Run pre-flight check
./scripts/preflight-check.sh
# Expected: All checks pass

# 5. Install dependencies
uv sync
# Expected: Dependencies installed successfully

# 6. Generate protos
just gen-proto
# Expected: gRPC stubs generated

# 7. Validate configs
uv run python scripts/validate-config.py
# Expected: All configurations valid
```

---

## Common Issues

### 1. Python 3.13 Not Found

**Symptom:**
```bash
$ python3.13 --version
bash: python3.13: command not found
```

**Resolution (Ubuntu/Debian):**
```bash
# Add deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.13
sudo apt install python3.13 python3.13-venv python3.13-dev

# Verify installation
python3.13 --version
```

**Resolution (macOS with Homebrew):**
```bash
# Install Python 3.13
brew install python@3.13

# Link to python3.13
brew link python@3.13

# Verify
python3.13 --version
```

**Resolution (pyenv - cross-platform):**
```bash
# Install pyenv if needed
curl https://pyenv.run | bash

# Install Python 3.13
pyenv install 3.13.0

# Set global version
pyenv global 3.13.0

# Verify
python --version  # Should show 3.13.x
```

---

### 2. uv Not Installed

**Symptom:**
```bash
$ uv --version
bash: uv: command not found
```

**Resolution (Linux/macOS):**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell or source profile
source ~/.bashrc  # or ~/.zshrc for zsh

# Verify
uv --version
```

**Resolution (Windows):**
```powershell
# Using PowerShell
irm https://astral.sh/uv/install.ps1 | iex

# Verify
uv --version
```

**Alternative (pip):**
```bash
pip install uv

# Verify
uv --version
```

---

### 3. Docker Not Installed or Not Running

**Symptom:**
```bash
$ docker ps
Cannot connect to the Docker daemon. Is the docker daemon running?
```

**Resolution (Ubuntu/Debian):**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (avoid sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker ps
```

**Resolution (macOS):**
```bash
# Install Docker Desktop
brew install --cask docker

# Start Docker Desktop from Applications
# Wait for Docker to start (whale icon in menu bar)

# Verify
docker ps
```

**Resolution (Windows):**
1. Download Docker Desktop from https://www.docker.com/products/docker-desktop
2. Install and restart computer
3. Start Docker Desktop
4. Verify: `docker ps` in PowerShell

**If Docker running but permission denied:**
```bash
# Check if user in docker group
groups | grep docker

# If not, add user
sudo usermod -aG docker $USER

# Apply group membership (logout/login or use)
newgrp docker

# Verify
docker ps  # Should work without sudo
```

---

### 4. Docker Daemon Not Starting

**Symptom:**
```bash
$ sudo systemctl status docker
● docker.service - Docker Application Container Engine
   Loaded: loaded
   Active: failed
```

**Check logs:**
```bash
sudo journalctl -u docker -n 50 --no-pager
```

**Common causes:**

**a) Port conflict:**
```bash
# Check if port 2375/2376 in use
sudo lsof -i :2375
sudo lsof -i :2376

# Kill conflicting process or reconfigure Docker
```

**b) Storage driver issue:**
```bash
# Check Docker data directory
ls -la /var/lib/docker

# If corrupted, backup and remove
sudo systemctl stop docker
sudo mv /var/lib/docker /var/lib/docker.backup
sudo mkdir /var/lib/docker
sudo systemctl start docker
```

**c) Configuration error:**
```bash
# Validate Docker daemon config
cat /etc/docker/daemon.json | jq

# If invalid, fix or remove
sudo mv /etc/docker/daemon.json /etc/docker/daemon.json.bak
sudo systemctl restart docker
```

---

### 5. NVIDIA Runtime Not Configured

**Symptom (GPU workers):**
```bash
$ docker run --rm --gpus all nvidia/cuda:12.8.0-base nvidia-smi
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```

**Check NVIDIA drivers:**
```bash
nvidia-smi
# Should show GPU info
```

**Install nvidia-container-toolkit:**
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.8.0-base nvidia-smi
```

---

### 6. Dependencies Installation Failed

**Symptom:**
```bash
$ uv sync
error: Failed to prepare distributions
```

**Check Python version:**
```bash
python3.13 --version
# Must be 3.13.x
```

**Clear cache and retry:**
```bash
# Remove virtual environment
rm -rf .venv

# Clear uv cache
uv cache clean

# Retry installation
uv sync

# Verify
uv run python --version
```

**If specific package fails:**
```bash
# Check for missing system dependencies
# For example, grpcio needs:
sudo apt-get install -y build-essential python3.13-dev

# Retry
uv sync
```

---

### 7. gRPC Stubs Not Generated

**Symptom:**
```bash
$ uv run python src/orchestrator/server.py
ModuleNotFoundError: No module named 'rpc.generated'
```

**Generate stubs:**
```bash
just gen-proto

# Or manually
uv run python -m grpc_tools.protoc \
  -I src/rpc \
  --python_out=src/rpc/generated \
  --grpc_python_out=src/rpc/generated \
  --mypy_out=src/rpc/generated \
  src/rpc/tts.proto
```

**Verify:**
```bash
ls -la src/rpc/generated/
# Should contain: tts_pb2.py, tts_pb2_grpc.py, tts_pb2.pyi
```

---

### 8. Port Already in Use

**Symptom:**
```bash
ERROR: Port 8080 already in use
```

**Find process using port:**
```bash
# Linux/macOS
lsof -i :8080

# Or
netstat -tuln | grep 8080

# Or
ss -tuln | grep 8080
```

**Kill process:**
```bash
# Find PID
lsof -t -i :8080

# Kill
kill -9 $(lsof -t -i :8080)
```

**Or change port in config:**
```yaml
# configs/orchestrator.yaml
transport:
  websocket:
    port: 8090  # Use different port
```

**Common port conflicts:**
- 6379: Redis (another Redis instance)
- 7001: gRPC worker (previous worker not stopped)
- 8080: Orchestrator WS (old orchestrator or other app)
- 8081: Health endpoint (monitoring tool)

**Resolution:**
```bash
# Check all required ports
for port in 6379 7001 8080 8081; do
  if lsof -i :$port > /dev/null 2>&1; then
    echo "❌ Port $port in use"
    lsof -i :$port
  else
    echo "✅ Port $port available"
  fi
done
```

---

### 9. Virtual Environment Path Issues

**Symptom:**
```bash
$ uv run python script.py
Error: No virtual environment found
```

**Check virtual environment:**
```bash
ls -la .venv/
# Should exist and contain bin/python
```

**Recreate if missing:**
```bash
rm -rf .venv
uv sync
```

**Activate manually (alternative to uv run):**
```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Verify
which python
# Should show: /path/to/project/.venv/bin/python
```

---

### 10. Permission Denied Errors

**Symptom:**
```bash
$ ./scripts/preflight-check.sh
bash: ./scripts/preflight-check.sh: Permission denied
```

**Make scripts executable:**
```bash
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

**Docker permission denied:**
```bash
$ docker ps
permission denied while trying to connect to the Docker daemon socket
```

**Fix:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Apply immediately (or logout/login)
newgrp docker

# Verify
docker ps  # Should work without sudo
```

---

### 11. Configuration Validation Errors

**Symptom:**
```bash
$ uv run python scripts/validate-config.py
❌ configs/worker.yaml: Invalid worker configuration
```

**Common validation errors:**

**a) Missing required field:**
```yaml
# Error: model_manager.default_model_id field required
# Fix: Add default_model_id
model_manager:
  default_model_id: "cosyvoice2-en-base"  # Add this
```

**b) Invalid value range:**
```yaml
# Error: grpc_port must be between 1024-65535
worker:
  grpc_port: 80  # Too low

# Fix: Use valid port
worker:
  grpc_port: 7001  # Valid
```

**c) Invalid type:**
```yaml
# Error: max_concurrent_sessions must be an integer
worker:
  capabilities:
    max_concurrent_sessions: "3"  # String, not int

# Fix: Remove quotes
worker:
  capabilities:
    max_concurrent_sessions: 3  # Integer
```

**Validate and fix:**
```bash
# Run validator
uv run python scripts/validate-config.py

# Check example configs
cat configs/worker.yaml
cat configs/orchestrator.yaml

# Reference documentation
cat docs/CONFIGURATION_REFERENCE.md
```

---

## Environment Validation Workflow

**Complete setup validation:**

```bash
#!/bin/bash
# Full environment setup validation

echo "=== Environment Validation ==="

# 1. Check Python
echo -n "Python 3.13: "
if command -v python3.13 > /dev/null 2>&1; then
  echo "✅ $(python3.13 --version)"
else
  echo "❌ Not found"
  echo "Install: sudo apt install python3.13"
  exit 1
fi

# 2. Check uv
echo -n "uv: "
if command -v uv > /dev/null 2>&1; then
  echo "✅ $(uv --version)"
else
  echo "❌ Not found"
  echo "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

# 3. Check Docker
echo -n "Docker: "
if docker ps > /dev/null 2>&1; then
  echo "✅ $(docker --version)"
else
  echo "❌ Not accessible"
  echo "Fix: sudo systemctl start docker && sudo usermod -aG docker $USER"
  exit 1
fi

# 4. Check NVIDIA (optional for GPU)
echo -n "NVIDIA GPU: "
if command -v nvidia-smi > /dev/null 2>&1; then
  echo "✅ $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  echo "⚠️  Not available (OK for CPU-only setup)"
fi

# 5. Check dependencies
echo -n "Dependencies: "
if [ -d ".venv" ]; then
  echo "✅ Installed"
else
  echo "⚠️  Not installed"
  echo "Run: uv sync"
  exit 1
fi

# 6. Check gRPC stubs
echo -n "gRPC stubs: "
if [ -f "src/rpc/generated/tts_pb2.py" ]; then
  echo "✅ Generated"
else
  echo "⚠️  Not generated"
  echo "Run: just gen-proto"
  exit 1
fi

# 7. Check port availability
echo -n "Ports: "
PORTS_OK=true
for port in 6379 7001 8080 8081; do
  if lsof -i :$port > /dev/null 2>&1; then
    echo "❌ Port $port in use"
    PORTS_OK=false
  fi
done
if $PORTS_OK; then
  echo "✅ All available (6379, 7001, 8080, 8081)"
fi

# 8. Validate configurations
echo -n "Configurations: "
if uv run python scripts/validate-config.py > /dev/null 2>&1; then
  echo "✅ Valid"
else
  echo "❌ Invalid"
  echo "Run: uv run python scripts/validate-config.py"
  exit 1
fi

echo ""
echo "✅ Environment ready!"
echo ""
echo "Next steps:"
echo "  1. Start Redis: just redis"
echo "  2. Start worker: just run-tts-sesame"
echo "  3. Start orchestrator: just run-orch"
echo "  4. Test: uv run python scripts/test-connection.py"
```

Save as `scripts/validate-environment.sh` and run:
```bash
chmod +x scripts/validate-environment.sh
./scripts/validate-environment.sh
```

---

## First-Time Setup Guide

**Complete first-time setup:**

```bash
# 1. Clone repository
git clone https://github.com/your-org/full-duplex-voice-chat.git
cd full-duplex-voice-chat

# 2. Install Python 3.13 (if needed)
# See "Python 3.13 Not Found" section above

# 3. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 4. Install Docker
# See "Docker Not Installed" section above

# 5. Install dependencies
uv sync

# 6. Generate gRPC stubs
just gen-proto

# 7. Validate configuration
uv run python scripts/validate-config.py

# 8. Start Redis
just redis

# 9. Verify Redis
redis-cli -u redis://localhost:6379 ping
# Expected: PONG

# 10. Start TTS worker
just run-tts-sesame

# 11. Start orchestrator (in new terminal)
just run-orch

# 12. Test connections
uv run python scripts/test-connection.py

# 13. Test with CLI client
just cli HOST="ws://localhost:8080"
```

---

## Troubleshooting Tips

### General Debugging

```bash
# Check all service status
docker ps -a

# Check all logs
docker compose logs --tail=50

# Check disk space
df -h

# Check memory
free -h

# Check network
netstat -tuln | grep -E '6379|7001|8080'
```

### Clean Reset

**If all else fails, clean reset:**

```bash
# Stop all containers
docker compose down

# Remove volumes (WARNING: deletes data)
docker compose down -v

# Clean Python environment
rm -rf .venv
rm -rf src/rpc/generated

# Reinstall
uv sync
just gen-proto

# Restart services
just redis
sleep 5
just run-tts-sesame &
sleep 5
just run-orch &

# Test
sleep 5
uv run python scripts/test-connection.py
```

---

## Prevention

### Automated Checks

**Add pre-commit hook:**
```bash
# .git/hooks/pre-commit
#!/bin/bash
./scripts/validate-environment.sh
uv run python scripts/validate-config.py
just lint
```

### CI/CD Integration

**GitHub Actions example:**
```yaml
- name: Environment Validation
  run: |
    ./scripts/validate-environment.sh
    uv run python scripts/validate-config.py
    just ci
```

---

## Related Runbooks

- **[WebSocket Errors](WEBSOCKET.md)** - Connection troubleshooting
- **[Redis Connection](REDIS.md)** - Redis setup and issues
- **[Port Conflicts](PORTS.md)** - Port management
- **[Log Debugging](LOG_DEBUGGING.md)** - Log analysis

---

## Further Help

**Still having issues?**

1. Run full diagnostic: `./scripts/preflight-check.sh`
2. Check logs: `docker compose logs`
3. Validate configs: `uv run python scripts/validate-config.py`
4. Review docs: `docs/CONFIGURATION_REFERENCE.md`
5. File issue with: OS version, Python version, error logs
