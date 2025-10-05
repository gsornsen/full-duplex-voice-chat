#!/usr/bin/env bash
# Pre-flight check for M2 Realtime Duplex Voice Demo
# Validates system prerequisites and common setup issues

set -e

echo "=== M2 Pre-Flight Check ==="
echo ""

EXIT_CODE=0

# 1. Docker daemon check
echo -n "Checking Docker daemon... "
if ! docker info > /dev/null 2>&1; then
    echo "FAILED"
    echo "   Cause: Docker daemon not running"
    echo "   Resolution: sudo systemctl start docker"
    echo "   See: docs/setup/DOCKER_SETUP.md"
    EXIT_CODE=1
else
    echo "OK"
fi

# 2. Docker permissions check
echo -n "Checking Docker permissions... "
if ! docker ps > /dev/null 2>&1; then
    echo "FAILED"
    echo "   Cause: Current user lacks Docker permissions"
    echo "   Resolution: sudo usermod -aG docker $USER && newgrp docker"
    echo "   See: docs/setup/DOCKER_SETUP.md#user-permissions"
    EXIT_CODE=1
else
    echo "OK"
fi

# 3. Python version check
echo -n "Checking Python 3.13... "
if command -v python3.13 > /dev/null 2>&1; then
    PYTHON_VERSION=$(python3.13 --version | awk '{print $2}')
    echo "OK ($PYTHON_VERSION)"
elif command -v python3 > /dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 13 ]; then
        echo "OK ($PYTHON_VERSION via python3)"
    else
        echo "WARNING"
        echo "   Cause: Python 3.13+ not found (found $PYTHON_VERSION)"
        echo "   Resolution: Install Python 3.13+ or use uv to manage Python versions"
        echo "   See: https://docs.python.org/3/using/unix.html"
    fi
else
    echo "FAILED"
    echo "   Cause: Python 3.13 not found"
    echo "   Resolution: Install Python 3.13+"
    echo "   See: https://docs.python.org/3/using/unix.html"
    EXIT_CODE=1
fi

# 4. uv check
echo -n "Checking uv package manager... "
if command -v uv > /dev/null 2>&1; then
    UV_VERSION=$(uv --version | awk '{print $2}')
    echo "OK ($UV_VERSION)"
else
    echo "FAILED"
    echo "   Cause: uv not installed"
    echo "   Resolution: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   See: https://github.com/astral-sh/uv"
    EXIT_CODE=1
fi

# 5. Port availability checks
echo -n "Checking port availability... "
PORTS_IN_USE=""
for port in 6379 7001 8080 8081; do
    if command -v lsof > /dev/null 2>&1; then
        if lsof -i :$port > /dev/null 2>&1; then
            PORTS_IN_USE="$PORTS_IN_USE $port"
        fi
    elif command -v ss > /dev/null 2>&1; then
        if ss -tuln | grep -q ":$port "; then
            PORTS_IN_USE="$PORTS_IN_USE $port"
        fi
    fi
done

if [ -n "$PORTS_IN_USE" ]; then
    echo "WARNING"
    echo "   Cause: Ports in use:$PORTS_IN_USE"
    echo "   Resolution: Kill conflicting processes or change port config"
    echo "   See: docs/runbooks/PORT_CONFLICTS.md"
else
    echo "OK (6379, 7001, 8080, 8081)"
fi

# 6. Redis container check
echo -n "Checking Redis container... "
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "redis"; then
    echo "OK"
else
    echo "WARNING"
    echo "   Cause: Redis container not running"
    echo "   Resolution: just redis (or: docker run -d --name redis-tts -p 6379:6379 redis:7-alpine)"
    echo "   Note: Required for worker registration"
fi

# 7. Proto stubs check
echo -n "Checking gRPC proto stubs... "
if [ -d "src/rpc/generated" ] && [ -f "src/rpc/generated/tts_pb2.py" ]; then
    echo "OK"
else
    echo "WARNING"
    echo "   Cause: gRPC stubs not generated"
    echo "   Resolution: just gen-proto"
    echo "   Note: Required before running services"
fi

# 8. Dependencies check
echo -n "Checking Python dependencies... "
if [ -f "uv.lock" ]; then
    if [ -d ".venv" ]; then
        echo "OK"
    else
        echo "WARNING"
        echo "   Cause: Virtual environment not created"
        echo "   Resolution: uv sync"
    fi
else
    echo "WARNING"
    echo "   Cause: Dependencies not locked"
    echo "   Resolution: uv sync"
fi

# 9. NVIDIA runtime check (optional, for GPU workers)
echo -n "Checking NVIDIA GPU support... "
if command -v nvidia-smi > /dev/null 2>&1; then
    # Test with timeout to avoid hanging
    if timeout 5 docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "GPU")
        echo "OK ($GPU_INFO)"
    else
        echo "WARNING"
        echo "   Cause: NVIDIA container runtime not configured or timed out"
        echo "   Resolution: Install nvidia-container-toolkit"
        echo "   See: docs/setup/DOCKER_SETUP.md#nvidia-runtime"
    fi
else
    echo "SKIPPED (No NVIDIA drivers - CPU-only mode)"
fi

# 10. Justfile check
echo -n "Checking justfile... "
if command -v just > /dev/null 2>&1; then
    echo "OK"
else
    echo "WARNING"
    echo "   Cause: just command runner not installed"
    echo "   Resolution: cargo install just (or use package manager)"
    echo "   Note: Optional but recommended for convenience"
fi

echo ""
echo "=== Pre-Flight Check Complete ==="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: READY"
    echo "Next steps:"
    echo "  1. Start Redis: just redis (or docker run redis)"
    echo "  2. Generate protos: just gen-proto"
    echo "  3. Install dependencies: uv sync"
    echo "  4. Start services: docker compose up --build"
    echo ""
    echo "For detailed setup guide, see: docs/QUICK_START.md"
else
    echo "Status: SETUP REQUIRED"
    echo "Please resolve FAILED checks above before proceeding."
    echo ""
    echo "For troubleshooting help, see: docs/runbooks/ENVIRONMENT.md"
fi

exit $EXIT_CODE
