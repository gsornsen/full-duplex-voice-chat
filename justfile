# Justfile for Realtime Duplex Voice Demo

# Default recipe (list all recipes)
default:
    @just --list

# Quality & CI
# ------------

# Run ruff linting
lint:
    uv run ruff check src/ tests/

# Auto-fix linting issues
fix:
    uv run ruff check --fix src/ tests/

# Run mypy type checking
typecheck:
    uv run mypy src/ tests/

# Run pytest tests (excludes performance benchmarks and integration tests requiring Docker)
test:
    uv run pytest tests/ -v -m "not performance and not docker"

# Run integration tests (requires Docker services to be running)
test-integration:
    #!/usr/bin/env bash
    set -e
    echo "Note: Integration tests require Docker services to be running."
    echo "Start services with: docker compose up -d"
    echo ""
    uv run pytest tests/integration/ -v -m "integration"

# Run performance benchmarks with process isolation (requires Docker services)
test-performance:
    #!/usr/bin/env bash
    set -e
    echo "Note: Performance tests require Docker services to be running."
    echo "Start services with: docker compose up -d"
    echo ""
    echo "Running performance tests with --forked flag for process isolation..."
    uv run pytest tests/performance/test_performance.py --forked -v -m performance

# Run all checks (lint + typecheck + test) - excludes Docker-dependent tests
ci: lint typecheck test

# Infrastructure
# --------------

# Start Redis container
redis:
    docker run -d --name redis-tts -p 6379:6379 redis:7-alpine

# Stop and remove Redis container
redis-stop:
    docker stop redis-tts && docker rm redis-tts

# Code Generation
# ---------------

# Generate gRPC stubs from proto files
gen-proto:
    uv run python -m grpc_tools.protoc \
        -I src/rpc \
        --python_out=src/rpc/generated \
        --grpc_python_out=src/rpc/generated \
        --pyi_out=src/rpc/generated \
        src/rpc/tts.proto && sed -i 's/^import tts_pb2 as tts__pb2/from . import tts_pb2 as tts__pb2/' src/rpc/generated/tts_pb2_grpc.py

# Runtime (Single-GPU)
# --------------------

# Run TTS worker with Sesame adapter
run-tts-sesame DEFAULT="cosyvoice2-en-base":
    CUDA_VISIBLE_DEVICES=0 uv run python -m src.tts.worker \
        --config configs/worker.yaml \
        --adapter sesame \
        --default-model {{DEFAULT}}

# Run TTS worker with CosyVoice adapter
run-tts-cosy DEFAULT="cosyvoice2-en-base":
    CUDA_VISIBLE_DEVICES=0 uv run python -m src.tts.worker \
        --adapter cosyvoice2 \
        --default-model {{DEFAULT}}

# Run mock TTS worker (for M1/M2 testing)
run-tts-mock:
    uv run python -m src.tts.worker \
        --adapter mock \
        --default-model mock-440hz

# Run orchestrator
run-orch:
    uv run python -m src.orchestrator.server

# Run CLI client
cli HOST="ws://localhost:8080":
    uv run python -m src.client.cli_client --host {{HOST}}

# Profiling
# ---------

# CPU profiling with py-spy (top view)
spy-top PID:
    py-spy top --pid {{PID}}

# CPU profiling with py-spy (record flamegraph)
spy-record PID OUT="profile.svg":
    py-spy record --pid {{PID}} --output {{OUT}}

# GPU profiling with Nsight Systems
nsys-tts:
    nsys profile -o tts-trace \
        --trace=cuda,nvtx,osrt \
        uv run python -m src.tts.worker --config configs/worker.yaml

# GPU profiling with Nsight Compute
ncu-tts:
    ncu --set full \
        --export tts-kernels \
        uv run python -m src.tts.worker --config configs/worker.yaml

# Docker
# ------

# Build all Docker images
docker-build:
    docker compose build

# Start full stack (redis + orchestrator + tts workers)
docker-up:
    docker compose up --build

# Stop all services
docker-down:
    docker compose down

# View all logs
docker-logs:
    docker compose logs -f

# View specific service logs
docker-logs-service SERVICE:
    docker compose logs -f {{SERVICE}}

# HTTPS Setup for Local Network Access
# -------------------------------------

# Update Caddyfile and .env.local with current host IP
update-ip IP="":
    #!/usr/bin/env bash
    if [[ -z "{{IP}}" ]]; then
        ./scripts/update-ip.sh
    else
        ./scripts/update-ip.sh --ip {{IP}}
    fi

# Show current IP address
show-ip:
    @hostname -I | awk '{print $1}'

# Restart Caddy to apply configuration changes
caddy-restart:
    docker compose restart caddy

# View Caddy logs
caddy-logs:
    docker compose logs -f caddy

# Test HTTPS setup (requires services to be running)
test-https:
    #!/usr/bin/env bash
    IP=$(hostname -I | awk '{print $1}')
    echo "Testing HTTPS endpoints..."
    echo ""
    echo "1. Testing web client (HTTPS):"
    curl -k -I "https://$IP" || echo "Failed to connect to web client"
    echo ""
    echo "2. Testing LiveKit (WSS):"
    curl -k -I "https://$IP:7443" || echo "Failed to connect to LiveKit"
    echo ""
    echo "Access URLs:"
    echo "  Web Client: https://$IP"
    echo "  LiveKit WSS: wss://$IP:7443"

# Development
# -----------

# Install dependencies
install:
    uv sync --all-extras

# Lock dependencies
lock:
    uv lock

# Clean generated files
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    rm -rf src/rpc/generated/*.py src/rpc/generated/*.pyi 2>/dev/null || true
    rm -rf .pytest_cache .mypy_cache .ruff_cache 2>/dev/null || true

# Format code
format:
    uv run ruff format src/ tests/
