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

# Run integration tests with process isolation (requires Docker services)
test-integration:
    #!/usr/bin/env bash
    set -e
    echo "Note: Integration tests require Docker services to be running."
    echo "Start services with: docker compose up -d"
    echo ""
    echo "Running integration tests with --forked flag for process isolation..."
    GRPC_TESTS_FORKED=1 uv run pytest tests/integration/ --forked -v -m "integration"

# Run integration tests without process isolation (faster but may segfault)
test-integration-fast:
    #!/usr/bin/env bash
    set -e
    echo "Note: Integration tests require Docker services to be running."
    echo "Start services with: docker compose up -d"
    echo ""
    echo "Running integration tests without process isolation (may segfault)..."
    uv run pytest tests/integration/ -v -m "integration"

# Run performance benchmarks with process isolation (requires Docker services)
test-performance:
    #!/usr/bin/env bash
    set -e
    echo "Note: Performance tests require Docker services to be running."
    echo "Start services with: docker compose up -d"
    echo ""
    echo "Running performance tests with --forked flag for process isolation..."
    GRPC_TESTS_FORKED=1 uv run pytest tests/performance/test_performance.py --forked -v -m performance

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
        --default-model mock-440hz \
        --host localhost \
        --port 7001

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
    sudo py-spy top --pid {{PID}}

# CPU profiling with py-spy (record flamegraph)
spy-record PID OUT="profile.svg":
    sudo py-spy record -o {{OUT}} --pid {{PID}}

# GPU profiling with Nsight Systems
nsys-tts:
    nsys profile -o tts_profile uv run python -m src.tts.worker --adapter mock

# GPU profiling with Nsight Compute
ncu-tts:
    ncu --set full -o tts_kernels uv run python -m src.tts.worker --adapter mock
