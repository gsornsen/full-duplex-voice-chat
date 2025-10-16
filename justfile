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

# Development Environment
# -----------------------

# Install honcho process manager (if not already installed)
install-honcho:
    uv pip install honcho

# Start all development services in parallel (bare metal, no Docker build wait)
dev:
    #!/usr/bin/env bash
    set -e
    # Create logs directory if it doesn't exist
    mkdir -p logs/dev-sessions

    # Generate timestamp for log filename
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    LOGFILE="logs/dev-sessions/dev-${TIMESTAMP}.log"

    echo "Starting development services with honcho..."
    echo "Services: LiveKit, Caddy, TTS Worker, Orchestrator"
    echo "Logs: ${LOGFILE}"
    echo ""
    echo "Stop with: Ctrl+C (single press, waits up to 10s for graceful shutdown)"
    echo ""

    # Unset Docker Compose environment variables that override local configs
    unset REDIS_URL

    # Run honcho (Docker containers have --stop-timeout 10 for graceful shutdown)
    uv run honcho start -f Procfile.dev 2>&1 | tee "${LOGFILE}"

# Start all services with LiveKit Agent orchestrator (Phase 1 POC)
dev-agent:
    #!/usr/bin/env bash
    set -e
    # Create logs directory if it doesn't exist
    mkdir -p logs/dev-sessions

    # Generate timestamp for log filename
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    LOGFILE="logs/dev-sessions/dev-agent-${TIMESTAMP}.log"

    echo "Starting development services with LiveKit Agent orchestrator (Phase 1 POC)..."
    echo "Services: LiveKit Server, Caddy, TTS Worker, LiveKit Agent"
    echo "Logs: ${LOGFILE}"
    echo ""
    echo "NOTE: Using OpenAI STT/LLM/TTS (temporary - requires OPENAI_API_KEY in .env)"
    echo "      Phases 2-3 will replace with custom WhisperX STT and gRPC TTS"
    echo ""
    echo "Stop with: Ctrl+C (single press, waits up to 10s for graceful shutdown)"
    echo ""

    # Unset Docker Compose environment variables that override local configs
    unset REDIS_URL

    # Run honcho with agent Procfile
    uv run honcho start -f Procfile.agent 2>&1 | tee "${LOGFILE}"

# Start all services with LiveKit Agent + custom WhisperX STT + gRPC TTS (Piper)
dev-agent-piper:
    #!/usr/bin/env bash
    set -e
    # Create logs directory if it doesn't exist
    mkdir -p logs/dev-sessions

    # Generate timestamp for log filename
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    LOGFILE="logs/dev-sessions/dev-agent-piper-${TIMESTAMP}.log"

    echo "Starting development services with LiveKit Agent + Custom Plugins..."
    echo "Services: LiveKit Server, Caddy, TTS Worker (Piper), LiveKit Agent"
    echo "Custom Plugins: WhisperX STT (4-8x faster), gRPC TTS (Piper)"
    echo "Logs: ${LOGFILE}"
    echo ""
    echo "NOTE: Using custom WhisperX STT + gRPC TTS (Piper CPU) + OpenAI LLM"
    echo "      Phase 4 will make LLM optional for direct passthrough"
    echo ""
    echo "Stop with: Ctrl+C (single press, waits up to 10s for graceful shutdown)"
    echo ""

    # Unset Docker Compose environment variables that override local configs
    unset REDIS_URL

    # Run honcho with agent Procfile (already configured with Piper TTS worker)
    uv run honcho start -f Procfile.agent 2>&1 | tee "${LOGFILE}"

# Start all services with web client included
dev-web:
    #!/usr/bin/env bash
    set -e
    # Create logs directory if it doesn't exist
    mkdir -p logs/dev-sessions

    # Generate timestamp for log filename
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    LOGFILE="logs/dev-sessions/dev-web-${TIMESTAMP}.log"

    echo "Starting development services with web client..."
    echo "Services: Redis, LiveKit, TTS Worker, Orchestrator, Web"
    echo "Logs: ${LOGFILE}"
    echo ""
    echo "Press Ctrl+C to stop all services"
    echo ""

    # Create temporary Procfile with web service enabled
    cp Procfile.dev /tmp/Procfile.dev.tmp
    echo "web: cd src/client/web && npm run dev" >> /tmp/Procfile.dev.tmp

    # Run honcho with tee to output to both console and log file
    uv run honcho start -f /tmp/Procfile.dev.tmp 2>&1 | tee "${LOGFILE}"

dev-kill:
    #!/usr/bin/env bash
    set -e
    pkill -f "honcho start"


# Log Management
# --------------

# List recent development session logs (last 10)
logs-list:
    #!/usr/bin/env bash
    if [ ! -d "logs/dev-sessions" ]; then
        echo "No logs directory found. Run 'just dev' to create logs."
        exit 0
    fi
    echo "Recent development session logs:"
    echo "--------------------------------"
    ls -lt logs/dev-sessions/*.log 2>/dev/null | head -10 | awk '{print $9, "(" $6, $7, $8 ")"}'

# Tail the most recent development session log
logs-tail:
    #!/usr/bin/env bash
    if [ ! -d "logs/dev-sessions" ]; then
        echo "No logs directory found. Run 'just dev' to create logs."
        exit 1
    fi
    LATEST=$(ls -t logs/dev-sessions/*.log 2>/dev/null | head -1)
    if [ -z "$LATEST" ]; then
        echo "No log files found in logs/dev-sessions/"
        exit 1
    fi
    echo "Tailing: $LATEST"
    echo "Press Ctrl+C to stop"
    echo ""
    tail -f "$LATEST"

# View a specific log file
logs-view LOG:
    #!/usr/bin/env bash
    if [ ! -f "logs/dev-sessions/{{LOG}}" ]; then
        echo "Log file not found: logs/dev-sessions/{{LOG}}"
        echo ""
        echo "Available logs:"
        just logs-list
        exit 1
    fi
    less -R "logs/dev-sessions/{{LOG}}"

# Clean old development session logs (keep last 20 sessions OR 7 days, whichever is more)
logs-clean:
    #!/usr/bin/env bash
    if [ ! -d "logs/dev-sessions" ]; then
        echo "No logs directory found."
        exit 0
    fi

    echo "Cleaning old development session logs..."

    # Count total log files
    TOTAL=$(ls logs/dev-sessions/*.log 2>/dev/null | wc -l)
    if [ "$TOTAL" -eq 0 ]; then
        echo "No log files to clean."
        exit 0
    fi

    echo "Total log files: $TOTAL"

    # Keep last 20 files
    KEEP_COUNT=20
    if [ "$TOTAL" -gt "$KEEP_COUNT" ]; then
        echo "Keeping last $KEEP_COUNT files, removing $((TOTAL - KEEP_COUNT)) old files..."
        ls -t logs/dev-sessions/*.log | tail -n +$((KEEP_COUNT + 1)) | xargs rm -f
    fi

    # Also delete files older than 7 days
    DELETED=$(find logs/dev-sessions -name "*.log" -type f -mtime +7 -delete -print | wc -l)
    if [ "$DELETED" -gt 0 ]; then
        echo "Deleted $DELETED log files older than 7 days."
    fi

    # Show remaining files
    REMAINING=$(ls logs/dev-sessions/*.log 2>/dev/null | wc -l)
    echo "Remaining log files: $REMAINING"

# Infrastructure (Single Services)
# ---------------------------------

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

# Runtime (Single Services - for debugging)
# ------------------------------------------

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

# Run LiveKit Agent orchestrator (Phase 1 POC - requires LiveKit server and OpenAI API key)
run-agent:
    uv run python -m src.orchestrator.agent dev

# Run CLI client
cli HOST="ws://localhost:8080":
    uv run python -m src.client.cli_client --host {{HOST}}

# Build web client for production
web-build:
    cd src/client/web && npm run build

# Run web client (development mode)
web-dev:
    cd src/client/web && npm run dev

# Run web client (production mode)
web:
    cd src/client/web && npm start

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
