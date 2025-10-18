# Justfile for Realtime Duplex Voice Demo - Unified Workflow
# All Docker commands use Docker Compose as single source of truth
#
# SMART BUILD DETECTION:
# By default, Docker images are only built if they don't exist.
# This dramatically speeds up development workflow (2s vs 100s).
#
# Control build behavior with environment variables:
#   - Default:           Smart detection (build only if image missing)
#   - FORCE_BUILD=true:  Always rebuild images (for code changes)
#   - SKIP_BUILD=true:   Never rebuild (fail if image missing)
#
# Examples:
#   just dev                           # Fast startup with cached images
#   FORCE_BUILD=true just dev          # Force rebuild after code changes
#   DEFAULT_MODEL=cosyvoice2 just dev  # Switch model (uses cached images)

# =============================================================================
# Configuration
# =============================================================================

# Default model (can be overridden: DEFAULT_MODEL=cosyvoice just dev)
DEFAULT_MODEL := env_var_or_default("DEFAULT_MODEL", "piper")

# Default orchestrator mode (can be overridden: ORCHESTRATOR_MODE=legacy just dev)
DEFAULT_MODE := env_var_or_default("ORCHESTRATOR_MODE", "agent")

# Build control (can be overridden: FORCE_BUILD=true just dev, SKIP_BUILD=true just dev)
FORCE_BUILD := env_var_or_default("FORCE_BUILD", "false")
SKIP_BUILD := env_var_or_default("SKIP_BUILD", "false")

# Compose file path
COMPOSE_FILE := "docker-compose.yml"

# Compose command
COMPOSE := "docker compose -f " + COMPOSE_FILE

# Default recipe (list all recipes)
default:
    @just --list

# =============================================================================
# Development Workflow (New Unified Interface)
# =============================================================================

# Start development environment with specified model and orchestrator mode (background mode)
dev model=DEFAULT_MODEL mode=DEFAULT_MODE:
    #!/usr/bin/env bash
    set -e
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  Starting Unified Development Environment                          ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Model: {{model}}"
    echo "Orchestrator Mode: {{mode}}"
    echo "Infrastructure: Redis, LiveKit, Caddy"
    echo "Services: Orchestrator ({{mode}} mode), TTS Worker ({{model}})"
    echo ""

    # Validate orchestrator mode
    if [[ "{{mode}}" != "agent" && "{{mode}}" != "legacy" ]]; then
        echo "Error: Invalid orchestrator mode '{{mode}}'"
        echo "Valid modes: agent, legacy"
        exit 1
    fi

    # Export orchestrator mode for docker-compose
    export ORCHESTRATOR_MODE="{{mode}}"

    # Determine build flags (smart detection)
    BUILD_FLAG=""
    if [ "{{FORCE_BUILD}}" = "true" ]; then
        echo "🔨 Force build enabled (FORCE_BUILD=true)"
        BUILD_FLAG="--build"
    elif [ "{{SKIP_BUILD}}" = "true" ]; then
        echo "⚡ Skipping build (SKIP_BUILD=true)"
        BUILD_FLAG=""
    else
        # Smart detection: check if images exist
        ORCHESTRATOR_IMAGE="full-duplex-voice-chat-orchestrator"
        TTS_IMAGE="full-duplex-voice-chat-tts0"
        COSYVOICE_IMAGE="full-duplex-voice-chat-tts-cosyvoice"

        ORCHESTRATOR_EXISTS=$(docker images -q "$ORCHESTRATOR_IMAGE" 2>/dev/null)
        TTS_EXISTS=$(docker images -q "$TTS_IMAGE" 2>/dev/null)
        COSYVOICE_EXISTS=$(docker images -q "$COSYVOICE_IMAGE" 2>/dev/null)

        if [ -z "$ORCHESTRATOR_EXISTS" ]; then
            echo "🔨 Building orchestrator (image not found)"
            BUILD_FLAG="--build"
        elif [ "{{model}}" = "piper" ] && [ -z "$TTS_EXISTS" ]; then
            echo "🔨 Building TTS worker (image not found)"
            BUILD_FLAG="--build"
        elif [[ "{{model}}" = "cosyvoice"* ]] && [ -z "$COSYVOICE_EXISTS" ]; then
            echo "🔨 Building CosyVoice worker (image not found)"
            BUILD_FLAG="--build"
        else
            echo "⚡ Using cached images (set FORCE_BUILD=true to rebuild)"
            BUILD_FLAG=""
        fi
    fi

    # Determine which profile to use for TTS worker
    if [ "{{model}}" = "piper" ]; then
        echo "Starting with default Piper worker (CPU)..."
        {{COMPOSE}} up -d $BUILD_FLAG
    elif [ "{{model}}" = "cosyvoice" ] || [ "{{model}}" = "cosyvoice2" ]; then
        echo "Starting with CosyVoice worker (GPU, PyTorch 2.3.1)..."
        {{COMPOSE}} --profile cosyvoice up -d $BUILD_FLAG
    else
        echo "Error: Unknown model '{{model}}'"
        echo "Valid models: piper, cosyvoice, cosyvoice2"
        exit 1
    fi

    echo ""
    echo "✓ Services started in background"
    echo "  Orchestrator mode: {{mode}}"
    if [ "{{mode}}" = "agent" ]; then
        echo "    - Use with web frontend (LiveKit Agent)"
    else
        echo "    - Use with CLI client (WebSocket server)"
    fi
    echo "  View logs:   just dev-logs"
    echo "  View status: just dev-status"
    echo "  Stop all:    just dev-stop"

# Start development environment in foreground (with logs)
dev-fg model=DEFAULT_MODEL mode=DEFAULT_MODE:
    #!/usr/bin/env bash
    set -e
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  Starting Development Environment (Foreground)                     ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Model: {{model}}"
    echo "Orchestrator Mode: {{mode}}"
    echo "Press Ctrl+C to stop all services"
    echo ""

    # Validate orchestrator mode
    if [[ "{{mode}}" != "agent" && "{{mode}}" != "legacy" ]]; then
        echo "Error: Invalid orchestrator mode '{{mode}}'"
        echo "Valid modes: agent, legacy"
        exit 1
    fi

    # Export orchestrator mode for docker-compose
    export ORCHESTRATOR_MODE="{{mode}}"

    # Determine build flags (smart detection)
    BUILD_FLAG=""
    if [ "{{FORCE_BUILD}}" = "true" ]; then
        echo "🔨 Force build enabled (FORCE_BUILD=true)"
        BUILD_FLAG="--build"
    elif [ "{{SKIP_BUILD}}" = "true" ]; then
        echo "⚡ Skipping build (SKIP_BUILD=true)"
        BUILD_FLAG=""
    else
        # Smart detection: check if images exist
        ORCHESTRATOR_IMAGE="full-duplex-voice-chat-orchestrator"
        TTS_IMAGE="full-duplex-voice-chat-tts0"
        COSYVOICE_IMAGE="full-duplex-voice-chat-tts-cosyvoice"

        ORCHESTRATOR_EXISTS=$(docker images -q "$ORCHESTRATOR_IMAGE" 2>/dev/null)
        TTS_EXISTS=$(docker images -q "$TTS_IMAGE" 2>/dev/null)
        COSYVOICE_EXISTS=$(docker images -q "$COSYVOICE_IMAGE" 2>/dev/null)

        if [ -z "$ORCHESTRATOR_EXISTS" ]; then
            echo "🔨 Building orchestrator (image not found)"
            BUILD_FLAG="--build"
        elif [ "{{model}}" = "piper" ] && [ -z "$TTS_EXISTS" ]; then
            echo "🔨 Building TTS worker (image not found)"
            BUILD_FLAG="--build"
        elif [[ "{{model}}" = "cosyvoice"* ]] && [ -z "$COSYVOICE_EXISTS" ]; then
            echo "🔨 Building CosyVoice worker (image not found)"
            BUILD_FLAG="--build"
        else
            echo "⚡ Using cached images (set FORCE_BUILD=true to rebuild)"
            BUILD_FLAG=""
        fi
    fi

    # Determine which profile to use
    if [ "{{model}}" = "piper" ]; then
        echo "Starting with default Piper worker (CPU)..."
        {{COMPOSE}} up $BUILD_FLAG
    elif [ "{{model}}" = "cosyvoice" ] || [ "{{model}}" = "cosyvoice2" ]; then
        echo "Starting with CosyVoice worker (GPU, PyTorch 2.3.1)..."
        {{COMPOSE}} --profile cosyvoice up $BUILD_FLAG
    else
        echo "Error: Unknown model '{{model}}'"
        echo "Valid models: piper, cosyvoice, cosyvoice2"
        exit 1
    fi

# Switch to different model (hot swap)
dev-switch model:
    #!/usr/bin/env bash
    set -e
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  Hot-Swapping TTS Model                                            ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Target model: {{model}}"
    echo ""

    # Check if infrastructure is running
    if ! {{COMPOSE}} ps redis --format json 2>/dev/null | grep -q "running"; then
        echo "Error: Infrastructure not running. Use 'just dev {{model}}' instead."
        exit 1
    fi

    # Stop all worker containers (idempotent)
    echo "Stopping current workers..."
    {{COMPOSE}} stop tts0 || true
    {{COMPOSE}} stop tts-cosyvoice || true

    # Determine build flags (smart detection)
    BUILD_FLAG=""
    if [ "{{FORCE_BUILD}}" = "true" ]; then
        echo "🔨 Force build enabled (FORCE_BUILD=true)"
        BUILD_FLAG="--build"
    elif [ "{{SKIP_BUILD}}" = "true" ]; then
        echo "⚡ Skipping build (SKIP_BUILD=true)"
        BUILD_FLAG=""
    else
        # Smart detection: check if images exist
        TTS_IMAGE="full-duplex-voice-chat-tts0"
        COSYVOICE_IMAGE="full-duplex-voice-chat-tts-cosyvoice"

        TTS_EXISTS=$(docker images -q "$TTS_IMAGE" 2>/dev/null)
        COSYVOICE_EXISTS=$(docker images -q "$COSYVOICE_IMAGE" 2>/dev/null)

        if [ "{{model}}" = "piper" ] && [ -z "$TTS_EXISTS" ]; then
            echo "🔨 Building TTS worker (image not found)"
            BUILD_FLAG="--build"
        elif [[ "{{model}}" = "cosyvoice"* ]] && [ -z "$COSYVOICE_EXISTS" ]; then
            echo "🔨 Building CosyVoice worker (image not found)"
            BUILD_FLAG="--build"
        else
            echo "⚡ Using cached images (set FORCE_BUILD=true to rebuild)"
            BUILD_FLAG=""
        fi
    fi

    # Start new worker based on model
    if [ "{{model}}" = "piper" ]; then
        echo "Starting Piper worker..."
        {{COMPOSE}} up -d $BUILD_FLAG tts0
    elif [ "{{model}}" = "cosyvoice" ] || [ "{{model}}" = "cosyvoice2" ]; then
        echo "Starting CosyVoice worker..."
        {{COMPOSE}} --profile cosyvoice up -d $BUILD_FLAG tts-cosyvoice
    else
        echo "Error: Unknown model '{{model}}'"
        echo "Valid models: piper, cosyvoice, cosyvoice2"
        exit 1
    fi

    echo ""
    echo "✓ Model switched to {{model}}"
    echo "  Orchestrator will reconnect automatically via Redis service discovery"

# Stop all services
dev-stop:
    @echo "Stopping development environment..."
    {{COMPOSE}} --profile cosyvoice down

# Stop all services and remove volumes
dev-clean:
    @echo "Cleaning development environment (removing volumes)..."
    {{COMPOSE}} --profile cosyvoice down -v

# Full reset (down + clean + up)
dev-reset model=DEFAULT_MODEL mode=DEFAULT_MODE:
    @echo "Resetting development environment..."
    just dev-clean
    just dev {{model}} {{mode}}

# Follow logs for all services
dev-logs service="":
    #!/usr/bin/env bash
    if [ -z "{{service}}" ]; then
        echo "Following logs for all services (Ctrl+C to stop)..."
        {{COMPOSE}} logs -f
    else
        echo "Following logs for service: {{service}}"
        {{COMPOSE}} logs -f {{service}}
    fi

# Show status of running services
dev-status:
    @echo "Service Status:"
    @echo "═══════════════════════════════════════════════════════════════════"
    {{COMPOSE}} ps

# =============================================================================
# Model-Specific Shortcuts
# =============================================================================

# Start with Piper (CPU, no GPU) in LiveKit Agent mode
dev-piper:
    just dev piper agent

# Start with CosyVoice (GPU, PyTorch 2.3.1) in LiveKit Agent mode
dev-cosyvoice:
    just dev cosyvoice2 agent

# =============================================================================
# Orchestrator Mode Shortcuts
# =============================================================================

# Start in LiveKit Agent mode (for web frontend) - DEFAULT
dev-agent model=DEFAULT_MODEL:
    just dev {{model}} agent

# Start in Legacy WebSocket mode (for CLI client)
dev-legacy model=DEFAULT_MODEL:
    just dev {{model}} legacy

# =============================================================================
# Infrastructure Only
# =============================================================================

# Start only infrastructure (Redis, LiveKit, Caddy)
dev-infra:
    @echo "Starting infrastructure only..."
    {{COMPOSE}} up -d redis livekit caddy

# =============================================================================
# Image Management
# =============================================================================

# Rebuild specific worker image
dev-rebuild model:
    #!/usr/bin/env bash
    if [ "{{model}}" = "piper" ]; then
        {{COMPOSE}} build tts0
    elif [ "{{model}}" = "cosyvoice" ] || [ "{{model}}" = "cosyvoice2" ]; then
        {{COMPOSE}} build tts-cosyvoice
    else
        echo "Error: Unknown model '{{model}}'"
        exit 1
    fi

# Rebuild orchestrator image
dev-rebuild-orch:
    {{COMPOSE}} build orchestrator

# =============================================================================
# Deprecated Commands (Preserved for 1 Release)
# =============================================================================

# DEPRECATED: Start Redis container (use 'just dev-infra' instead)
redis:
    #!/usr/bin/env bash
    echo "⚠️  WARNING: 'just redis' is DEPRECATED"
    echo "   Use 'just dev-infra' to start all infrastructure services"
    echo "   This command will be removed in the next release."
    echo ""
    echo "Starting infrastructure services..."
    just dev-infra

# DEPRECATED: Stop and remove Redis container
redis-stop:
    #!/usr/bin/env bash
    echo "⚠️  WARNING: 'just redis-stop' is DEPRECATED"
    echo "   Use 'just dev-stop' or 'just dev-clean' instead"
    echo "   This command will be removed in the next release."
    echo ""
    {{COMPOSE}} stop redis

# DEPRECATED: Run orchestrator (use 'just dev' instead)
run-orch:
    #!/usr/bin/env bash
    echo "⚠️  WARNING: 'just run-orch' is DEPRECATED"
    echo "   Use 'just dev' to start the full development environment"
    echo "   This command will be removed in the next release."
    echo ""
    echo "ERROR: Orchestrator must be run as part of unified workflow"
    echo "       Run 'just dev' to start all services properly"
    exit 1

# =============================================================================
# Quality & CI
# =============================================================================

# Check CUDA/cuDNN compatibility before runtime
check-cuda:
    uv run python scripts/check_cuda_compatibility.py

# Test WhisperX GPU in Docker container (WSL2 compatibility test)
test-whisperx-gpu:
    #!/usr/bin/env bash
    set -e
    echo "Building WhisperX GPU test container..."
    docker build -f docker/Dockerfile.whisperx-gpu-test -t whisperx-gpu-test .
    echo ""
    echo "Running GPU benchmark in container..."
    docker run --rm --gpus all whisperx-gpu-test

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
    echo "Starting infrastructure..."
    just dev-infra
    echo ""
    echo "Waiting for services to be healthy..."
    sleep 10
    echo ""
    echo "Running integration tests with --forked flag for process isolation..."
    GRPC_TESTS_FORKED=1 uv run pytest tests/integration/ --forked -v -m "integration"

# Run integration tests without process isolation (faster but may segfault)
test-integration-fast:
    #!/usr/bin/env bash
    set -e
    echo "Note: Integration tests require Docker services to be running."
    echo "Start services with: just dev-infra"
    echo ""
    echo "Running integration tests without process isolation (may segfault)..."
    uv run pytest tests/integration/ -v -m "integration"

# Run performance benchmarks with process isolation (requires Docker services)
test-performance:
    #!/usr/bin/env bash
    set -e
    echo "Note: Performance tests require Docker services to be running."
    echo "Start services with: just dev-infra"
    echo ""
    echo "Running performance tests with --forked flag for process isolation..."
    GRPC_TESTS_FORKED=1 uv run pytest tests/performance/test_performance.py --forked -v -m performance

# Run all checks (lint + typecheck + test) - excludes Docker-dependent tests
ci: lint typecheck test

# =============================================================================
# Development Environment (Bare Metal - Legacy)
# =============================================================================

# Install honcho process manager (if not already installed)
install-honcho:
    uv pip install honcho

# DEPRECATED: Start all development services in parallel (bare metal, no Docker build wait)
dev-bare:
    #!/usr/bin/env bash
    set -e
    echo "⚠️  WARNING: Bare metal workflow is DEPRECATED"
    echo "   Use 'just dev' for unified Docker Compose workflow"
    echo "   This command may be removed in future releases."
    echo ""
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

# DEPRECATED: Start all services with LiveKit Agent orchestrator (Phase 1 POC)
dev-agent-honcho:
    #!/usr/bin/env bash
    set -e
    echo "⚠️  WARNING: 'just dev-agent-honcho' is DEPRECATED"
    echo "   Use 'just dev-agent' for unified Docker workflow"
    echo "   This command may be removed in future releases."
    echo ""
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

# DEPRECATED: Start all services with LiveKit Agent + custom WhisperX STT + gRPC TTS (Piper)
dev-agent-piper:
    #!/usr/bin/env bash
    set -e
    echo "⚠️  WARNING: 'just dev-agent-piper' is DEPRECATED"
    echo "   Use 'just dev-agent piper' for unified Docker workflow"
    echo "   This command may be removed in future releases."
    echo ""
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

# DEPRECATED: Start all services with web client included
dev-web:
    #!/usr/bin/env bash
    set -e
    echo "⚠️  WARNING: 'just dev-web' is DEPRECATED"
    echo "   Use 'just dev' and 'just web-dev' in separate terminals"
    echo "   This command may be removed in future releases."
    echo ""
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

# DEPRECATED: Kill honcho processes
dev-kill:
    #!/usr/bin/env bash
    set -e
    echo "⚠️  WARNING: 'just dev-kill' is DEPRECATED"
    echo "   Use 'just dev-stop' for graceful shutdown"
    echo "   This command will be removed in the next release."
    echo ""
    pkill -f "honcho start" || echo "No honcho processes found"

# =============================================================================
# Log Management (Legacy - for Procfile logs)
# =============================================================================

# List recent development session logs (last 10)
logs-list:
    #!/usr/bin/env bash
    if [ ! -d "logs/dev-sessions" ]; then
        echo "No logs directory found. Run 'just dev-bare' to create logs."
        exit 0
    fi
    echo "Recent development session logs:"
    echo "--------------------------------"
    ls -lt logs/dev-sessions/*.log 2>/dev/null | head -10 | awk '{print $9, "(" $6, $7, $8 ")"}'

# Tail the most recent development session log
logs-tail:
    #!/usr/bin/env bash
    if [ ! -d "logs/dev-sessions" ]; then
        echo "No logs directory found. Run 'just dev-bare' to create logs."
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

# =============================================================================
# Code Generation
# =============================================================================

# Generate gRPC stubs from proto files
gen-proto:
    uv run python -m grpc_tools.protoc \
        -I src/rpc \
        --python_out=src/rpc/generated \
        --grpc_python_out=src/rpc/generated \
        --pyi_out=src/rpc/generated \
        src/rpc/tts.proto && sed -i 's/^import tts_pb2 as tts__pb2/from . import tts_pb2 as tts__pb2/' src/rpc/generated/tts_pb2_grpc.py

# =============================================================================
# Runtime (Single Services - for debugging only)
# =============================================================================

# Run TTS worker with Sesame adapter (bare metal)
run-tts-sesame DEFAULT="cosyvoice2-en-base":
    @echo "⚠️  Note: This command runs the worker outside Docker"
    @echo "   For production use, prefer 'just dev sesame' (when available)"
    CUDA_VISIBLE_DEVICES=0 uv run python -m src.tts.worker \
        --config configs/worker.yaml \
        --adapter sesame \
        --default-model {{DEFAULT}}

# Run TTS worker with CosyVoice adapter (bare metal)
run-tts-cosy DEFAULT="cosyvoice2-en-base":
    @echo "⚠️  Note: This command runs the worker outside Docker"
    @echo "   For production use, prefer 'just dev cosyvoice'"
    CUDA_VISIBLE_DEVICES=0 uv run python -m src.tts.worker \
        --adapter cosyvoice2 \
        --default-model {{DEFAULT}}

# Run mock TTS worker (for M1/M2 testing, bare metal)
run-tts-mock:
    @echo "⚠️  Note: This command runs the worker outside Docker"
    uv run python -m src.tts.worker \
        --adapter mock \
        --default-model mock-440hz \
        --host localhost \
        --port 7001

# Run LiveKit Agent orchestrator (Phase 1 POC - requires LiveKit server and OpenAI API key)
run-agent:
    @echo "⚠️  Note: This command runs the agent outside Docker"
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

# =============================================================================
# Profiling
# =============================================================================

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
