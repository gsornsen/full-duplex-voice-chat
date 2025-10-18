#!/bin/bash
# Orchestrator entrypoint script - selects mode based on ORCHESTRATOR_MODE env var
set -e

MODE="${ORCHESTRATOR_MODE:-agent}"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Orchestrator Entrypoint                                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Mode: ${MODE}"
echo ""

case "${MODE}" in
    agent)
        echo "Starting LiveKit Agent mode (for web frontend)..."
        echo "Entry point: src.orchestrator.agent"
        echo ""
        exec uv run python -m src.orchestrator.agent dev
        ;;
    legacy)
        echo "Starting Legacy WebSocket Server mode (for CLI client)..."
        echo "Entry point: src.orchestrator.server"
        echo ""
        exec uv run python -m src.orchestrator.server --config configs/orchestrator.docker.yaml
        ;;
    *)
        echo "ERROR: Invalid ORCHESTRATOR_MODE='${MODE}'"
        echo "Valid modes: agent, legacy"
        exit 1
        ;;
esac
