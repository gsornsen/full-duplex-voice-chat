#!/bin/bash
# Orchestrator entrypoint script - selects mode based on ORCHESTRATOR_MODE env var
set -e

# Ensure HuggingFace cache directories exist with proper permissions
# This fixes permission errors when writing commit hashes to model refs
# HuggingFace creates nested directories like hub/models--Owner--Name/refs/main
mkdir -p /home/orchestrator/.cache/huggingface/hub
# Fix permissions recursively on existing files/directories
find /home/orchestrator/.cache/huggingface -type d -exec chmod u+w {} + 2>/dev/null || true
find /home/orchestrator/.cache/huggingface -type f -exec chmod u+w {} + 2>/dev/null || true
chmod -R u+w /home/orchestrator/.cache/huggingface 2>/dev/null || true

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
        echo "Entry point: orchestrator.agent"
        echo ""
        exec python -m orchestrator.agent dev
        ;;
    legacy)
        echo "Starting Legacy WebSocket Server mode (for CLI client)..."
        echo "Entry point: orchestrator.server"
        echo ""
        exec python -m orchestrator.server --config configs/orchestrator.docker.yaml
        ;;
    *)
        echo "ERROR: Invalid ORCHESTRATOR_MODE='${MODE}'"
        echo "Valid modes: agent, legacy"
        exit 1
        ;;
esac
