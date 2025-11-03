#!/bin/bash
set -e

# This entrypoint runs as root to fix volume permissions
# It then switches to the cosyvoice user before executing the command

# Get the cosyvoice user ID (should be 1000)
COSYVOICE_UID=${COSYVOICE_UID:-1000}
COSYVOICE_GID=${COSYVOICE_GID:-1000}

# Fix cache directory permissions at runtime
# Volumes are often mounted with root ownership, so we fix them as root
# Create nested cache directory structures with proper permissions
# ModelScope creates directories like hub/owner/repo/ so we pre-create the full path
mkdir -p /home/cosyvoice/.cache/modelscope/hub/pengzhendong/wetext
mkdir -p /home/cosyvoice/.cache/modelscope/hub
mkdir -p /home/cosyvoice/.cache/huggingface/hub
mkdir -p /home/cosyvoice/.cache/torch
mkdir -p /home/cosyvoice/.triton/autotune

# Change ownership of cache directories to cosyvoice user
chown -R ${COSYVOICE_UID}:${COSYVOICE_GID} /home/cosyvoice/.cache 2>/dev/null || true
chown -R ${COSYVOICE_UID}:${COSYVOICE_GID} /home/cosyvoice/.triton 2>/dev/null || true

# Fix permissions for ModelScope cache (allows writing .msc files)
# ModelScope creates nested directories like hub/owner/repo/ so we need recursive permissions
chmod -R u+w /home/cosyvoice/.cache/modelscope 2>/dev/null || true
# Also fix ownership on any existing files
find /home/cosyvoice/.cache/modelscope -type d -exec chown ${COSYVOICE_UID}:${COSYVOICE_GID} {} + 2>/dev/null || true
find /home/cosyvoice/.cache/modelscope -type f -exec chown ${COSYVOICE_UID}:${COSYVOICE_GID} {} + 2>/dev/null || true

# Fix permissions for HuggingFace cache (allows writing refs/main files)
# HuggingFace creates directories like hub/models--Owner--Name/refs/main
chmod -R u+w /home/cosyvoice/.cache/huggingface 2>/dev/null || true
find /home/cosyvoice/.cache/huggingface -type d -exec chown ${COSYVOICE_UID}:${COSYVOICE_GID} {} + 2>/dev/null || true
find /home/cosyvoice/.cache/huggingface -type f -exec chown ${COSYVOICE_UID}:${COSYVOICE_GID} {} + 2>/dev/null || true

# Fix permissions for torch cache
chmod -R u+w /home/cosyvoice/.cache/torch 2>/dev/null || true
find /home/cosyvoice/.cache/torch -type d -exec chown ${COSYVOICE_UID}:${COSYVOICE_GID} {} + 2>/dev/null || true
find /home/cosyvoice/.cache/torch -type f -exec chown ${COSYVOICE_UID}:${COSYVOICE_GID} {} + 2>/dev/null || true

# Ensure voicepacks directory exists (CosyVoice will auto-download models here)
mkdir -p /app/voicepacks/cosyvoice
chown -R ${COSYVOICE_UID}:${COSYVOICE_GID} /app/voicepacks 2>/dev/null || true

# Switch to cosyvoice user before executing the command
# Use gosu for proper signal handling and user switching
if command -v gosu >/dev/null 2>&1; then
    exec gosu cosyvoice "$@"
else
    # Fallback to su if gosu is not available (shouldn't happen since we install gosu)
    exec su cosyvoice -s /bin/bash -c "exec \"\$@\"" -- "$@"
fi
