# Migration Guide: Unified Development Workflow

**Created**: 2025-10-17
**Target Audience**: Developers migrating from manual multi-terminal workflow to unified Honcho-based workflow

---

## Overview

This guide helps you migrate from the **old manual workflow** (multiple terminals, manual Docker commands) to the **new unified workflow** (single command, automatic logging, parallel startup).

### Why Migrate?

**Old Workflow Problems:**
- 4-5 separate terminals to manage
- 5+ minute Docker Compose build wait
- Manual log aggregation across terminals
- Difficult to correlate events by timestamp
- No automatic cleanup on exit
- Restart each service individually after code changes

**New Workflow Benefits:**
- Single command starts everything: `just dev-agent-piper`
- 10-second startup (vs 5+ minutes)
- Automatic timestamped log files
- Color-coded service output
- Graceful shutdown with one Ctrl+C
- Hot-reload friendly (fast restarts)

---

## Quick Migration

### TL;DR

**Before:**
```bash
# Terminal 1
just redis

# Terminal 2
just run-tts-piper

# Terminal 3
just run-orch

# Terminal 4
just cli
```

**After:**
```bash
# Single terminal
just dev-agent-piper

# CLI client in separate terminal (optional)
just cli
```

**That's it!** All services start in parallel with automatic logging.

---

## Step-by-Step Migration

### Step 1: Update Dependencies

```bash
# Ensure honcho is installed
uv sync --all-extras

# Verify honcho is available
uv run honcho --version
# Expected: honcho 1.1.0 or higher
```

**What honcho does:**
- Industry-standard process manager (Heroku Procfile format)
- Manages multiple processes with color-coded output
- Handles graceful shutdown (SIGINT/SIGTERM)
- Cross-platform (Linux, macOS, WSL2, Windows)

### Step 2: Understand New Commands

| Old Workflow | New Unified Workflow | Notes |
|--------------|----------------------|-------|
| `just redis` (manual) | `just dev-agent-piper` | Redis not needed, uses LiveKit's built-in state |
| `just run-tts-piper` | (automatic) | Started by Honcho |
| `just run-orch` | (automatic) | Started by Honcho (agent mode) |
| `just cli` | `just cli` (unchanged) | Still separate for testing |
| `docker compose up --build` | `docker compose up` (unchanged) | Still works for production testing |

**New commands added:**
- `just dev-agent-piper` - Start all services with LiveKit Agent + Piper TTS
- `just dev` - Start all services with legacy orchestrator (for comparison)
- `just dev-web` - Include Next.js web client
- `just logs-list` - List recent development session logs
- `just logs-tail` - Tail most recent log file
- `just logs-clean` - Clean old logs

### Step 3: First Run

```bash
# Start unified development mode
just dev-agent-piper

# You'll see:
# Starting development services with LiveKit Agent + Custom Plugins...
# Services: LiveKit Server, Caddy, TTS Worker (Piper), LiveKit Agent
# Custom Plugins: WhisperX STT (4-8x faster), gRPC TTS (Piper)
# Logs: logs/dev-sessions/dev-agent-piper-20251017-143022.log
#
# [livekit]     | LiveKit server started on :7880
# [caddy]       | Caddy HTTPS proxy listening on :8443, :8444
# [tts]         | TTS worker (Piper) ready on :7001
# [agent]       | LiveKit agent ready, waiting for participants
```

**Services started:**
1. LiveKit Server (Docker container, ports 7880-7882)
2. Caddy HTTPS Proxy (Docker container, ports 8443/8444)
3. TTS Worker (Python process, port 7001, Piper adapter)
4. LiveKit Agent (Python process, custom STT/TTS/LLM)

**Access points:**
- Web client: https://localhost:8443
- LiveKit WebSocket: wss://localhost:8444
- Logs: `logs/dev-sessions/dev-agent-piper-YYYYMMDD-HHMMSS.log`

### Step 4: Verify Everything Works

```bash
# In a separate terminal, test with CLI client
just cli

# You should connect successfully and be able to speak/type
```

**What to check:**
- All 4 services start without errors
- Log file created in `logs/dev-sessions/`
- CLI client connects successfully
- No port conflict errors

### Step 5: Stop Services

```bash
# In the terminal running `just dev-agent-piper`:
# Press Ctrl+C (once)

# You'll see:
# Shutting down gracefully...
# Stopping livekit (SIGINT)
# Stopping caddy (SIGINT)
# Stopping tts (SIGINT)
# Stopping agent (SIGINT)
# All processes stopped
```

**Graceful shutdown:**
- Honcho sends SIGINT to all processes
- Docker containers have `--stop-timeout 10` (10 second grace period)
- Python processes handle SIGINT for cleanup
- All processes exit within 10 seconds

### Step 6: Review Logs

```bash
# List recent development sessions
just logs-list

# Output:
# Recent development session logs:
# --------------------------------
# logs/dev-sessions/dev-agent-piper-20251017-143022.log (Oct 17 14:30)
# logs/dev-sessions/dev-20251017-101522.log (Oct 17 10:15)
# logs/dev-sessions/dev-agent-piper-20251017-093045.log (Oct 17 09:30)

# View specific log
just logs-view dev-agent-piper-20251017-143022.log

# Or tail the most recent log (follow in real-time)
just logs-tail
```

---

## Command Mapping Reference

### Quality & CI (Unchanged)

| Command | Purpose |
|---------|---------|
| `just lint` | Run ruff linting |
| `just fix` | Auto-fix linting issues |
| `just typecheck` | Run mypy type checking |
| `just test` | Run pytest tests |
| `just ci` | Run all checks (lint + typecheck + test) |

### Development Workflow (Updated)

| Old Command | New Command | Notes |
|-------------|-------------|-------|
| **Multiple terminals** | `just dev-agent-piper` | **Recommended: One command, all services** |
| `just redis` | (not needed) | LiveKit handles state internally |
| `just run-tts-piper` | (automatic in dev mode) | For debugging: still available |
| `just run-orch` | (automatic in dev mode) | For debugging: still available |
| `just cli` | `just cli` | Unchanged, still separate for testing |
| N/A | `just dev` | Alternative: legacy orchestrator |
| N/A | `just dev-web` | Alternative: include web client |

### Individual Services (For Debugging)

These commands still work if you need fine-grained control:

```bash
# Start Redis (if needed separately)
just redis

# Run TTS worker only
just run-tts-piper DEFAULT="piper-en-us-lessac-medium"

# Run orchestrator only (legacy mode)
just run-orch

# Run LiveKit Agent only (new mode)
just run-agent
```

**When to use:**
- Debugging specific service in isolation
- Profiling single service (py-spy, nsys)
- Need to restart single service frequently
- Want separate log files per service

### Log Management (New)

| Command | Purpose |
|---------|---------|
| `just logs-list` | List last 10 development session logs |
| `just logs-tail` | Tail most recent log file (follow) |
| `just logs-view <LOG>` | View specific log file with pager |
| `just logs-clean` | Clean old logs (keep last 20 or 7 days) |

### Docker Compose (Enhanced)

| Command | Purpose | Notes |
|---------|---------|-------|
| `docker compose up` | Start all services (default profile) | Piper TTS worker |
| `docker compose --profile cosyvoice up` | Start with CosyVoice GPU worker | Isolated PyTorch 2.3.1 |
| `docker compose up redis livekit caddy` | Start specific services only | Useful for hybrid mode |

---

## Configuration Changes

### New Files

**Procfile.dev** - Unified development workflow definition:
```bash
# Procfile.dev (Heroku-compatible format)
livekit: docker run --rm --name livekit-dev-honcho ...
caddy: docker run --rm --name caddy-dev-honcho ...
tts: uv run python -u -m src.tts --adapter piper ...
orchestrator: uv run python -u -m src.orchestrator.server ...
```

**Procfile.agent** - LiveKit Agent workflow:
```bash
# Procfile.agent (uses LiveKit Agent instead of old orchestrator)
livekit: docker run --rm --name livekit-dev-honcho ...
caddy: docker run --rm --name caddy-dev-honcho ...
tts: uv run python -u -m src.tts --adapter piper ...
agent: uv run python -u -m src.orchestrator.agent dev
```

**Justfile** - Updated with new commands:
- `just dev-agent-piper` - Runs `honcho start -f Procfile.agent`
- `just dev` - Runs `honcho start -f Procfile.dev`
- `just dev-web` - Adds Next.js web client to Procfile.dev
- Log management commands added

### Modified Files

**docker-compose.yml** - Added Docker Compose profiles:
```yaml
services:
  tts-cosyvoice:
    profiles:
      - cosyvoice  # Only start with: docker compose --profile cosyvoice up
```

**No Changes Required:**
- Config files (`configs/*.yaml`) - Unchanged
- Environment variables (`.env`) - Unchanged
- Python source code - Unchanged
- Test files - Unchanged

---

## Environment Variables

### Old Workflow

```bash
# .env (before migration)
REDIS_URL=redis://localhost:6379
TTS_WORKER_URL=grpc://localhost:7001
```

### New Workflow

```bash
# .env (after migration - SAME FILE, just ensure these are set)
REDIS_URL=redis://localhost:6379
TTS_WORKER_URL=grpc://localhost:7001

# Additional for LiveKit Agent (new)
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=devsecret1234567890abcdefghijklmn

# Temporary: OpenAI for LLM (will be optional in future)
OPENAI_API_KEY=sk-proj-...
```

**No breaking changes** - Old environment variables still work.

---

## Troubleshooting Migration

### Issue: `honcho: command not found`

**Solution:**
```bash
# Install honcho
uv sync --all-extras

# Verify installation
uv run honcho --version
```

### Issue: Port Already in Use

**Symptom:**
```
Error: bind: address already in use (port 7880)
```

**Solution:**
```bash
# Find and kill process using port
lsof -i :7880
kill -9 <PID>

# Or stop all Docker containers
docker stop $(docker ps -q)

# Retry
just dev-agent-piper
```

### Issue: Docker Container Name Conflict

**Symptom:**
```
Error: container name livekit-dev-honcho already in use
```

**Solution:**
```bash
# Remove existing containers
docker stop livekit-dev-honcho caddy-dev-honcho
docker rm livekit-dev-honcho caddy-dev-honcho

# Retry
just dev-agent-piper
```

### Issue: Services Start But Don't Work

**Symptom:**
- Services start without errors
- But connections fail or timeouts occur

**Diagnostics:**
```bash
# Check service health
curl http://localhost:7880/  # LiveKit
curl https://localhost:8443/ # Caddy (HTTPS)

# Check logs for errors
just logs-tail

# Verify environment variables
env | grep LIVEKIT
env | grep OPENAI
```

**Common fixes:**
```bash
# Missing environment variable
echo "OPENAI_API_KEY=sk-proj-..." >> .env

# Missing TLS certificates
# Follow setup instructions in docs/DEVELOPMENT.md
```

### Issue: Logs Not Being Created

**Symptom:**
- No `logs/dev-sessions/` directory
- Empty log files

**Solution:**
```bash
# Create directory manually
mkdir -p logs/dev-sessions

# Check write permissions
ls -ld logs/dev-sessions

# Verify tee command exists
which tee

# Check disk space
df -h
```

---

## What to Delete (Cleanup)

### Safe to Delete

**Old shell scripts (if you created custom wrappers):**
```bash
# Example old scripts (delete if they exist)
rm scripts/start-dev-services.sh
rm scripts/stop-dev-services.sh
rm scripts/start-all.sh
```

**Old log files (manual cleanup):**
```bash
# Clean old logs (automated)
just logs-clean

# Or manual cleanup
rm logs/dev-sessions/*.log  # (keep recent ones!)
```

### DO NOT Delete

**Keep these files:**
- `justfile` - Updated with new commands
- `docker-compose.yml` - Updated with profiles
- `Procfile.dev` - New unified workflow definition
- `Procfile.agent` - New LiveKit Agent workflow
- All `configs/*.yaml` - Still used by services
- `.env` - Still used for environment variables
- All Python source code (`src/`)
- All test files (`tests/`)

---

## Rollback Procedure

If you need to revert to the old workflow:

### Step 1: Stop New Workflow

```bash
# If running, stop with Ctrl+C
# Or force kill
just dev-kill
```

### Step 2: Use Individual Services

```bash
# Old workflow still works!
# Terminal 1
just redis

# Terminal 2
just run-tts-piper

# Terminal 3
just run-orch

# Terminal 4
just cli
```

### Step 3: Report Issues

If rollback was necessary, please report:
- OS and version
- Python version (`python --version`)
- Error messages from logs (`just logs-tail`)
- Output of `just dev-agent-piper`
- Environment variables (`env | grep LIVEKIT`)

**File bug report with:**
```bash
# Gather diagnostics
uv run honcho --version
docker --version
docker ps
lsof -i :7880
env | grep -E "(LIVEKIT|OPENAI|REDIS)"
cat logs/dev-sessions/dev-agent-piper-*.log | tail -100
```

---

## Advanced: Customizing the Workflow

### Edit Procfile for Different TTS Models

```bash
# Edit Procfile.agent
vim Procfile.agent

# Change TTS adapter line:
# Before:
tts: uv run python -u -m src.tts --adapter piper --default-model piper-en-us-lessac-medium

# After (example: different voice):
tts: uv run python -u -m src.tts --adapter piper --default-model piper-en-us-amy-low

# Save and restart
just dev-agent-piper
```

### Add Custom Services to Procfile

```bash
# Edit Procfile.dev or Procfile.agent
vim Procfile.agent

# Add new service at the end:
myservice: uv run python -u -m my_module --port 9000

# Restart
just dev-agent-piper
```

### Disable Specific Services

```bash
# Edit Procfile temporarily
vim Procfile.agent

# Comment out service:
# caddy: docker run --rm ...  # Disabled for debugging

# Restart
just dev-agent-piper
```

---

## Summary

### Migration Checklist

- [x] Install honcho: `uv sync --all-extras`
- [x] Understand new commands: `just dev-agent-piper`
- [x] First run: `just dev-agent-piper`
- [x] Verify services start correctly
- [x] Test with CLI client: `just cli`
- [x] Review logs: `just logs-list`
- [x] Clean up old logs: `just logs-clean`
- [x] (Optional) Delete old custom scripts
- [x] Bookmark this guide for troubleshooting

### Quick Reference

**Most Common Commands:**
```bash
# Start everything (recommended)
just dev-agent-piper

# Stop everything
Ctrl+C

# View recent logs
just logs-tail

# Clean old logs
just logs-clean

# Individual debugging (if needed)
just redis
just run-tts-piper
just run-orch
```

### Next Steps

1. **Try the new workflow** - `just dev-agent-piper`
2. **Read the detailed guide** - [docs/DOCKER_UNIFIED_WORKFLOW.md](docs/DOCKER_UNIFIED_WORKFLOW.md)
3. **Learn log management** - `just logs-list`, `just logs-tail`
4. **Customize Procfile** - Edit for different models
5. **Report issues** - File bugs with logs and diagnostics

---

**Need Help?**
- Comprehensive guide: [docs/DOCKER_UNIFIED_WORKFLOW.md](docs/DOCKER_UNIFIED_WORKFLOW.md)
- Development guide: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- Known issues: [docs/known-issues/README.md](docs/known-issues/README.md)
- File bugs with: `just logs-tail` output

**Happy developing with the unified workflow!**
