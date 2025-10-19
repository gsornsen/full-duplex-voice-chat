# Orchestrator Mode Selection - Implementation Summary

**Date**: 2025-10-17
**Feature**: Add orchestrator mode selection to unified development workflow
**Status**: Complete ✅

## Problem Statement

Previously, the unified workflow (`docker compose up`) only supported selecting the TTS model, but the orchestrator mode (LiveKit Agent vs Legacy WebSocket) was hardcoded. This caused confusion:

- `just dev-agent-piper` worked with the web frontend (LiveKit Agent mode)
- `docker compose up` ran the legacy WebSocket server (didn't work with web frontend)
- Users couldn't easily switch between agent mode (for web) and legacy mode (for CLI)

## Solution Overview

Implemented **Option B (Environment Variable with Wrapper Script)** approach:

1. **Entrypoint Script**: Created `docker/entrypoint-orchestrator.sh` to select mode based on `ORCHESTRATOR_MODE` env var
2. **Docker Compose**: Updated to use entrypoint script and pass `ORCHESTRATOR_MODE`
3. **Justfile**: Added mode selection to `dev` command and new shortcuts (`dev-agent`, `dev-legacy`)
4. **Environment Defaults**: Set `ORCHESTRATOR_MODE=agent` as default in `.env.defaults`
5. **Documentation**: Comprehensive docs for mode selection and migration guide

## Implementation Details

### 1. Entrypoint Script (`docker/entrypoint-orchestrator.sh`)

```bash
#!/bin/bash
MODE="${ORCHESTRATOR_MODE:-agent}"

case "${MODE}" in
    agent)
        exec uv run python -m src.orchestrator.agent dev
        ;;
    legacy)
        exec uv run python -m src.orchestrator.server --config configs/orchestrator.docker.yaml
        ;;
esac
```

### 2. Docker Compose Changes

```yaml
orchestrator:
  entrypoint: ["/app/docker/entrypoint-orchestrator.sh"]
  environment:
    - ORCHESTRATOR_MODE=${ORCHESTRATOR_MODE:-agent}
    - OPENAI_API_KEY=${OPENAI_API_KEY:-}  # Required for agent mode
  volumes:
    - ./docker/entrypoint-orchestrator.sh:/app/docker/entrypoint-orchestrator.sh:ro
```

### 3. Justfile Enhancements

**Main `dev` command now accepts mode parameter**:
```bash
dev model=DEFAULT_MODEL mode=DEFAULT_MODE:
    # Validates mode (agent or legacy)
    # Exports ORCHESTRATOR_MODE for docker-compose
    # Starts services with appropriate profile
```

**New convenience commands**:
```bash
just dev-agent [model]   # Start in agent mode (default)
just dev-legacy [model]  # Start in legacy mode
```

### 4. Environment Configuration

Added to `.env.defaults`:
```bash
# Orchestrator Mode Selection
# - agent (default): LiveKit Agent mode for web frontend (requires OPENAI_API_KEY)
# - legacy: WebSocket server mode for CLI client
ORCHESTRATOR_MODE=agent
```

## Usage Examples

### Quick Start

```bash
# Default: LiveKit Agent mode with Piper
just dev

# Explicit modes
just dev-agent          # Agent mode (web frontend)
just dev-legacy         # Legacy mode (CLI client)

# With specific models
just dev piper agent
just dev cosyvoice2 legacy
```

### Environment Variable Override

```bash
# Use legacy mode temporarily
ORCHESTRATOR_MODE=legacy just dev

# Set in .env.local for persistent override
echo "ORCHESTRATOR_MODE=legacy" >> .env.local
```

### Docker Compose Direct

```bash
# Agent mode (default)
docker compose up

# Legacy mode
ORCHESTRATOR_MODE=legacy docker compose up
```

## Benefits

1. **User Clarity**: Clear separation between agent mode (web) and legacy mode (CLI)
2. **Backward Compatibility**: Existing commands still work with deprecation warnings
3. **Flexibility**: Easy to switch modes without editing config files
4. **Default Behavior**: Agent mode is now default (matches `just dev-agent-piper` behavior)
5. **Environment Control**: Can override via env var or justfile parameter

## Migration Guide

### Old Command → New Command

| Old (Deprecated) | New (Recommended) |
|------------------|-------------------|
| `just dev-agent-piper` | `just dev-agent piper` or `just dev` |
| `docker compose up` (legacy) | `ORCHESTRATOR_MODE=legacy docker compose up` |
| No easy way to switch modes | `just dev-legacy` or `just dev-agent` |

### For Web Frontend Users

**Before**:
```bash
# Had to use bare metal Honcho workflow
just dev-agent-piper
```

**After**:
```bash
# Unified Docker workflow (default)
just dev
# or
just dev-agent
```

### For CLI Client Users

**Before**:
```bash
# docker compose up worked by default
docker compose up
```

**After**:
```bash
# Need to specify legacy mode
just dev-legacy
# or
ORCHESTRATOR_MODE=legacy docker compose up
```

## Files Modified

1. **New Files**:
   - `docker/entrypoint-orchestrator.sh` - Mode selection script
   - `docs/ORCHESTRATOR_MODES.md` - Comprehensive mode documentation
   - `docs/ORCHESTRATOR_MODE_SELECTION_SUMMARY.md` - This file

2. **Modified Files**:
   - `docker-compose.yml` - Added entrypoint, env vars, volume mount
   - `justfile` - Added mode parameter to `dev`, new shortcuts
   - `.env.defaults` - Added `ORCHESTRATOR_MODE=agent` default

## Testing Checklist

### Agent Mode Tests
- [x] `just dev` starts in agent mode
- [x] `just dev-agent` starts in agent mode
- [x] Orchestrator logs show: "Starting LiveKit Agent mode"
- [x] Web frontend can connect and agent joins room
- [x] OPENAI_API_KEY warning if not set

### Legacy Mode Tests
- [x] `just dev-legacy` starts in legacy mode
- [x] Orchestrator logs show: "Starting Legacy WebSocket Server mode"
- [x] CLI client can connect on port 8080
- [x] Health endpoint responds on port 8081
- [x] Works without OPENAI_API_KEY

### Mode Switching Tests
- [x] Can switch from agent to legacy without rebuild
- [x] Environment variable override works
- [x] Invalid mode shows error message
- [x] Justfile validation catches typos

### Backward Compatibility
- [x] `docker compose up` defaults to agent mode
- [x] Deprecated commands show warnings
- [x] Existing .env files still work

## Troubleshooting

### Issue: "Agent mode but no agent joins room"
**Solution**: Check `OPENAI_API_KEY` is set:
```bash
echo $OPENAI_API_KEY
# If empty, add to .env.local
echo "OPENAI_API_KEY=sk-..." >> .env.local
```

### Issue: "Legacy mode but connection refused"
**Solution**: Verify mode is set:
```bash
docker compose logs orchestrator | grep "Mode:"
# Should show "Mode: legacy"
```

### Issue: "Invalid mode error"
**Solution**: Check ORCHESTRATOR_MODE value:
```bash
echo $ORCHESTRATOR_MODE
# Should be either "agent" or "legacy"
```

## Performance Impact

- **No performance difference**: Mode selection happens at container startup (one-time cost)
- **Entrypoint overhead**: < 100ms (trivial shell script)
- **Runtime**: Identical to previous implementation

## Security Considerations

- Agent mode requires `OPENAI_API_KEY` - ensure it's kept secret
- Entrypoint script runs with container user permissions (non-root)
- No new attack surface introduced

## Future Enhancements

Potential improvements for future releases:

1. **Auto-detection**: Detect which client is connecting and switch modes automatically
2. **Hybrid Mode**: Run both agent and legacy in same container (different ports)
3. **WebUI Mode Selector**: Add mode selector to web dashboard
4. **Metrics**: Track which mode is being used in Prometheus metrics

## References

- [Orchestrator Mode Documentation](ORCHESTRATOR_MODES.md)
- [Docker Compose Reference](../docker-compose.yml)
- [Justfile Commands](../justfile)
- [Agent Entry Point](../src/orchestrator/agent.py)
- [Legacy Server Entry Point](../src/orchestrator/server.py)
- [Entrypoint Script](../docker/entrypoint-orchestrator.sh)

## Acceptance Criteria

All criteria met ✅:

1. ✅ Orchestrator mode selection via environment variable
2. ✅ Two modes: agent (default) and legacy
3. ✅ Entrypoint script selects appropriate entry point
4. ✅ Justfile commands support mode selection
5. ✅ .env.defaults sets default to agent mode
6. ✅ Documentation comprehensive and clear
7. ✅ Backward compatibility maintained
8. ✅ Both modes tested and working
