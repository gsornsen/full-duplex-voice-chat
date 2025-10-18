# Quick Reference - Development Commands

## Orchestrator Modes

### Agent Mode (Default) - For Web Frontend
```bash
just dev                # Default: Agent + Piper
just dev-agent          # Explicit agent mode
just dev-agent piper    # Agent mode with Piper
just dev piper agent    # Same as above
```

### Legacy Mode - For CLI Client
```bash
just dev-legacy         # Legacy mode + Piper
just dev-legacy piper   # Explicit model
just dev piper legacy   # Same as above
```

## Model Selection

### Piper (CPU, baseline)
```bash
just dev piper          # Agent mode
just dev piper agent    # Explicit agent mode
just dev piper legacy   # Legacy mode
```

### CosyVoice2 (GPU, high quality)
```bash
just dev cosyvoice2          # Agent mode
just dev cosyvoice2 agent    # Explicit agent mode
just dev cosyvoice2 legacy   # Legacy mode
```

## Common Workflows

### Web Frontend Development
```bash
# Terminal 1: Start backend (agent mode)
just dev-agent

# Terminal 2: Start web frontend
just web-dev

# Access at: https://voicechat.local:8443
```

### CLI Client Development
```bash
# Terminal 1: Start backend (legacy mode)
just dev-legacy

# Terminal 2: Run CLI client
just cli

# WebSocket at: ws://localhost:8080
```

## Logs and Status

```bash
just dev-logs              # Follow all service logs
just dev-logs orchestrator # Follow specific service
just dev-status            # Show service status
```

## Cleanup

```bash
just dev-stop    # Stop services (keeps volumes)
just dev-clean   # Stop services + remove volumes
just dev-reset   # Clean + restart
```

## Advanced Usage

### Hot-Swap TTS Model
```bash
just dev-switch cosyvoice2  # Switch to CosyVoice without full restart
```

### Environment Variable Override
```bash
ORCHESTRATOR_MODE=legacy just dev  # Override mode
DEFAULT_MODEL=cosyvoice2 just dev  # Override model
```

### Infrastructure Only
```bash
just dev-infra  # Start Redis, LiveKit, Caddy only
```

## Troubleshooting

### Agent Mode Issues
```bash
# Check if agent is running
just dev-logs orchestrator | grep "LiveKit Agent"

# Verify OpenAI API key
echo $OPENAI_API_KEY

# Check agent joined room
just dev-logs orchestrator | grep "Agent session started"
```

### Legacy Mode Issues
```bash
# Check if legacy server is running
just dev-logs orchestrator | grep "Legacy WebSocket"

# Test health endpoint
curl http://localhost:8081/health

# Test WebSocket connection
just cli
```

### Worker Issues
```bash
# Check worker status
just dev-logs tts0

# Check Redis connection
docker exec -it redis-tts redis-cli PING

# Check worker registration
docker exec -it redis-tts redis-cli KEYS "worker:*"
```

## Quality Checks

```bash
just lint       # Run ruff linting
just fix        # Auto-fix linting issues
just typecheck  # Run mypy type checking
just test       # Run pytest (unit tests)
just ci         # Run all checks
```

## Direct Docker Compose

```bash
# Agent mode (default)
docker compose up

# Legacy mode
ORCHESTRATOR_MODE=legacy docker compose up

# With specific profile
docker compose --profile cosyvoice up

# View logs
docker compose logs -f orchestrator
```

## Mode Selection Matrix

| Use Case | Command | Transport | Entry Point |
|----------|---------|-----------|-------------|
| Web frontend | `just dev-agent` | LiveKit WebRTC | `src.orchestrator.agent` |
| CLI client | `just dev-legacy` | WebSocket | `src.orchestrator.server` |
| Default | `just dev` | LiveKit WebRTC | `src.orchestrator.agent` |

## Environment Files

```bash
.env.defaults       # Default settings (DO NOT EDIT)
.env.local          # Your local overrides (gitignored)
.env                # Loaded if exists (gitignored)
```

To override orchestrator mode persistently:
```bash
echo "ORCHESTRATOR_MODE=legacy" >> .env.local
```

## See Also

- [ORCHESTRATOR_MODES.md](ORCHESTRATOR_MODES.md) - Detailed mode documentation
- [DEVELOPMENT.md](DEVELOPMENT.md) - Full development guide
- [CURRENT_STATUS.md](CURRENT_STATUS.md) - Implementation status
