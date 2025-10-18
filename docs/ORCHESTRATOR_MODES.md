# Orchestrator Mode Selection

The orchestrator supports two operating modes to accommodate different client types:

## Modes Overview

### 1. Agent Mode (Default)
**Use Case**: Web frontend with LiveKit Agent protocol

- **Entry Point**: `src.orchestrator.agent`
- **Protocol**: LiveKit Agent protocol (WebRTC)
- **Client**: Next.js web frontend (`src/client/web`)
- **Features**:
  - LiveKit Agent SDK integration
  - WebRTC-based real-time communication
  - Custom plugins: WhisperX STT, gRPC TTS, OpenAI LLM
  - Automatic agent joining on participant connection
- **Requirements**:
  - `OPENAI_API_KEY` environment variable (for LLM integration)
  - LiveKit server running

### 2. Legacy Mode
**Use Case**: CLI client with WebSocket/HTTP API

- **Entry Point**: `src.orchestrator.server`
- **Protocol**: WebSocket + HTTP
- **Client**: CLI client (`src/client/cli_client.py`)
- **Features**:
  - WebSocket transport for audio/text streaming
  - HTTP health check endpoints
  - Direct gRPC TTS integration
  - VAD-based barge-in detection
  - ASR transcription (Whisper/WhisperX)
- **Requirements**: None (works standalone)

## Usage

### Quick Start Commands

```bash
# Default: LiveKit Agent mode with Piper
just dev

# Explicit agent mode
just dev-agent

# Legacy mode for CLI client
just dev-legacy

# With specific TTS model
just dev piper agent        # Agent mode with Piper
just dev cosyvoice2 legacy  # Legacy mode with CosyVoice
```

### Environment Variable Control

Set `ORCHESTRATOR_MODE` in your environment or `.env.local`:

```bash
# Agent mode (default)
ORCHESTRATOR_MODE=agent just dev

# Legacy mode
ORCHESTRATOR_MODE=legacy just dev
```

### Docker Compose Direct Usage

```bash
# Agent mode (default)
docker compose up

# Legacy mode
ORCHESTRATOR_MODE=legacy docker compose up
```

## Architecture Comparison

### Agent Mode Flow
```
Web Client (LiveKit) → LiveKit Server → Agent Orchestrator → TTS Worker
                                            ↓
                                    WhisperX STT + OpenAI LLM
```

### Legacy Mode Flow
```
CLI Client (WebSocket) → Legacy Orchestrator → TTS Worker
                              ↓
                        VAD + ASR (Whisper/WhisperX)
```

## Configuration

### Agent Mode Configuration
```yaml
# .env or .env.local
ORCHESTRATOR_MODE=agent
LIVEKIT_URL=ws://livekit:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=devsecret1234567890abcdefghijklmn
OPENAI_API_KEY=sk-...  # Required for LLM
```

### Legacy Mode Configuration
```yaml
# .env or .env.local
ORCHESTRATOR_MODE=legacy
ORCHESTRATOR_WS_PORT=8080
ORCHESTRATOR_HEALTH_PORT=8081
ASR_ENABLED=true
ASR_ADAPTER=whisperx
```

## Testing Both Modes

### Test Agent Mode
1. Start services: `just dev-agent`
2. Start web frontend: `just web-dev`
3. Navigate to `https://voicechat.local:8443`
4. Agent should join room automatically

### Test Legacy Mode
1. Start services: `just dev-legacy`
2. Run CLI client: `just cli`
3. Send text or audio
4. Receive synthesized speech

## Entrypoint Script

The orchestrator uses a wrapper script (`docker/entrypoint-orchestrator.sh`) to select the appropriate entry point based on `ORCHESTRATOR_MODE`:

```bash
# Agent mode
exec uv run python -m src.orchestrator.agent dev

# Legacy mode
exec uv run python -m src.orchestrator.server --config configs/orchestrator.docker.yaml
```

## Migration Guide

### From `just dev-agent-piper` to `just dev-agent`

**Before** (deprecated):
```bash
just dev-agent-piper
```

**After** (new unified workflow):
```bash
just dev-agent piper
# or simply
just dev  # Defaults to agent mode with piper
```

### From `docker compose up` to Agent Mode

**Before**: `docker compose up` ran legacy WebSocket server

**After**: `docker compose up` runs LiveKit Agent mode (default)

To restore legacy behavior:
```bash
ORCHESTRATOR_MODE=legacy docker compose up
```

## Troubleshooting

### Agent Mode: "No agent joined the room"
- Check `OPENAI_API_KEY` is set
- Verify LiveKit server is running: `docker ps | grep livekit`
- Check orchestrator logs: `just dev-logs orchestrator`

### Legacy Mode: "Connection refused on port 8080"
- Verify mode is set: `echo $ORCHESTRATOR_MODE`
- Check orchestrator is running in legacy mode: `just dev-logs orchestrator`
- Confirm health endpoint: `curl http://localhost:8081/health`

### Both Modes: "TTS worker unavailable"
- Check TTS worker is running: `docker ps | grep tts-worker`
- Verify Redis is healthy: `docker ps | grep redis`
- Check worker connection: `just dev-logs tts0`

## Advanced: Running Both Modes Simultaneously

You can run both modes on the same machine by using different containers:

```bash
# Terminal 1: Agent mode (default ports)
just dev-agent

# Terminal 2: Legacy mode (different ports)
# Edit docker-compose.yml to use different ports for legacy service
ORCHESTRATOR_MODE=legacy docker compose -f docker-compose.legacy.yml up
```

Note: This requires creating a separate compose file with different port mappings.

## References

- [LiveKit Agent SDK](https://docs.livekit.io/agents/)
- [WebSocket Transport](../src/orchestrator/transport/websocket_transport.py)
- [LiveKit Transport](../src/orchestrator/transport/livekit_transport.py)
- [Agent Entry Point](../src/orchestrator/agent.py)
- [Legacy Server Entry Point](../src/orchestrator/server.py)
