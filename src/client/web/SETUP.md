# Web Client Setup Guide

This guide explains how to set up and run the web client for the Realtime Duplex Voice Demo.

## Overview

The web client is a Next.js application that provides a browser-based interface for voice conversations with the TTS system. It uses LiveKit for WebRTC transport and connects to the orchestrator service through the LiveKit server.

## Architecture

```
┌─────────────┐      WebRTC       ┌──────────────┐      gRPC      ┌─────────────┐
│  Browser    │ ◄──────────────► │   LiveKit    │ ◄────────────► │ Orchestrator│
│ (Next.js)   │                   │   Server     │                 │   (Agent)   │
└─────────────┘                   └──────────────┘                 └─────────────┘
                                                                            │
                                                                            │ gRPC
                                                                            ▼
                                                                    ┌─────────────┐
                                                                    │ TTS Workers │
                                                                    └─────────────┘
```

## Prerequisites

- **Node.js**: 18.x or later (20.x recommended)
- **pnpm**: 9.15.9 or later (specified in package.json)
- **Docker & Docker Compose**: For running backend services
- **Port availability**: 3000 (web client), 7880 (LiveKit), 8080 (orchestrator)

## Quick Start

### 1. Start Backend Services

First, ensure all backend services are running using Docker Compose:

```bash
# From the project root directory
docker compose up -d

# Verify services are running
docker compose ps

# Check logs if needed
docker compose logs -f
```

This starts:
- **Redis** (port 6380) - Service discovery
- **LiveKit Server** (port 7880) - WebRTC transport
- **Orchestrator** (port 8080) - Voice agent
- **TTS Worker** (port 7001) - Text-to-speech synthesis

Wait 10-15 seconds for all services to be healthy.

### 2. Install Web Client Dependencies

```bash
cd src/client/web
pnpm install
```

**Note**: The project uses pnpm 9.15.9 as specified in package.json. If you don't have pnpm:

```bash
npm install -g pnpm@9.15.9
```

### 3. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env.local
```

The default values in `.env.example` are configured for local development with Docker Compose:

```env
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_URL=ws://localhost:7880
```

**Important**: These defaults match the configuration in `configs/livekit.yaml` and `docker-compose.yml`. Only change them if you've modified those files.

### 4. Start the Development Server

```bash
pnpm dev
```

The web client will start on http://localhost:3000

Open your browser and navigate to http://localhost:3000

## Configuration

### App Configuration (app-config.ts)

The web client's branding and features are configured in `app-config.ts`:

```typescript
export const APP_CONFIG_DEFAULTS: AppConfig = {
  companyName: 'Realtime Duplex Voice Demo',
  pageTitle: 'Realtime Duplex Voice Chat',
  pageDescription: 'Full-duplex voice conversation with barge-in support',

  supportsChatInput: true,      // Enable text chat
  supportsVideoInput: false,    // Video disabled (voice-only)
  supportsScreenShare: false,   // Screen share disabled
  isPreConnectBufferEnabled: true,

  startButtonText: 'Start voice chat',
  agentName: 'voice-assistant', // Default agent name
};
```

### LiveKit Configuration (configs/livekit.yaml)

The LiveKit server configuration is in `configs/livekit.yaml`:

```yaml
port: 7880
keys:
  devkey: secret  # Must match .env.local
room:
  auto_create: true
  empty_timeout: 300
  max_participants: 10
```

## Development Workflow

### Running in Development Mode

```bash
cd src/client/web
pnpm dev
```

Changes to the code will automatically trigger hot reload.

### Building for Production

```bash
cd src/client/web
pnpm build
pnpm start
```

The production build optimizes the application for performance.

### Code Quality

```bash
# Lint the code
pnpm lint

# Format the code
pnpm format

# Check formatting
pnpm format:check
```

## Troubleshooting

### Port Already in Use

If port 3000 is already in use, you can specify a different port:

```bash
PORT=3001 pnpm dev
```

### LiveKit Connection Failed

If the web client can't connect to LiveKit:

1. Verify LiveKit server is running:
   ```bash
   docker compose ps livekit
   ```

2. Check LiveKit logs:
   ```bash
   docker compose logs -f livekit
   ```

3. Verify the URL in `.env.local` matches the LiveKit service:
   ```bash
   # Should be ws://localhost:7880 for local Docker
   cat .env.local | grep LIVEKIT_URL
   ```

4. Test LiveKit health endpoint:
   ```bash
   curl http://localhost:7881/
   ```

### Backend Services Not Responding

If the orchestrator or TTS workers aren't responding:

1. Check all services are healthy:
   ```bash
   docker compose ps
   ```

2. Restart services:
   ```bash
   docker compose restart
   ```

3. Check orchestrator logs:
   ```bash
   docker compose logs -f orchestrator
   ```

4. Check TTS worker logs:
   ```bash
   docker compose logs -f tts0
   ```

### Environment Variables Not Loading

If changes to `.env.local` aren't taking effect:

1. Restart the Next.js development server (Ctrl+C and `pnpm dev` again)
2. Ensure the file is named `.env.local` exactly (not `.env` or `.env.development`)
3. Check for syntax errors in the `.env.local` file

### PNPM Installation Issues

If pnpm has issues installing dependencies:

```bash
# Clear pnpm cache
pnpm store prune

# Remove node_modules and lock file
rm -rf node_modules pnpm-lock.yaml

# Reinstall
pnpm install
```

## Next Steps

After successfully starting the web client, see [USER_GUIDE.md](./USER_GUIDE.md) for instructions on how to use the voice chat interface.

## Additional Resources

- [LiveKit Documentation](https://docs.livekit.io/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Project PRD](../../../project_documentation/PRD.md)
- [Project TDD](../../../project_documentation/TDD.md)
