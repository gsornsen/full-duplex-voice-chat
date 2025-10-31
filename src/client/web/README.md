# Realtime Duplex Voice Demo - Web Client

Browser-based voice interface for the Realtime Duplex Voice Demo system, built with Next.js and LiveKit.

## Features

- **Full-duplex voice conversation** with natural turn-taking
- **Barge-in support** - interrupt the agent mid-sentence (< 50ms latency)
- **Low-latency TTS** - first audio response in < 300ms
- **Hot-swappable TTS models** - backend supports multiple TTS engines
- **Text chat fallback** - type messages as alternative to voice
- **Modern UI** - dark/light theme with audio visualization

Based on [LiveKit's agent-starter-react](https://github.com/livekit-examples/agent-starter-react) template.

## Quick Start

### 1. Start Backend Services

From the project root:

```bash
docker compose up -d
docker compose ps  # Verify all services are healthy
```

This starts Redis, LiveKit server, orchestrator, and TTS worker.

### 2. Install Dependencies

```bash
cd src/client/web
pnpm install
```

### 3. Configure Environment

```bash
cp .env.example .env.local
```

Default values work for local Docker Compose setup.

### 4. Start Web Client

```bash
pnpm dev
```

Open http://localhost:3000 in your browser.

## Documentation

- **[SETUP.md](./SETUP.md)** - Complete setup guide with troubleshooting
- **[USER_GUIDE.md](./USER_GUIDE.md)** - How to use the voice interface
- **[app-config.ts](./app-config.ts)** - Customize branding and features

## Project Structure

```
src/client/web/
├── app/                    # Next.js App Router
│   ├── (app)/             # Main app routes
│   ├── api/               # API routes (connection details)
│   └── components/        # Page components
├── components/            # React components
│   ├── livekit/          # LiveKit-specific UI
│   └── ui/               # Reusable UI components
├── hooks/                # Custom React hooks
├── lib/                  # Utilities and types
├── public/               # Static assets
├── app-config.ts         # App configuration
└── package.json          # Dependencies
```

## Configuration

### App Settings (app-config.ts)

```typescript
export const APP_CONFIG_DEFAULTS: AppConfig = {
  companyName: 'Realtime Duplex Voice Demo',
  pageTitle: 'Realtime Duplex Voice Chat',
  supportsChatInput: true,
  supportsVideoInput: false,  // Voice-only
  supportsScreenShare: false,
  agentName: 'voice-assistant',
};
```

### Environment Variables (.env.local)

```env
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_URL=ws://localhost:7880
```

## Development

```bash
pnpm dev       # Start development server
pnpm build     # Build for production
pnpm start     # Start production server
pnpm lint      # Lint code
pnpm format    # Format code
```

## Architecture

```
Browser          LiveKit         Orchestrator      TTS Workers
  │                │                  │                 │
  │──WebRTC────────│                  │                 │
  │                │──WebSocket───────│                 │
  │                │                  │────gRPC─────────│
  └────────────────┴──────────────────┴─────────────────┘
```

The web client uses LiveKit for WebRTC transport, which connects to the orchestrator (LiveKit agent) that manages TTS worker routing.

## Technology Stack

- **Next.js 15** - React framework with App Router
- **React 19** - UI library
- **LiveKit** - WebRTC platform for real-time communication
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **pnpm** - Package manager

## Related Documentation

- [Project PRD](../../../project_documentation/PRD.md)
- [Technical Design](../../../project_documentation/TDD.md)
- [Integration Tests](../../../tests/integration/README.md)
- [Performance Tests](../../../tests/performance/README.md)
