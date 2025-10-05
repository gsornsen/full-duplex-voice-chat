# System Architecture

## Overview

The Realtime Duplex Voice Demo is a two-tier streaming architecture designed for low-latency speech↔speech conversations with barge-in support.

**Architecture Principles:**
- **Modular:** Orchestrator and TTS workers run as separate processes
- **Scalable:** Single-GPU to multi-GPU and multi-host deployments
- **Pluggable:** Multiple TTS models via unified streaming ABI
- **Resilient:** Redis-based service discovery with health monitoring
- **Low-latency:** < 50ms barge-in, < 300ms first audio latency (GPU)

---

## Table of Contents

- [System Architecture Diagram](#system-architecture-diagram)
- [Component Overview](#component-overview)
- [WebSocket Flow Sequence](#websocket-flow-sequence)
- [gRPC Streaming Flow](#grpc-streaming-flow)
- [Session State Machine](#session-state-machine)
- [Deployment Topologies](#deployment-topologies)
- [Data Flow](#data-flow)
- [Barge-In Architecture](#barge-in-architecture)

---

## System Architecture Diagram

### High-Level Components

```mermaid
graph TB
    subgraph "Clients"
        Browser[Browser Client<br/>WebRTC/WebSocket]
        CLI[CLI Client<br/>WebSocket]
    end

    subgraph "Orchestrator Process"
        WS[WebSocket Transport<br/>:8080]
        LK[LiveKit Transport<br/>:7880]
        VAD[VAD<br/>Voice Activity Detection]
        ASR[ASR<br/>Whisper M10+]
        Router[Worker Router<br/>Capability-aware]
        SessionMgr[Session Manager<br/>State Machine]
    end

    subgraph "Service Discovery"
        Redis[(Redis<br/>:6379)]
    end

    subgraph "TTS Workers"
        Worker1[TTS Worker 0<br/>:7001<br/>GPU 0]
        Worker2[TTS Worker 1<br/>:7002<br/>GPU 1]
        WorkerN[TTS Worker N<br/>:700N<br/>GPU N]
    end

    subgraph "Model Storage"
        VP[(Voice Packs<br/>Models + Metadata)]
    end

    Browser --> WS
    CLI --> WS
    Browser -.-> LK
    WS --> SessionMgr
    LK --> SessionMgr
    SessionMgr --> VAD
    SessionMgr --> ASR
    SessionMgr --> Router
    Router --> Redis
    Router --> Worker1
    Router --> Worker2
    Router --> WorkerN
    Worker1 --> VP
    Worker2 --> VP
    WorkerN --> VP
    Worker1 --> Redis
    Worker2 --> Redis
    WorkerN --> Redis

    classDef client fill:#e1f5ff,stroke:#01579b
    classDef orchestrator fill:#fff9c4,stroke:#f57f17
    classDef worker fill:#c8e6c9,stroke:#2e7d32
    classDef storage fill:#f3e5f5,stroke:#4a148c

    class Browser,CLI client
    class WS,LK,VAD,ASR,Router,SessionMgr orchestrator
    class Worker1,Worker2,WorkerN worker
    class Redis,VP storage
```

**Component Responsibilities:**

| Component | Responsibility | Port |
|-----------|---------------|------|
| **Browser Client** | Web UI, WebRTC audio | - |
| **CLI Client** | Command-line testing | - |
| **WebSocket Transport** | JSON message protocol | 8080 |
| **LiveKit Transport** | WebRTC streaming (M3+) | 7880 |
| **VAD** | Speech detection for barge-in | - |
| **ASR** | Speech-to-text (M10+) | - |
| **Worker Router** | Load balancing, routing | - |
| **Session Manager** | State machine, coordination | - |
| **Redis** | Service discovery, registry | 6379 |
| **TTS Workers** | gRPC synthesis servers | 7001+ |
| **Voice Packs** | Model files + metadata | - |

---

## Component Overview

### Orchestrator

**Purpose:** Client-facing service managing sessions and routing requests.

**Key Modules:**
- `transport/`: WebSocket and LiveKit transport implementations
- `vad.py`: Voice activity detection for barge-in
- `asr.py`: Automatic speech recognition (M10+)
- `routing.py`: Worker selection and load balancing
- `registry.py`: Redis-based worker discovery
- `server.py`: Main orchestrator entry point

**Configuration:** `configs/orchestrator.yaml`

**Scaling:** Typically one per deployment, can run multiple for HA

---

### TTS Worker

**Purpose:** GPU-accelerated TTS synthesis with model lifecycle management.

**Key Modules:**
- `worker.py`: gRPC server and adapter host
- `model_manager.py`: Model load/unload, TTL eviction (M4+)
- `tts_base.py`: Adapter interface protocol
- `adapters/`: Model-specific implementations
- `audio/`: Frame repacketization, normalization

**Configuration:** `configs/worker.yaml`

**Scaling:** One per GPU (or multiple per GPU for small models)

---

### Redis

**Purpose:** Service discovery, worker registry, health tracking.

**Data Structures:**
```
worker:<worker-name> → JSON metadata (TTL: 30s)
{
  "name": "tts-worker-0",
  "addr": "grpc://tts-worker:7002",
  "capabilities": {...},
  "resident_models": [...],
  "metrics": {...}
}
```

**Operations:**
- Worker registration on startup
- Heartbeat refresh every 10s
- Orchestrator queries for available workers
- Stale worker eviction after TTL expires

---

## WebSocket Flow Sequence

### Connection and Synthesis

```mermaid
sequenceDiagram
    participant C as Client
    participant O as Orchestrator
    participant R as Redis
    participant W as TTS Worker

    Note over C,W: Connection Phase
    C->>O: WebSocket Upgrade
    O-->>C: 101 Switching Protocols
    O->>R: Query available workers
    R-->>O: Worker list
    O->>W: gRPC: StartSession
    W-->>O: Session created
    O->>C: SessionStartMessage<br/>{session_id}

    Note over C,W: Synthesis Phase
    C->>O: TextMessage<br/>{text: "Hello world"}
    O->>W: gRPC: Synthesize(TextChunk)
    W-->>O: AudioFrame (seq=1)
    O-->>C: AudioMessage {pcm, seq=1}
    W-->>O: AudioFrame (seq=2)
    O-->>C: AudioMessage {pcm, seq=2}
    W-->>O: AudioFrame (seq=3)
    O-->>C: AudioMessage {pcm, seq=3}
    W-->>O: AudioFrame (is_final=true)
    O->>W: gRPC: EndSession
    W-->>O: Session ended
    O->>C: SessionEndMessage<br/>{reason: "completed"}

    Note over C,W: Connection Close
    C->>O: WebSocket Close
    O-->>C: Close Ack
```

### Error Handling

```mermaid
sequenceDiagram
    participant C as Client
    participant O as Orchestrator
    participant W as TTS Worker

    C->>O: TextMessage {text: "..."}
    O->>W: gRPC: Synthesize
    W-->>O: gRPC Error (synthesis failed)
    O->>C: ErrorMessage<br/>{code: "SYNTHESIS_FAILED", message: "..."}
    O->>W: gRPC: EndSession
    O->>C: SessionEndMessage<br/>{reason: "error"}
```

---

## gRPC Streaming Flow

### Orchestrator → Worker Communication

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant W as Worker<br/>gRPC Server
    participant M as Model Manager
    participant A as Adapter<br/>(e.g., CosyVoice)

    Note over O,A: Session Initialization
    O->>W: StartSession(session_id, model_id)
    W->>M: Ensure model loaded
    M-->>W: Model ready
    W-->>O: StartSessionResponse<br/>{success: true}

    Note over O,A: Streaming Synthesis
    O->>W: Synthesize(stream TextChunk)
    W->>A: synthesize_streaming(text)
    loop For each 20ms frame
        A-->>W: PCM frame (960 samples @ 48kHz)
        W-->>O: AudioFrame (seq++)
    end
    A-->>W: End of synthesis
    W-->>O: AudioFrame (is_final=true)

    Note over O,A: Session Cleanup
    O->>W: EndSession(session_id)
    W->>M: Release model refcount
    W-->>O: EndSessionResponse<br/>{success: true}
```

### Control Commands (Barge-In)

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant W as Worker
    participant A as Adapter

    Note over O,A: Normal Synthesis
    O->>W: Synthesize(stream TextChunk)
    W->>A: synthesize_streaming(text)
    A-->>W: AudioFrame (seq=1)
    W-->>O: AudioFrame (seq=1)
    A-->>W: AudioFrame (seq=2)
    W-->>O: AudioFrame (seq=2)

    Note over O,A: Barge-In Detected
    O->>W: Control(PAUSE)
    W->>A: pause()
    A-->>A: Stop emitting frames<br/>(< 50ms)
    W-->>O: ControlResponse<br/>{success: true, timestamp_ms}

    Note over O,A: Resume After Silence
    O->>W: Control(RESUME)
    W->>A: resume()
    A-->>W: AudioFrame (seq=3)
    W-->>O: AudioFrame (seq=3)
    A-->>W: AudioFrame (seq=4)
    W-->>O: AudioFrame (seq=4)
```

---

## Session State Machine

### Orchestrator Session States

```mermaid
stateDiagram-v2
    [*] --> INITIALIZING: WebSocket Connect

    INITIALIZING --> LISTENING: StartSession Success
    INITIALIZING --> ERROR: StartSession Failed

    LISTENING --> SPEAKING: TextMessage Received
    LISTENING --> CLOSED: Client Disconnect

    SPEAKING --> LISTENING: Synthesis Complete
    SPEAKING --> BARGED_IN: VAD Detects Speech
    SPEAKING --> ERROR: Synthesis Failed
    SPEAKING --> CLOSED: Client Disconnect

    BARGED_IN --> LISTENING: STOP Command
    BARGED_IN --> SPEAKING: RESUME Command
    BARGED_IN --> CLOSED: Client Disconnect

    ERROR --> CLOSED: Send Error Message
    CLOSED --> [*]: Cleanup Resources

    note right of LISTENING
        Waiting for user input
        VAD monitoring active
    end note

    note right of SPEAKING
        TTS synthesis active
        Audio frames streaming
    end note

    note right of BARGED_IN
        Synthesis paused
        Awaiting control command
    end note
```

**State Descriptions:**

| State | Description | Valid Transitions |
|-------|-------------|-------------------|
| **INITIALIZING** | Session starting, connecting to worker | LISTENING, ERROR |
| **LISTENING** | Idle, waiting for text input | SPEAKING, CLOSED |
| **SPEAKING** | TTS synthesis in progress | LISTENING, BARGED_IN, ERROR, CLOSED |
| **BARGED_IN** | Synthesis paused due to interruption | LISTENING, SPEAKING, CLOSED |
| **ERROR** | Error occurred during session | CLOSED |
| **CLOSED** | Session terminated | (final state) |

---

## Deployment Topologies

### Single-GPU Development (M2 Current)

```mermaid
graph LR
    subgraph "Single Host"
        CLI[CLI Client]
        O[Orchestrator<br/>:8080]
        R[(Redis<br/>:6379)]
        W[TTS Worker<br/>:7001<br/>GPU 0]
    end

    CLI --> O
    O --> R
    O --> W
    W --> R

    style W fill:#c8e6c9
```

**Use Case:** Development, testing, demos

**Resources:** 1 GPU, 8GB+ VRAM

**Command:**
```bash
docker compose up --build
```

---

### Multi-GPU Same Host (M9+)

```mermaid
graph LR
    subgraph "Single Host"
        Browser[Browser]
        CLI[CLI Client]
        O[Orchestrator<br/>:8080]
        R[(Redis<br/>:6379)]
        W1[Worker 0<br/>:7001<br/>GPU 0]
        W2[Worker 1<br/>:7002<br/>GPU 1]
        W3[Worker 2<br/>:7003<br/>GPU 2]
    end

    Browser --> O
    CLI --> O
    O --> R
    O --> W1
    O --> W2
    O --> W3
    W1 --> R
    W2 --> R
    W3 --> R

    style W1 fill:#c8e6c9
    style W2 fill:#c8e6c9
    style W3 fill:#c8e6c9
```

**Use Case:** Production, high throughput

**Resources:** 3+ GPUs, 24GB+ VRAM each

**Setup:**
```bash
# Worker 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 just run-tts-cosy

# Worker 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 just run-tts-xtts

# Worker 2 on GPU 2
CUDA_VISIBLE_DEVICES=2 just run-tts-sesame
```

---

### Multi-Host LAN Deployment (M13)

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[nginx/traefik<br/>:443 HTTPS]
    end

    subgraph "Orchestrator Cluster"
        O1[Orchestrator 1]
        O2[Orchestrator 2]
    end

    subgraph "Central Services"
        Redis[(Redis Cluster)]
    end

    subgraph "TTS Worker Host 1"
        W1[Worker 0<br/>GPU 0]
        W2[Worker 1<br/>GPU 1]
    end

    subgraph "TTS Worker Host 2"
        W3[Worker 2<br/>GPU 0]
        W4[Worker 3<br/>GPU 1]
    end

    LB --> O1
    LB --> O2
    O1 --> Redis
    O2 --> Redis
    O1 --> W1
    O1 --> W2
    O2 --> W3
    O2 --> W4
    W1 --> Redis
    W2 --> Redis
    W3 --> Redis
    W4 --> Redis

    style W1 fill:#c8e6c9
    style W2 fill:#c8e6c9
    style W3 fill:#c8e6c9
    style W4 fill:#c8e6c9
```

**Use Case:** Enterprise, high availability

**Resources:** Multiple hosts, distributed GPUs

**Features:**
- Load balancing across orchestrators
- Redis cluster for HA
- Cross-host worker discovery
- Automatic failover

---

## Data Flow

### Audio Frame Pipeline

```mermaid
graph LR
    subgraph "TTS Worker"
        Model[TTS Model<br/>Output:<br/>Variable rate<br/>Variable chunk size]
        Resample[Resample<br/>to 48kHz]
        Frame[Reframe<br/>to 20ms chunks]
        Normalize[Loudness<br/>Normalize<br/>-16 LUFS]
        Encode[Base64<br/>Encode]
    end

    subgraph "Transport"
        GRPC[gRPC<br/>Stream]
        WS[WebSocket<br/>JSON]
    end

    subgraph "Client"
        Decode[Base64<br/>Decode]
        Play[Audio<br/>Playback]
    end

    Model --> Resample
    Resample --> Frame
    Frame --> Normalize
    Normalize --> Encode
    Encode --> GRPC
    GRPC --> WS
    WS --> Decode
    Decode --> Play

    style Model fill:#fff9c4
    style Frame fill:#c8e6c9
    style Play fill:#e1f5ff
```

**Frame Specifications:**
- **Sample Rate:** 48000 Hz (fixed)
- **Frame Duration:** 20 ms (960 samples)
- **Format:** 16-bit signed PCM, little-endian
- **Channels:** Mono (1 channel)
- **Frame Size:** 1920 bytes
- **Encoding:** Base64 for WebSocket transport
- **Target Loudness:** -16 LUFS

---

## Barge-In Architecture

### Barge-In Timing Diagram

```mermaid
sequenceDiagram
    participant U as User (Mic)
    participant O as Orchestrator
    participant V as VAD
    participant W as Worker
    participant S as Speakers

    Note over U,S: Normal Synthesis
    O->>W: Synthesize("Long text...")
    W->>S: Frame 1
    W->>S: Frame 2
    W->>S: Frame 3

    Note over U,S: User Interrupts
    U->>O: Speech detected
    O->>V: Process audio
    V-->>O: Speech detected (50-100ms buffer)
    O->>W: Control(PAUSE)
    Note over W: Stop emitting<br/>< 50ms
    W-->>O: ControlResponse

    Note over U,S: Pause Duration
    U->>O: Continued speech...
    Note over S: Silence

    Note over U,S: User Stops Speaking
    U->>O: Silence
    O->>V: Process silence
    V-->>O: Silence detected (200-500ms)
    O->>W: Control(RESUME)
    W->>S: Frame 4
    W->>S: Frame 5
```

**Latency Budget:**

| Stage | Latency | Target |
|-------|---------|--------|
| VAD Detection | 10-30ms | < 50ms |
| Network (Orch→Worker) | 1-5ms | < 10ms |
| Worker Pause Processing | 2-10ms | < 20ms |
| **Total Pause Latency** | **13-45ms** | **< 50ms (p95)** |

---

### VAD Configuration Impact

```mermaid
graph TD
    Audio[Input Audio] --> VAD{VAD<br/>Aggressiveness}

    VAD -->|0: Least Aggressive| High[High Sensitivity<br/>More False Positives<br/>Better for Soft Speech]
    VAD -->|1: Low| Med1[Balanced Sensitivity<br/>Some False Positives]
    VAD -->|2: Moderate| Med2[Good Balance<br/>Recommended Default]
    VAD -->|3: Most Aggressive| Low[High Specificity<br/>Fewer False Positives<br/>May Miss Soft Speech]

    style Med2 fill:#c8e6c9
```

**Tuning Guidelines:**
- **Noisy Environment:** Use aggressiveness = 3
- **Quiet Environment:** Use aggressiveness = 1
- **General Use:** Use aggressiveness = 2 (default)

---

## Model Manager Architecture (M4+)

```mermaid
graph TB
    subgraph "Worker Process"
        GRPC[gRPC Server]
        MM[Model Manager]
        LRU[LRU Cache]
        TTL[TTL Eviction]

        subgraph "Model Slots"
            Slot1[Slot 1<br/>Model A<br/>refcount=2<br/>last_used=now]
            Slot2[Slot 2<br/>Model B<br/>refcount=0<br/>last_used=-5min]
            Slot3[Slot 3<br/>Empty]
        end

        subgraph "Adapters"
            AdapterA[CosyVoice Adapter]
            AdapterB[XTTS Adapter]
        end
    end

    subgraph "Storage"
        VP[(Voice Packs)]
    end

    GRPC --> MM
    MM --> LRU
    MM --> TTL
    MM --> Slot1
    MM --> Slot2
    MM --> Slot3
    Slot1 --> AdapterA
    Slot2 --> AdapterB
    AdapterA --> VP
    AdapterB --> VP

    TTL -.->|Evict if idle > TTL| Slot2
    LRU -.->|Evict if slots full| Slot2

    style Slot1 fill:#c8e6c9
    style Slot2 fill:#ffccbc
    style Slot3 fill:#f5f5f5
```

**Model Manager Operations:**

1. **Load Model:**
   - Check if already resident → return
   - Check resident_cap → evict LRU if needed
   - Load from voice pack
   - Warmup (~300ms synthetic utterance)
   - Register in slots

2. **Eviction Logic:**
   - Check every `evict_check_interval_ms` (30s)
   - Evict if: `refcount == 0` AND `idle_time > ttl_ms`
   - Respect `min_residency_ms` (prevent thrashing)
   - LRU eviction when `resident_count > resident_cap`

3. **Reference Counting:**
   - Increment on session start
   - Decrement on session end
   - Never evict if `refcount > 0`

---

## Network Protocols

### WebSocket Protocol Stack

```
┌─────────────────────────────────────┐
│         Application Layer           │
│    JSON Messages (Text/Audio)       │
├─────────────────────────────────────┤
│         WebSocket Layer             │
│       RFC 6455 Framing              │
├─────────────────────────────────────┤
│          HTTP/1.1 Upgrade           │
│       (Connection Handshake)        │
├─────────────────────────────────────┤
│            TCP Layer                │
│          Port 8080                  │
└─────────────────────────────────────┘
```

### gRPC Protocol Stack

```
┌─────────────────────────────────────┐
│       Application Layer             │
│    Protobuf Messages (Binary)       │
├─────────────────────────────────────┤
│          gRPC Layer                 │
│   Bidirectional Streaming RPC       │
├─────────────────────────────────────┤
│          HTTP/2 Layer               │
│     Multiplexing, Flow Control      │
├─────────────────────────────────────┤
│            TCP Layer                │
│         Ports 7001+                 │
└─────────────────────────────────────┘
```

---

## Security Architecture (Production)

```mermaid
graph TB
    Internet[Internet]
    FW[Firewall]
    LB[Load Balancer<br/>TLS Termination]
    OZ[Orchestrator Zone<br/>DMZ]
    WZ[Worker Zone<br/>Private Network]
    RZ[Redis Zone<br/>Private Network]

    Internet --> FW
    FW --> LB
    LB --> OZ
    OZ --> WZ
    OZ --> RZ
    WZ --> RZ

    style OZ fill:#fff9c4
    style WZ fill:#c8e6c9
    style RZ fill:#f3e5f5
```

**Security Layers:**

| Component | Security Measure | Implementation |
|-----------|-----------------|----------------|
| **Load Balancer** | TLS termination | Let's Encrypt, nginx |
| **Orchestrator** | API key auth | Header validation |
| **gRPC (Orch→Worker)** | mTLS (optional) | Certificate validation |
| **Redis** | AUTH password | Redis ACL |
| **Network** | Private VLANs | Firewall rules |
| **Secrets** | Environment vars | Vault, AWS Secrets Manager |

---

## Related Documentation

- [WebSocket Protocol Specification](../WEBSOCKET_PROTOCOL.md) - Message format details
- [Configuration Reference](../CONFIGURATION_REFERENCE.md) - System configuration
- [Component READMEs](../../src/README.md) - Module documentation
- [Quick Start Guide](../QUICKSTART.md) - Deployment walkthrough
- [Performance Guide](../PERFORMANCE.md) - Tuning and optimization

---

## Changelog

**v0.2.0 (M2):**
- Initial architecture documentation
- System component diagram
- WebSocket and gRPC flow diagrams
- Session state machine
- Deployment topology examples

**Future:**
- M3: LiveKit WebRTC architecture
- M4: Model Manager detailed architecture
- M9: Multi-worker routing architecture
- M13: Multi-host scaling architecture
