---
title: "Session Management"
tags: ["session", "multi-turn", "timeout", "state-machine", "orchestrator", "m10-polish"]
related_files:
  - "src/orchestrator/session.py"
  - "src/orchestrator/config.py"
  - "tests/unit/orchestrator/test_session.py"
  - "tests/integration/test_multi_turn*.py"
dependencies:
  - ".claude/modules/features/vad.md#state-aware-vad-gating"
  - ".claude/modules/architecture.md#state-machine"
estimated_tokens: 800
priority: "high"
keywords: ["session", "multi-turn", "idle timeout", "WAITING_FOR_INPUT", "state machine", "conversation"]
---

# Session Management

**Last Updated**: 2025-10-17

Session Management enables multi-turn conversations with idle timeout, session limits, and graceful cleanup.

> 📖 **Quick Summary**: See [CLAUDE.md#important-patterns](../../../CLAUDE.md#important-patterns)

## Overview

**Implementation**: `src/orchestrator/session.py` (M10 Polish)

**Purpose**: Manage conversation sessions across multiple user interactions.

**Key Features** (M10 Polish):
- Multi-turn conversation support (sessions persist between interactions)
- Configurable idle timeout (default: 5 minutes)
- Session duration limits (default: 1 hour)
- Message count limits (default: 100 messages)
- Graceful timeout handling with automatic cleanup
- Non-blocking timeout implementation using `asyncio.wait_for()`

## State Machine

```
IDLE
  ↓ (user connects)
LISTENING
  ↓ (user speaks, ASR transcribes)
WAITING_FOR_INPUT (optional LLM processing)
  ↓ (TTS synthesis starts)
SPEAKING
  ↓ (user interrupts)
BARGED_IN
  ↓ (VAD detects silence)
LISTENING or WAITING_FOR_INPUT
  ↓ (idle timeout or session limit)
TERMINATED
```

**New State** (M10 Polish): **WAITING_FOR_INPUT**
- Session transitions here after TTS synthesis completes
- Waits up to `idle_timeout_seconds` for next user input
- Maintains conversation context (multi-turn)
- Gracefully terminates on timeout or session limits

## Configuration

```yaml
# configs/orchestrator.yaml
session:
  idle_timeout_seconds: 300  # 5 minutes
  max_session_duration_seconds: 3600  # 1 hour
  max_messages_per_session: 100
```

## Usage

```python
from src.orchestrator.session import SessionState, Session
from src.orchestrator.config import SessionConfig

config = SessionConfig(
    idle_timeout_seconds=300,
    max_session_duration_seconds=3600
)

session = Session(session_id="user_123", config=config)

# Start session
await session.transition_to(SessionState.LISTENING)

# After TTS completes
await session.transition_to(SessionState.WAITING_FOR_INPUT)

# Automatically times out after 5 min if no user input
# Or transitions to LISTENING when user starts speaking

# Check session status
if session.is_terminated:
    # Session ended due to timeout or limits
    pass
```

## Timeout Handling

### Non-Blocking Timeout

```python
# Example from orchestrator
try:
    user_input = await asyncio.wait_for(
        session.wait_for_input(),
        timeout=config.idle_timeout_seconds
    )
except asyncio.TimeoutError:
    # Gracefully terminate session
    await session.transition_to(SessionState.TERMINATED)
```

### Session Limits

Session automatically terminates when:
- Idle for > `idle_timeout_seconds` (default: 5 min)
- Total duration > `max_session_duration_seconds` (default: 1 hour)
- Message count > `max_messages_per_session` (default: 100)

## Test Coverage

**Total**: 34/34 passing (M10 Polish)
- Session timeout: 21/21 tests
- Multi-turn conversation: 13/13 tests

## Implementation Files

- `src/orchestrator/session.py`: Session state machine
- `src/orchestrator/config.py`: SessionConfig
- `tests/unit/orchestrator/test_session.py`: Session tests
- `tests/integration/test_multi_turn.py`: Multi-turn tests

## References

- **VAD**: [.claude/modules/features/vad.md](vad.md)
- **Architecture**: [.claude/modules/architecture.md](../architecture.md)
- **Core Documentation**: [CLAUDE.md](../../../CLAUDE.md)

---

**Last Updated**: 2025-10-17
**Status**: Complete (M10 Polish)
