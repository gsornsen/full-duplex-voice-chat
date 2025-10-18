---
title: "Model Manager"
tags: ["model-manager", "lifecycle", "ttl", "lru", "eviction", "tts", "m4"]
related_files:
  - "src/tts/model_manager.py"
  - "tests/unit/tts/test_model_manager.py"
  - "tests/integration/test_model_lifecycle.py"
dependencies:
  - ".claude/modules/architecture.md#tts-worker-layer"
estimated_tokens: 800
priority: "high"
keywords: ["model manager", "model lifecycle", "TTL eviction", "LRU eviction", "load model", "unload model", "warmup"]
---

# Model Manager

**Last Updated**: 2025-10-17

Model Manager handles TTS model lifecycle: load/unload, TTL eviction, LRU caching, and warmup.

> 📖 **Quick Summary**: See [CLAUDE.md#architecture-summary](../../../CLAUDE.md#architecture-summary)

## Overview

**Implementation**: `src/tts/model_manager.py` (M4)

**Purpose**: Manage TTS model lifecycle to optimize VRAM usage and loading performance.

**Key Features**:
- Default model preloading at startup
- Dynamic load/unload with reference counting
- TTL-based eviction (idle > 10 min)
- LRU eviction when resident_cap exceeded
- Warmup synthesis (~300ms per model)
- Max parallel loads semaphore

## Configuration

```yaml
# configs/worker.yaml
model_manager:
  default_model_id: "piper-en-us-lessac-medium"
  preload_model_ids: []
  ttl_ms: 600000            # 10 min idle → unload
  min_residency_ms: 120000  # keep at least 2 min
  evict_check_interval_ms: 30000  # Check every 30s
  resident_cap: 3           # Max models in memory
  max_parallel_loads: 1     # Prevent VRAM fragmentation
```

## Usage

```python
from src.tts.model_manager import ModelManager

manager = ModelManager(
    adapter_class=PiperTTSAdapter,
    default_model_id="piper-en-us-lessac-medium",
    ttl_ms=600000
)

# Initialize (loads default + preload models)
await manager.initialize()

# Load model (increments refcount)
adapter = await manager.load("piper-en-us-amy-medium")

# Use adapter
async for frame in adapter.synthesize_stream(text):
    # Process frame
    pass

# Release model (decrements refcount)
await manager.release("piper-en-us-amy-medium")

# Background eviction handles TTL/LRU automatically
```

## Lifecycle

### Startup Behavior

1. Load **default_model_id** (required, must exist)
2. Optionally load **preload_model_ids** list
3. Warmup each model (~300ms synthetic utterance)

### Runtime Operations

- `load(model_id)`: Respects max_parallel_loads semaphore, increments refcount
- `release(model_id)`: Decrements refcount, updates last_used_ts
- `evict_idle()`: Background task unloads models with refcount==0 and idle > ttl_ms
- LRU eviction when resident models exceed resident_cap

### Safety Guarantees

- **No mid-stream unloads**: Reference counting prevents in-use model unload
- **Min residency**: Models kept at least min_residency_ms even if idle
- **Serialized loads**: max_parallel_loads=1 prevents VRAM fragmentation

## Test Coverage

**Total**: 35/35 passing (M4)
- Unit: 20/20 tests (`test_model_manager.py`)
- Integration: 15/15 tests (`test_model_lifecycle.py`)

## Implementation Files

- `src/tts/model_manager.py`: Model lifecycle manager
- `src/tts/worker.py`: gRPC servicer integration
- `configs/worker.yaml`: Configuration

## References

- **Architecture**: [.claude/modules/architecture.md](../architecture.md)
- **Piper Adapter**: [.claude/modules/adapters/piper.md](../adapters/piper.md)
- **Core Documentation**: [CLAUDE.md](../../../CLAUDE.md)

---

**Last Updated**: 2025-10-17
**Status**: Complete (M4)
