# UV Workspace Quick Start

**TL;DR**: Use uv workspace to solve PyTorch 2.7.0 vs 2.3.1 conflict between orchestrator and CosyVoice.

## Installation

### Install Shared Packages Only

```bash
cd /home/gerald/git/full-duplex-voice-chat
uv sync
```

### Install Orchestrator (PyTorch 2.7.0)

```bash
cd packages/orchestrator
uv pip install -e .
# OR
cd /home/gerald/git/full-duplex-voice-chat && uv pip install -e packages/orchestrator
```

### Install Piper (CPU, no PyTorch)

```bash
cd packages/tts-piper
uv pip install -e .
# OR
cd /home/gerald/git/full-duplex-voice-chat && uv pip install -e packages/tts-piper
```

### Install CosyVoice (PyTorch 2.3.1)

```bash
cd packages/tts-cosyvoice
uv pip install -e .
# OR
cd /home/gerald/git/full-duplex-voice-chat && uv pip install -e packages/tts-cosyvoice
```

## Running Services

### Orchestrator

```bash
uv run python -m orchestrator.server
```

### TTS Worker (Piper)

```bash
uv run python -m tts.worker --adapter piper
```

### TTS Worker (CosyVoice)

```bash
uv run python -m tts.worker --adapter cosyvoice
```

## Import Paths

### Old (src/)

```python
from src.rpc.generated import tts_pb2
from src.common.types import WorkerCapabilities
from src.tts.tts_base import TTSAdapter
from src.orchestrator.server import OrchestrationServer
```

### New (packages/)

```python
from rpc.generated import tts_pb2  # proto package → rpc module
from shared.types import WorkerCapabilities  # common package → shared module
from tts.tts_base import TTSAdapter  # tts-base package → tts module
from orchestrator.server import OrchestrationServer  # orchestrator package → orchestrator module
```

## Testing

### Shared Packages

```bash
uv run pytest tests/unit/test_m1_proto.py
```

### Orchestrator

```bash
cd packages/orchestrator
uv run pytest ../../tests/unit/orchestrator/
```

### TTS Adapters

```bash
cd packages/tts-piper
uv run pytest ../../tests/unit/tts/adapters/test_adapter_piper.py
```

## Troubleshooting

### "Module not found" error

**Solution**: Install the package in editable mode:
```bash
uv pip install -e packages/proto -e packages/common -e packages/tts-base
```

### "Conflicting dependencies" error

**Expected**: This means you're trying to install both orchestrator (PyTorch 2.7.0) and cosyvoice (PyTorch 2.3.1) in the same environment.

**Solution**: Install each service in its own virtual environment:
```bash
# Orchestrator env
cd packages/orchestrator && uv venv && uv sync

# CosyVoice env
cd packages/tts-cosyvoice && uv venv && uv sync
```

## File Locations

- **Workspace root**: `/home/gerald/git/full-duplex-voice-chat`
- **Shared packages**: `packages/proto`, `packages/common`, `packages/tts-base`
- **Service packages**: `packages/orchestrator`, `packages/tts-piper`, `packages/tts-cosyvoice`
- **Legacy code**: `src/` (kept for backward compatibility)
- **Documentation**: `docs/UV_WORKSPACE_MIGRATION_GUIDE.md`

## Quick Commands

```bash
# Sync workspace
uv sync

# Lock dependencies
uv lock

# Install package in editable mode
uv pip install -e packages/orchestrator

# Run tests
uv run pytest tests/

# Check installed packages
uv pip list | grep -E "(proto|common|tts|orchestrator)"
```

## Status

- ✅ Workspace structure created
- ✅ Shared packages installable
- ✅ Imports work (for workspace packages)
- ⏸ Service isolation (not tested yet)
- ⏸ Import migration (not started)
- ⏸ Docker integration (not started)

**Next**: Phase 2 - Import Migration (update code to use new import paths)

---

**Full Guide**: [UV_WORKSPACE_MIGRATION_GUIDE.md](./UV_WORKSPACE_MIGRATION_GUIDE.md)
**Summary**: [UV_WORKSPACE_SUMMARY.md](./UV_WORKSPACE_SUMMARY.md)
