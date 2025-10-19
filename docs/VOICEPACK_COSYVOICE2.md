# CosyVoice 2 Voicepack Structure

**Version**: 1.0
**Last Updated**: 2025-10-17
**Status**: M6 Implementation Reference

This document defines the directory structure, metadata format, and setup procedures for CosyVoice 2 voicepacks in the Realtime Duplex Voice Demo system.

---

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Required Files](#required-files)
- [Metadata Specification](#metadata-specification)
- [Configuration Format](#configuration-format)
- [Model Download and Setup](#model-download-and-setup)
- [Model Manager Integration](#model-manager-integration)
- [Comparison with Piper Voicepacks](#comparison-with-piper-voicepacks)
- [References](#references)

---

## Overview

CosyVoice 2 is a GPU-accelerated neural TTS model supporting:

- **Zero-shot voice cloning**: 3-10s reference audio
- **Streaming synthesis**: Incremental text processing
- **Multi-speaker support**: Voice embeddings and reference audio
- **High-quality output**: 24kHz native sample rate (upsampled to 48kHz)

**Key Differences from Piper**:

| Feature | Piper | CosyVoice 2 |
|---------|-------|-------------|
| Hardware | CPU (ONNX Runtime) | GPU (CUDA 12.1, PyTorch 2.3.1) |
| Model Format | `.onnx` + `.onnx.json` | `.pt` (PyTorch checkpoint) |
| Sample Rate | 22050 Hz (varies) | 24000 Hz (fixed) |
| Voice Cloning | No | Yes (zero-shot with reference audio) |
| Docker Isolation | Optional | **Required** (PyTorch version conflict) |

---

## Directory Structure

### Standard Voicepack Layout

```
voicepacks/cosyvoice/
├── en-base/                      # Model ID directory
│   ├── model.pt                  # PyTorch checkpoint (REQUIRED)
│   ├── config.json               # Model configuration (REQUIRED)
│   ├── metadata.yaml             # Voicepack metadata (REQUIRED)
│   ├── speakers.json             # Speaker embeddings (OPTIONAL)
│   ├── reference_audio/          # Reference audio for voice cloning (OPTIONAL)
│   │   ├── speaker1_ref.wav      # 3-10s reference audio samples
│   │   ├── speaker1_ref.txt      # Corresponding transcript
│   │   ├── speaker2_ref.wav
│   │   └── speaker2_ref.txt
│   └── README.md                 # Model-specific documentation (OPTIONAL)
│
├── zh-instruct/                  # Additional language/variant
│   ├── model.pt
│   ├── config.json
│   ├── metadata.yaml
│   └── ...
│
└── custom-voice/                 # User-trained model
    ├── model.pt
    ├── config.json
    ├── metadata.yaml
    ├── reference_audio/
    └── ...
```

### ModelScope Download Structure

When downloading from ModelScope, the structure may differ slightly:

```
iic_CosyVoice2-0.5B/              # ModelScope repository
├── cosyvoice.yaml                # Original config (convert to config.json)
├── llm.pt                        # Language model checkpoint
├── flow.pt                       # Flow model checkpoint
├── speech_tokenizer_v2.yaml      # Tokenizer config
└── ...
```

**Conversion Required**: See [Model Download and Setup](#model-download-and-setup).

---

## Required Files

### 1. model.pt (REQUIRED)

**Description**: PyTorch checkpoint containing CosyVoice model weights.

**Format**: PyTorch serialized format (`.pt` or `.pth`)

**Size**: ~1-2 GB (varies by model variant)

**Location**: `voicepacks/cosyvoice/{model_id}/model.pt`

**Notes**:
- For multi-file models (e.g., `llm.pt` + `flow.pt`), consolidate or create a directory structure
- Ensure checkpoint is compatible with PyTorch 2.3.1
- Verify CUDA compute capability matches GPU (e.g., sm_89 for RTX 4090)

---

### 2. config.json (REQUIRED)

**Description**: Model configuration specifying audio parameters and architecture.

**Format**: JSON

**Location**: `voicepacks/cosyvoice/{model_id}/config.json`

**Schema**:

```json
{
  "audio": {
    "sample_rate": 24000,           // Fixed for CosyVoice 2
    "n_fft": 1024,                  // FFT size for mel-spectrogram
    "hop_length": 256,              // Hop length (samples between frames)
    "win_length": 1024,             // Window length for STFT
    "n_mels": 80,                   // Number of mel filterbank channels
    "fmin": 0,                      // Minimum frequency (Hz)
    "fmax": 12000                   // Maximum frequency (Hz)
  },
  "model": {
    "hidden_channels": 256,         // Encoder hidden dimension
    "inter_channels": 192,          // Intermediate projection dimension
    "filter_channels": 768,         // Filter dimension for convolutions
    "n_heads": 2,                   // Number of attention heads
    "n_layers": 6,                  // Number of transformer layers
    "kernel_size": 3,               // Convolution kernel size
    "p_dropout": 0.1                // Dropout probability
  },
  "inference": {
    "max_tokens": 2048,             // Maximum token sequence length
    "temperature": 1.0,             // Sampling temperature
    "speed": 1.0,                   // Speech speed multiplier
    "streaming_chunk_size": 512     // Chunk size for streaming (tokens)
  }
}
```

**Conversion from YAML**:

If the model ships with `cosyvoice.yaml`, convert to JSON:

```bash
python -c "
import yaml
import json
with open('cosyvoice.yaml') as f:
    config = yaml.safe_load(f)
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
```

---

### 3. metadata.yaml (REQUIRED)

**Description**: Voicepack metadata for Model Manager registration and discovery.

**Format**: YAML

**Location**: `voicepacks/cosyvoice/{model_id}/metadata.yaml`

**Schema**:

```yaml
model_id: "cosyvoice2-en-base"    # Unique identifier (must match directory name)
family: "cosyvoice2"               # Model family for adapter routing
version: "2.0.5"                   # Model version
language: "en"                     # ISO 639-1 language code
sample_rate: 24000                 # Native sample rate (Hz)

# Capabilities
capabilities:
  - zero_shot                      # Supports zero-shot voice cloning
  - multi_speaker                  # Supports multiple speakers
  - streaming                      # Supports streaming synthesis
  - emotion_control                # Supports explicit emotion controls (optional)

# Hardware Requirements
gpu_required: true                 # Requires GPU (CUDA)
vram_mb: 1500                      # Estimated VRAM usage (MB)
cuda_compute_capability: 8.9       # Minimum CUDA compute capability (optional)

# Speakers (for multi-speaker models)
speakers:
  - default                        # Default speaker ID

# Model Provenance
license: "Apache-2.0"              # License (Apache-2.0, MIT, etc.)
source_url: "https://github.com/FunAudioLLM/CosyVoice"
model_url: "https://modelscope.cn/models/iic/CosyVoice2-0.5B"
citation: "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models"

# Description
description: "CosyVoice 2.0 English base model for zero-shot TTS"

# Tags for discovery and filtering
tags:
  - gpu_required
  - streaming
  - neural
  - zero_shot
  - transformer
```

**Field Descriptions**:

- `model_id`: **REQUIRED**. Unique identifier matching directory name. Used by Model Manager for registration.
- `family`: **REQUIRED**. Model family (`cosyvoice2`) for adapter routing.
- `version`: Model version string (semver recommended).
- `language`: ISO 639-1 language code (e.g., `en`, `zh`, `ja`).
- `sample_rate`: Native audio sample rate (Hz). Fixed at 24000 for CosyVoice 2.
- `capabilities`: List of supported features (see schema above).
- `gpu_required`: Boolean indicating GPU requirement.
- `vram_mb`: Estimated VRAM usage in MB (for capacity planning).
- `speakers`: List of speaker IDs for multi-speaker models.
- `license`: SPDX license identifier.
- `source_url`: Official repository URL.
- `model_url`: Model download URL (ModelScope, HuggingFace, etc.).
- `description`: Human-readable description.
- `tags`: List of tags for discovery and filtering.

---

### 4. speakers.json (OPTIONAL)

**Description**: Speaker embeddings for multi-speaker models or pre-computed reference audio embeddings.

**Format**: JSON

**Location**: `voicepacks/cosyvoice/{model_id}/speakers.json`

**Schema**:

```json
{
  "speakers": [
    {
      "id": "default",
      "name": "Default Voice",
      "language": "en",
      "description": "Neutral English voice",
      "embedding": null,           // Optional: pre-computed embedding vector
      "reference_audio": null      // Optional: path to reference audio file
    },
    {
      "id": "speaker1",
      "name": "Speaker 1",
      "language": "en",
      "description": "Custom cloned voice",
      "embedding": [0.123, -0.456, ...],  // 256-dim vector (example)
      "reference_audio": "reference_audio/speaker1_ref.wav"
    }
  ],
  "embedding_dim": 256,            // Dimension of speaker embeddings
  "default_speaker": "default"     // Default speaker ID
}
```

**Notes**:
- Not required for zero-shot models using runtime reference audio
- Useful for pre-registering speaker voices to reduce inference latency
- Embeddings are model-specific (not transferable between models)

---

### 5. reference_audio/ (OPTIONAL)

**Description**: Reference audio files for zero-shot voice cloning.

**Format**: WAV files (16kHz or 24kHz, mono, 16-bit PCM)

**Location**: `voicepacks/cosyvoice/{model_id}/reference_audio/`

**Guidelines**:
- **Duration**: 3-10 seconds (optimal: 5-7 seconds)
- **Quality**: Clean speech, minimal background noise
- **Content**: Natural speech with varied prosody
- **Transcript**: Provide corresponding `.txt` file with exact transcript

**Example**:

```
reference_audio/
├── speaker1_ref.wav         # 5.2s, "Hello, this is a reference audio sample."
├── speaker1_ref.txt         # Transcript: "Hello, this is a reference audio sample."
├── speaker2_ref.wav         # 6.8s, "My name is Sarah, and I'm happy to help you."
└── speaker2_ref.txt         # Transcript: "My name is Sarah, and I'm happy to help you."
```

**Usage**:
- Runtime voice cloning: Provide reference audio + transcript via API
- Pre-computed embeddings: Extract embeddings offline, store in `speakers.json`

---

## Metadata Specification

### Complete Example: metadata.yaml

```yaml
model_id: "cosyvoice2-en-base"
family: "cosyvoice2"
version: "2.0.5"
language: "en"
sample_rate: 24000

# Capabilities
capabilities:
  - zero_shot
  - multi_speaker
  - streaming

# Hardware Requirements
gpu_required: true
vram_mb: 1500
cuda_compute_capability: 8.9

# Speakers
speakers:
  - default

# Model Provenance
license: "Apache-2.0"
source_url: "https://github.com/FunAudioLLM/CosyVoice"
model_url: "https://modelscope.cn/models/iic/CosyVoice2-0.5B"
citation: "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models"

# Description
description: "CosyVoice 2.0 English base model for zero-shot TTS"

# Tags
tags:
  - gpu_required
  - streaming
  - neural
  - zero_shot
  - transformer
```

### Validation Rules

**Model Manager validates**:
1. `model_id` matches directory name
2. `family` is `cosyvoice2`
3. `sample_rate` is 24000
4. `gpu_required` is `true`
5. Required files exist: `model.pt`, `config.json`

**Warnings (non-fatal)**:
- Missing `speakers.json` for multi-speaker models
- Missing `reference_audio/` for zero-shot models
- Large VRAM usage (`vram_mb > 2000`)

---

## Configuration Format

### Complete Example: config.json

```json
{
  "audio": {
    "sample_rate": 24000,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mels": 80,
    "fmin": 0,
    "fmax": 12000
  },
  "model": {
    "hidden_channels": 256,
    "inter_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1
  },
  "inference": {
    "max_tokens": 2048,
    "temperature": 1.0,
    "speed": 1.0,
    "streaming_chunk_size": 512
  }
}
```

### Adapter Usage

The CosyVoiceAdapter reads `config.json` to:
- Verify native sample rate (24000 Hz)
- Configure mel-spectrogram parameters
- Set inference hyperparameters (temperature, speed)

**Code Example**:

```python
from pathlib import Path
import json

model_path = Path("voicepacks/cosyvoice/en-base")
config_path = model_path / "config.json"

with open(config_path) as f:
    config = json.load(f)

native_sample_rate = config["audio"]["sample_rate"]  # 24000
assert native_sample_rate == 24000, "CosyVoice 2 requires 24kHz"
```

---

## Model Download and Setup

### Automated Setup (Recommended)

Use the provided setup script:

```bash
./scripts/setup_cosyvoice_voicepack.sh en-base
```

**What it does**:
1. Downloads model from ModelScope (requires `modelscope` Python package)
2. Organizes files into voicepack structure
3. Converts YAML config to JSON
4. Creates `metadata.yaml` with sensible defaults
5. Validates voicepack integrity

**Prerequisites**:
```bash
pip install modelscope
```

**Script Usage**:
```bash
# Download default model (en-base)
./scripts/setup_cosyvoice_voicepack.sh

# Download specific model variant
./scripts/setup_cosyvoice_voicepack.sh zh-instruct

# Download to custom location
VOICEPACK_ROOT=/path/to/voicepacks ./scripts/setup_cosyvoice_voicepack.sh en-base
```

### Manual Setup

#### Step 1: Download Model from ModelScope

```python
from modelscope import snapshot_download

model_dir = snapshot_download(
    'iic/CosyVoice2-0.5B',
    local_dir='./downloads/cosyvoice2-en-base',
    revision='v2.0.5'  # Optional: specific version
)
```

**Alternative: Git LFS**:

```bash
git lfs install
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git downloads/cosyvoice2-en-base
```

#### Step 2: Organize Voicepack Structure

```bash
# Create voicepack directory
mkdir -p voicepacks/cosyvoice/en-base

# Copy model checkpoint
cp downloads/cosyvoice2-en-base/llm.pt voicepacks/cosyvoice/en-base/model.pt

# Convert YAML config to JSON
python -c "
import yaml
import json
with open('downloads/cosyvoice2-en-base/cosyvoice.yaml') as f:
    config = yaml.safe_load(f)
with open('voicepacks/cosyvoice/en-base/config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
```

#### Step 3: Create metadata.yaml

```bash
cat > voicepacks/cosyvoice/en-base/metadata.yaml <<EOF
model_id: "cosyvoice2-en-base"
family: "cosyvoice2"
version: "2.0.5"
language: "en"
sample_rate: 24000
capabilities:
  - zero_shot
  - multi_speaker
  - streaming
gpu_required: true
vram_mb: 1500
speakers:
  - default
license: "Apache-2.0"
source_url: "https://github.com/FunAudioLLM/CosyVoice"
model_url: "https://modelscope.cn/models/iic/CosyVoice2-0.5B"
description: "CosyVoice 2.0 English base model for zero-shot TTS"
tags:
  - gpu_required
  - streaming
  - neural
  - zero_shot
  - transformer
EOF
```

#### Step 4: Validate Voicepack

```bash
# Check directory structure
tree voicepacks/cosyvoice/en-base/

# Expected output:
# voicepacks/cosyvoice/en-base/
# ├── config.json
# ├── metadata.yaml
# └── model.pt
```

---

## Model Manager Integration

### Registration

The Model Manager automatically discovers voicepacks at startup:

```python
# src/tts/model_manager.py
from pathlib import Path
import yaml

def discover_voicepacks(voicepack_root: Path) -> dict[str, dict]:
    """Discover all voicepacks in the root directory."""
    voicepacks = {}

    for family_dir in voicepack_root.iterdir():
        if not family_dir.is_dir():
            continue

        for model_dir in family_dir.iterdir():
            if not model_dir.is_dir():
                continue

            metadata_path = model_dir / "metadata.yaml"
            if not metadata_path.exists():
                logger.warning(f"Missing metadata.yaml: {model_dir}")
                continue

            with open(metadata_path) as f:
                metadata = yaml.safe_load(f)

            model_id = metadata["model_id"]
            voicepacks[model_id] = {
                "path": model_dir,
                "metadata": metadata,
            }

    return voicepacks
```

### Runtime Loading

```python
# Load model via Model Manager
model_id = "cosyvoice2-en-base"
adapter = await model_manager.load_model(model_id)

# Synthesize audio
async def text_chunks():
    yield "Hello, world!"

async for frame in adapter.synthesize_stream(text_chunks()):
    # Process 20ms audio frame at 48kHz
    pass
```

### Configuration (worker config.yaml)

```yaml
worker:
  device: "cuda:0"
  sample_rate: 48000
  frame_ms: 20

model_manager:
  voicepack_root: "voicepacks"
  default_model_id: "cosyvoice2-en-base"
  preload_model_ids: []
  ttl_ms: 600000                 # 10 minutes idle → unload
  min_residency_ms: 120000       # keep at least 2 minutes
  evict_check_interval_ms: 30000 # every 30s
  resident_cap: 3                # at most 3 models resident
  max_parallel_loads: 1
```

### Docker Deployment

**⚠️ IMPORTANT**: CosyVoice 2 requires isolated Docker container due to PyTorch version conflict (2.3.1 vs 2.7.0).

**docker-compose.yml**:

```yaml
services:
  tts-cosyvoice:
    build:
      context: .
      dockerfile: Dockerfile.tts-cosyvoice
    volumes:
      - ./voicepacks:/app/voicepacks:ro
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_ID=cosyvoice2-en-base
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Dockerfile.tts-cosyvoice**:

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install Python 3.10 (CosyVoice tested version)
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install PyTorch 2.3.1 + CUDA 12.1
RUN pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install CosyVoice
RUN git clone https://github.com/FunAudioLLM/CosyVoice.git /opt/cosyvoice
WORKDIR /opt/cosyvoice
RUN pip3 install -r requirements.txt

# Copy project files (minimal dependencies)
COPY src/tts /app/src/tts
COPY src/rpc /app/src/rpc
COPY pyproject.toml /app/

# Install project dependencies (gRPC, Redis, audio only)
WORKDIR /app
RUN pip3 install grpcio grpcio-tools redis numpy scipy librosa

CMD ["python3", "-m", "src.tts.worker", "--default_model", "cosyvoice2-en-base"]
```

---

## Comparison with Piper Voicepacks

### Structural Differences

| Component | Piper | CosyVoice 2 |
|-----------|-------|-------------|
| **Model File** | `{name}.onnx` (60-200 MB) | `model.pt` (1-2 GB) |
| **Config File** | `{name}.onnx.json` | `config.json` |
| **Metadata File** | `metadata.yaml` | `metadata.yaml` |
| **Speaker Data** | Embedded in ONNX | `speakers.json` (optional) |
| **Reference Audio** | N/A | `reference_audio/` (optional) |

### Metadata Differences

| Field | Piper | CosyVoice 2 |
|-------|-------|-------------|
| `family` | `piper` | `cosyvoice2` |
| `sample_rate` | 22050 (varies) | 24000 (fixed) |
| `gpu_required` | `false` | `true` |
| `vram_mb` | N/A | 1500 (typical) |
| `capabilities` | `[streaming, neural]` | `[zero_shot, multi_speaker, streaming]` |

### Adapter Selection

The Model Manager routes to the correct adapter based on `family`:

```python
def get_adapter_class(family: str) -> type:
    """Get adapter class for model family."""
    if family == "piper":
        from src.tts.adapters.adapter_piper import PiperTTSAdapter
        return PiperTTSAdapter
    elif family == "cosyvoice2":
        from src.tts.adapters.adapter_cosyvoice import CosyVoiceAdapter
        return CosyVoiceAdapter
    else:
        raise ValueError(f"Unknown model family: {family}")
```

### Migration Path

Existing Piper voicepacks remain compatible. CosyVoice 2 voicepacks coexist in parallel:

```
voicepacks/
├── piper/
│   └── en-us-lessac-medium/
│       ├── en_US-lessac-medium.onnx
│       ├── en_US-lessac-medium.onnx.json
│       └── metadata.yaml
│
└── cosyvoice/
    └── en-base/
        ├── model.pt
        ├── config.json
        └── metadata.yaml
```

---

## References

### Official Documentation

- **CosyVoice GitHub**: https://github.com/FunAudioLLM/CosyVoice
- **ModelScope**: https://modelscope.cn/models/iic/CosyVoice2-0.5B
- **Paper**: "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models"

### Project Documentation

- **CLAUDE.md**: [/home/gerald/git/full-duplex-voice-chat/CLAUDE.md](/home/gerald/git/full-duplex-voice-chat/CLAUDE.md)
- **TDD.md**: [/home/gerald/git/full-duplex-voice-chat/project_documentation/TDD.md](/home/gerald/git/full-duplex-voice-chat/project_documentation/TDD.md)
- **CosyVoice PyTorch Conflict**: [/home/gerald/git/full-duplex-voice-chat/docs/COSYVOICE_PYTORCH_CONFLICT.md](/home/gerald/git/full-duplex-voice-chat/docs/COSYVOICE_PYTORCH_CONFLICT.md)

### Related Adapters

- **Piper Adapter**: [/home/gerald/git/full-duplex-voice-chat/src/tts/adapters/adapter_piper.py](/home/gerald/git/full-duplex-voice-chat/src/tts/adapters/adapter_piper.py)
- **CosyVoice Adapter**: [/home/gerald/git/full-duplex-voice-chat/src/tts/adapters/adapter_cosyvoice.py](/home/gerald/git/full-duplex-voice-chat/src/tts/adapters/adapter_cosyvoice.py)

---

## Appendix: Example Voicepack

### Complete Directory Listing

```
voicepacks/cosyvoice/en-base/
├── config.json                   # Model configuration (JSON)
├── metadata.yaml                 # Voicepack metadata (YAML)
├── model.pt                      # PyTorch checkpoint (1.2 GB)
├── speakers.json                 # Speaker embeddings (optional)
├── reference_audio/              # Reference audio samples
│   ├── speaker1_ref.wav          # 5.2s, 24kHz, mono
│   ├── speaker1_ref.txt          # "Hello, this is a reference audio sample."
│   ├── speaker2_ref.wav          # 6.8s, 24kHz, mono
│   └── speaker2_ref.txt          # "My name is Sarah, and I'm happy to help you."
└── README.md                     # Model-specific documentation
```

### Total Size: ~1.5 GB

---

**Next Steps**:
1. Run setup script: `./scripts/setup_cosyvoice_voicepack.sh en-base`
2. Validate voicepack: `python -m src.tts.model_manager --validate voicepacks/cosyvoice/en-base`
3. Start TTS worker: `docker compose up tts-cosyvoice`
4. Test synthesis: `just test-integration -k cosyvoice`

---

**Feedback**: Report issues or suggestions to the project team via GitHub Issues.
