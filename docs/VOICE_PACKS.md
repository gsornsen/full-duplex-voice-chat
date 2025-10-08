# Voice Pack Format Specification

This document defines the structure and conventions for TTS model voice packs in the Realtime Duplex Voice Demo system.

---

## Overview

A **voice pack** is a directory containing all assets needed to load and run a TTS model. Voice packs enable:

- **Model modularity:** Hot-swap models without code changes
- **Model discovery:** Adapters auto-discover available models at startup
- **Capability declaration:** Models advertise supported features via metadata
- **Zero-shot cloning:** Store reference audio for voice cloning models (XTTS)

---

## Directory Structure

Voice packs are stored in the `voicepacks/` directory at the repository root.

### Hierarchy

```
voicepacks/
├── {family}/          # Model family (e.g., cosyvoice2, xtts-v2, piper)
│   ├── {model_id}/    # Unique model identifier
│   │   ├── model.safetensors   # Model weights (required)
│   │   ├── config.json         # Model configuration (required)
│   │   ├── metadata.yaml       # Voice pack metadata (required)
│   │   ├── ref/                # Reference audio for cloning (optional)
│   │   │   └── seed.wav
│   │   └── ...                 # Additional model-specific files
│   └── ...
└── ...
```

### Example: CosyVoice 2 English Base

```
voicepacks/
└── cosyvoice2/
    └── en-base/
        ├── model.safetensors       # 1.2 GB
        ├── config.json
        ├── metadata.yaml
        └── tokenizer.json
```

### Example: XTTS-v2 with Voice Cloning

```
voicepacks/
└── xtts-v2/
    └── en-demo/
        ├── model.safetensors       # 800 MB
        ├── config.json
        ├── metadata.yaml
        ├── ref/
        │   ├── seed.wav            # Default reference voice (5-30s)
        │   ├── female-calm.wav     # Additional reference voices
        │   └── male-expressive.wav
        └── vocab.json
```

### Example: Piper (CPU-Only)

```
voicepacks/
└── piper/
    └── en-us-lessac-medium/
        ├── model.onnx              # ONNX format for CPU inference
        ├── config.json
        └── metadata.yaml
```

---

## Required Files

### 1. `model.safetensors` or `model.onnx`

**Format:** SafeTensors (PyTorch) or ONNX (CPU models)

**Purpose:** Model weights

**Loading:**

```python
# PyTorch safetensors
from safetensors.torch import load_file
state_dict = load_file("voicepacks/cosyvoice2/en-base/model.safetensors")
model.load_state_dict(state_dict)

# ONNX
import onnxruntime as ort
session = ort.InferenceSession("voicepacks/piper/en-us-lessac-medium/model.onnx")
```

**Constraints:**
- File size: Typically 200 MB - 3 GB
- Must be compatible with adapter's model architecture
- Use SafeTensors for PyTorch (safer than pickle)

---

### 2. `config.json`

**Format:** JSON

**Purpose:** Model hyperparameters and architecture configuration

**Schema:** Adapter-specific (no universal schema)

**Example (CosyVoice 2):**

```json
{
  "model_type": "cosyvoice2",
  "version": "2.0.1",
  "sample_rate": 22050,
  "hidden_dim": 768,
  "num_layers": 12,
  "num_heads": 12,
  "vocab_size": 50257,
  "languages": ["en", "zh"],
  "max_text_length": 500,
  "use_flash_attention": false
}
```

**Example (XTTS-v2):**

```json
{
  "model_type": "xtts_v2",
  "version": "2.0.3",
  "sample_rate": 24000,
  "gpt_num_layers": 30,
  "gpt_hidden_dim": 1024,
  "vocoder_type": "hifigan",
  "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja"],
  "zero_shot_capable": true,
  "min_reference_audio_duration": 3.0,
  "max_reference_audio_duration": 30.0
}
```

**Example (Piper):**

```json
{
  "model_type": "piper",
  "version": "1.0.0",
  "sample_rate": 22050,
  "phoneme_type": "espeak",
  "language": "en-us",
  "speaker_id": "lessac",
  "quality": "medium",
  "num_speakers": 1
}
```

**Loading:**

```python
import json

with open("voicepacks/cosyvoice2/en-base/config.json") as f:
    config = json.load(f)

model = CosyVoice2Model(
    hidden_dim=config["hidden_dim"],
    num_layers=config["num_layers"],
    # ... other params from config
)
```

---

### 3. `metadata.yaml`

**Format:** YAML

**Purpose:** Voice pack metadata for discovery, routing, and capability advertisement

**Schema:** Universal across all adapters

**Required Fields:**

```yaml
# Voice pack metadata schema v1.0

# Unique identifier (matches directory name: {family}/{model_id})
model_id: "cosyvoice2-en-base"

# Human-readable name
name: "CosyVoice 2 English Base"

# Model family (adapter type)
family: "cosyvoice2"

# Version (semantic versioning)
version: "2.0.1"

# Short description
description: "High-quality English TTS with natural prosody"

# Supported languages (ISO 639-1 codes)
languages:
  - en

# Capabilities (boolean flags)
capabilities:
  streaming: true         # Supports streaming synthesis
  zero_shot: false        # Supports zero-shot voice cloning
  lora: false             # Supports LoRA adapters
  cpu_ok: false           # Can run on CPU (no GPU required)
  emotive_zero_prompt: true  # Supports emotion control without reference audio

# Resource requirements
resources:
  min_vram_gb: 4.0        # Minimum GPU VRAM (GB)
  recommended_vram_gb: 8.0
  model_size_mb: 1200
  warmup_time_ms: 300     # Typical warmup time

# Audio specifications
audio:
  sample_rate: 22050      # Native sample rate (resampled to 48kHz for output)
  output_format: "pcm_s16le"

# Performance characteristics
performance:
  typical_rtf: 0.2        # Real-time factor (0.2 = 5x faster than realtime)
  typical_fal_ms: 250     # First audio latency (ms)

# Tags for filtering and search
tags:
  - english
  - high-quality
  - production-ready
  - low-latency

# License
license: "Apache-2.0"

# Author/source
author: "CosyVoice Team"
source_url: "https://github.com/FunAudioLLM/CosyVoice"

# Creation date
created_at: "2024-09-15"
```

**Full Example (XTTS-v2 with Zero-Shot):**

```yaml
model_id: "xtts-v2-en-demo"
name: "XTTS-v2 English Demo Voice"
family: "xtts"
version: "2.0.3"
description: "Multilingual TTS with zero-shot voice cloning"

languages:
  - en
  - es
  - fr
  - de
  - it
  - pt
  - pl
  - tr
  - ru
  - nl
  - cs
  - ar
  - zh
  - ja

capabilities:
  streaming: true
  zero_shot: true           # Supports voice cloning
  lora: false
  cpu_ok: false
  emotive_zero_prompt: true

resources:
  min_vram_gb: 6.0
  recommended_vram_gb: 10.0
  model_size_mb: 800
  warmup_time_ms: 400

audio:
  sample_rate: 24000
  output_format: "pcm_s16le"

performance:
  typical_rtf: 0.25
  typical_fal_ms: 320

# Reference audio configuration
reference_audio:
  default_reference: "ref/seed.wav"
  min_duration_sec: 3.0
  max_duration_sec: 30.0
  recommended_duration_sec: 10.0

tags:
  - multilingual
  - voice-cloning
  - zero-shot
  - expressive

license: "AGPL-3.0"
author: "Coqui TTS"
source_url: "https://github.com/coqui-ai/TTS"
created_at: "2024-08-10"
```

**Full Example (Piper CPU-Only):**

```yaml
model_id: "piper-en-us-lessac-medium"
name: "Piper English (Lessac Medium)"
family: "piper"
version: "1.0.0"
description: "CPU-friendly English TTS, medium quality"

languages:
  - en

capabilities:
  streaming: true
  zero_shot: false
  lora: false
  cpu_ok: true              # Can run on CPU
  emotive_zero_prompt: false

resources:
  min_vram_gb: 0.0          # No GPU required
  recommended_vram_gb: 0.0
  model_size_mb: 63
  warmup_time_ms: 100

audio:
  sample_rate: 22050
  output_format: "pcm_s16le"

performance:
  typical_rtf: 0.15         # Very fast on CPU
  typical_fal_ms: 450

tags:
  - english
  - cpu
  - fast
  - fallback

license: "MIT"
author: "Piper TTS"
source_url: "https://github.com/rhasspy/piper"
created_at: "2024-07-20"
```

---

## Optional Files

### 4. `ref/` Directory (Zero-Shot Models Only)

**Purpose:** Store reference audio for voice cloning

**File Format:** WAV (16-bit PCM, mono or stereo)

**File Naming:**

- `seed.wav` - Default reference voice (required for zero-shot models)
- `{name}.wav` - Additional named references (optional)

**Example:**

```
ref/
├── seed.wav                # Default voice (10s, 24kHz)
├── female-calm.wav         # Alternative voice
├── male-expressive.wav
└── narrator-deep.wav
```

**Constraints:**

- **Duration:** 3-30 seconds (see `metadata.yaml` for model-specific limits)
- **Sample rate:** Match model's native sample rate (e.g., 24kHz for XTTS)
- **Quality:** Clean, clear speech with minimal background noise
- **Content:** Natural speech (avoid synthetic voices, music, or effects)

**Loading:**

```python
import torchaudio

# Load default reference audio
ref_audio, sr = torchaudio.load("voicepacks/xtts-v2/en-demo/ref/seed.wav")

# Resample if needed
if sr != model.sample_rate:
    ref_audio = torchaudio.functional.resample(ref_audio, sr, model.sample_rate)

# Use for zero-shot synthesis
output = model.generate(
    text="Hello, this is a cloned voice.",
    reference_audio=ref_audio
)
```

### 5. Adapter-Specific Files

Different adapters may require additional files:

**CosyVoice 2:**
- `tokenizer.json` - Text tokenizer
- `phoneme_dict.txt` - Phoneme mappings

**XTTS-v2:**
- `vocab.json` - Token vocabulary
- `speakers.json` - Pre-trained speaker embeddings (optional)

**Piper:**
- `phoneme_map.txt` - eSpeak phoneme mappings

**Unsloth (LoRA):**
- `lora_adapter.safetensors` - LoRA weights
- `adapter_config.json` - LoRA configuration

---

## Model ID Naming Conventions

### Format

```
{family}-{language}-{variant}
```

**Components:**

- **family:** Model family (e.g., `cosyvoice2`, `xtts`, `piper`)
- **language:** Primary language code (ISO 639-1, e.g., `en`, `zh`, `es`)
- **variant:** Model variant (e.g., `base`, `expressive`, `high-quality`, `fast`)

### Examples

**Good:**
- `cosyvoice2-en-base`
- `cosyvoice2-zh-expressive`
- `xtts-v2-en-demo`
- `xtts-v2-multilingual-large`
- `piper-en-us-lessac-medium`
- `piper-de-thorsten-low`
- `sesame-en-lora-formal`

**Avoid:**
- `MyAwesomeVoice` (not descriptive)
- `model_v2_final` (not structured)
- `test123` (not meaningful)

### Language Codes

Use ISO 639-1 two-letter codes:

- `en` - English
- `zh` - Chinese
- `es` - Spanish
- `fr` - French
- `de` - German
- `ja` - Japanese
- `ko` - Korean
- `ar` - Arabic

For locale-specific variants, use `{lang}-{region}`:

- `en-us` - English (US)
- `en-gb` - English (UK)
- `fr-ca` - French (Canada)
- `pt-br` - Portuguese (Brazil)

---

## Model Loading Conventions

### Discovery

Adapters discover available voice packs on startup:

```python
# In adapter __init__.py
from pathlib import Path
import yaml

def discover_voice_packs(family: str) -> List[str]:
    """Discover all voice packs for this adapter family."""
    voicepacks_dir = Path("voicepacks") / family
    if not voicepacks_dir.exists():
        return []

    model_ids = []
    for model_dir in voicepacks_dir.iterdir():
        if not model_dir.is_dir():
            continue

        metadata_file = model_dir / "metadata.yaml"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = yaml.safe_load(f)
                model_ids.append(metadata["model_id"])

    return model_ids

# Usage
available_models = discover_voice_packs("cosyvoice2")
print(f"Available CosyVoice 2 models: {available_models}")
# Output: ['cosyvoice2-en-base', 'cosyvoice2-zh-expressive']
```

### Loading

Models are loaded on-demand by the Model Manager:

```python
# In src/tts/model_manager.py
async def load_model(self, model_id: str):
    """Load model from voice pack directory."""
    # Parse model_id to extract family
    family = model_id.split('-')[0]  # e.g., "cosyvoice2" from "cosyvoice2-en-base"

    # Determine voice pack directory
    # Try: voicepacks/{family}/{model_id}/
    # Or:  voicepacks/{family}/{variant}/  where variant is after language code

    # Example: cosyvoice2-en-base
    # Tries: voicepacks/cosyvoice2/cosyvoice2-en-base/
    # Then:  voicepacks/cosyvoice2/en-base/

    voicepack_dir = Path("voicepacks") / family / model_id
    if not voicepack_dir.exists():
        # Try shortened path
        variant = '-'.join(model_id.split('-')[1:])  # "en-base"
        voicepack_dir = Path("voicepacks") / family / variant

    if not voicepack_dir.exists():
        raise ModelNotFoundError(f"Voice pack not found: {model_id}")

    # Load metadata
    with open(voicepack_dir / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)

    # Load config
    with open(voicepack_dir / "config.json") as f:
        config = json.load(f)

    # Load model weights
    model_path = voicepack_dir / "model.safetensors"
    if not model_path.exists():
        model_path = voicepack_dir / "model.onnx"  # Try ONNX

    # Adapter-specific loading logic
    adapter = self.adapter_factory.create(family)
    model = adapter.load_model(model_path, config)

    # Store in cache
    self.loaded_models[model_id] = {
        "model": model,
        "metadata": metadata,
        "config": config,
        "voicepack_dir": voicepack_dir,
    }

    logger.info(f"Loaded model {model_id} from {voicepack_dir}")
```

### Validation

Validate voice pack structure before loading:

```python
def validate_voice_pack(voicepack_dir: Path) -> None:
    """Validate voice pack has required files."""
    required_files = ["metadata.yaml", "config.json"]

    # Check for model weights
    has_weights = (
        (voicepack_dir / "model.safetensors").exists()
        or (voicepack_dir / "model.onnx").exists()
        or (voicepack_dir / "model.pt").exists()
    )

    if not has_weights:
        raise ValueError(f"No model weights found in {voicepack_dir}")

    for filename in required_files:
        filepath = voicepack_dir / filename
        if not filepath.exists():
            raise ValueError(f"Missing required file: {filepath}")

    # Validate metadata schema
    with open(voicepack_dir / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)

    required_metadata_keys = ["model_id", "name", "family", "version", "languages", "capabilities"]
    for key in required_metadata_keys:
        if key not in metadata:
            raise ValueError(f"Missing required metadata key: {key}")

    logger.info(f"Voice pack validated: {voicepack_dir}")
```

---

## Creating a New Voice Pack

### Step 1: Organize Files

```bash
# Create voice pack directory
mkdir -p voicepacks/cosyvoice2/en-custom

# Copy model weights
cp /path/to/model.safetensors voicepacks/cosyvoice2/en-custom/

# Copy config
cp /path/to/config.json voicepacks/cosyvoice2/en-custom/
```

### Step 2: Create metadata.yaml

```bash
cd voicepacks/cosyvoice2/en-custom
cat > metadata.yaml <<EOF
model_id: "cosyvoice2-en-custom"
name: "CosyVoice 2 English Custom"
family: "cosyvoice2"
version: "2.0.1"
description: "Custom-trained English TTS model"

languages:
  - en

capabilities:
  streaming: true
  zero_shot: false
  lora: false
  cpu_ok: false
  emotive_zero_prompt: true

resources:
  min_vram_gb: 4.0
  recommended_vram_gb: 8.0
  model_size_mb: 1250
  warmup_time_ms: 300

audio:
  sample_rate: 22050
  output_format: "pcm_s16le"

performance:
  typical_rtf: 0.2
  typical_fal_ms: 260

tags:
  - english
  - custom
  - experimental

license: "Apache-2.0"
author: "Your Name"
source_url: "https://example.com/model"
created_at: "2025-10-05"
EOF
```

### Step 3: Validate

```bash
# Run validation script (M11+ feature)
python scripts/validate-voice-pack.py voicepacks/cosyvoice2/en-custom

# Expected output:
# ✓ Voice pack structure valid
# ✓ metadata.yaml schema valid
# ✓ Model weights present (1250 MB)
# ✓ Config valid
# Voice pack ready: cosyvoice2-en-custom
```

### Step 4: Test Loading

```bash
# Start worker with new model as default
just run-tts-cosy DEFAULT="cosyvoice2-en-custom"

# Check logs for successful loading
grep "Model loaded" /var/log/tts-worker.log
# Expected: "Model loaded: cosyvoice2-en-custom in 2.8s"
```

### Step 5: Register with Orchestrator (M9+)

```yaml
# In worker.yaml
model_manager:
  default_model_id: "cosyvoice2-en-custom"
  preload_model_ids:
    - "cosyvoice2-en-custom"
```

---

## Voice Pack Best Practices

### Storage

**Location:**
- Development: `voicepacks/` in repository root
- Production: Separate volume (e.g., `/data/voicepacks`) mounted to containers

**Organization:**
- Group by family: `voicepacks/{family}/{model_id}/`
- Keep families separate for adapter modularity
- Use symlinks for shared dependencies (e.g., tokenizers)

### Versioning

**Semantic Versioning:**
- `1.0.0` - Initial release
- `1.1.0` - New features (e.g., added languages)
- `1.0.1` - Bug fixes (e.g., fixed phoneme mappings)
- `2.0.0` - Breaking changes (e.g., incompatible architecture)

**Migration:**
- Keep old versions available during transition
- Use `model_id` suffix: `cosyvoice2-en-base-v1`, `cosyvoice2-en-base-v2`
- Document migration in `CHANGELOG.md`

### Licensing

**Respect Model Licenses:**
- Check original model license before redistribution
- Include `LICENSE` file in voice pack directory
- Document attribution in `metadata.yaml`

**Common Licenses:**
- Apache-2.0: Permissive, commercial use allowed
- MIT: Permissive, minimal restrictions
- AGPL-3.0: Open-source, network use triggers copyleft
- CC-BY-NC: Non-commercial use only

---

## Voice Pack Registry (Future Feature)

**M13+ Feature:** Central registry for community-contributed voice packs.

**Concept:**

```bash
# Install voice pack from registry
just install-voice-pack cosyvoice2/en-expressive

# List available voice packs
just list-voice-packs --family cosyvoice2

# Update all voice packs
just update-voice-packs
```

**Registry Format:**

```yaml
# In voicepack-registry.yaml
registry:
  - model_id: "cosyvoice2-en-expressive"
    family: "cosyvoice2"
    download_url: "https://example.com/voicepacks/cosyvoice2-en-expressive-v2.0.1.tar.gz"
    checksum_sha256: "abc123..."
    size_mb: 1300
    version: "2.0.1"
    tags: ["english", "expressive", "official"]

  - model_id: "xtts-v2-multilingual-large"
    family: "xtts"
    download_url: "https://example.com/voicepacks/xtts-v2-multilingual-large-v2.0.3.tar.gz"
    checksum_sha256: "def456..."
    size_mb: 2500
    version: "2.0.3"
    tags: ["multilingual", "large", "high-quality"]
```

---

## Troubleshooting

### Issue: Model Not Found

**Error:** `ModelNotFoundError: Voice pack not found: cosyvoice2-en-base`

**Diagnosis:**

```bash
# Check directory exists
ls -la voicepacks/cosyvoice2/

# Expected: directory "en-base" present
```

**Solution:**
- Verify voice pack directory matches model_id (see Naming Conventions)
- Check `metadata.yaml` has correct `model_id` field

### Issue: Invalid Metadata Schema

**Error:** `ValueError: Missing required metadata key: capabilities`

**Diagnosis:**

```bash
# Validate metadata
python -c "import yaml; print(yaml.safe_load(open('voicepacks/cosyvoice2/en-base/metadata.yaml')))"
```

**Solution:**
- Add missing fields to `metadata.yaml` (see Required Fields)
- Use validation script: `python scripts/validate-voice-pack.py ...`

### Issue: Model Loading Fails

**Error:** `RuntimeError: Error loading model weights`

**Diagnosis:**

```bash
# Check model file integrity
sha256sum voicepacks/cosyvoice2/en-base/model.safetensors

# Compare with known good checksum
```

**Solution:**
- Re-download model weights
- Check for disk corruption: `fsck` or `chkdsk`
- Verify sufficient disk space and VRAM

---

## Further Reading

- **Configuration Reference:** `/home/gerald/git/full-duplex-voice-chat/docs/CONFIGURATION_REFERENCE.md`
- **Multi-GPU Deployment:** `/home/gerald/git/full-duplex-voice-chat/docs/MULTI_GPU.md`
- **Model Manager Design:** `/home/gerald/git/full-duplex-voice-chat/project_documentation/TDD.md#model-manager`
- **Adapter Implementation:** `/home/gerald/git/full-duplex-voice-chat/src/tts/README.md`
- **SafeTensors Format:** https://github.com/huggingface/safetensors
- **ONNX Format:** https://onnx.ai/

---

**Last Updated:** 2025-10-05
**Target Milestone:** M4+ (Model Manager), M6+ (Adapter implementations)
