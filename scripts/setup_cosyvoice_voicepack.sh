#!/usr/bin/env bash
#
# Setup CosyVoice 2 Voicepack
#
# This script downloads CosyVoice 2 models from ModelScope and organizes them
# into the standard voicepack structure for the Realtime Duplex Voice Demo.
#
# Usage:
#   ./scripts/setup_cosyvoice_voicepack.sh [model_variant] [options]
#
# Examples:
#   ./scripts/setup_cosyvoice_voicepack.sh                    # Download default (en-base)
#   ./scripts/setup_cosyvoice_voicepack.sh en-base            # Download en-base variant
#   ./scripts/setup_cosyvoice_voicepack.sh zh-instruct        # Download zh-instruct variant
#   VOICEPACK_ROOT=/custom/path ./scripts/setup_cosyvoice_voicepack.sh
#
# Environment Variables:
#   VOICEPACK_ROOT    Root directory for voicepacks (default: ./voicepacks)
#   DOWNLOAD_DIR      Temporary download directory (default: ./downloads)
#   SKIP_DOWNLOAD     Skip download step (use existing download) [0|1]
#   SKIP_VALIDATION   Skip validation step [0|1]
#
# Prerequisites:
#   - Python 3.10+
#   - modelscope package (pip install modelscope)
#   - Git LFS (optional, for manual cloning)
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Default model variant (can be overridden by first argument)
MODEL_VARIANT="${1:-en-base}"

# Root directories
VOICEPACK_ROOT="${VOICEPACK_ROOT:-./voicepacks}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-./downloads}"

# Skip flags
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"

# Model metadata (variant → ModelScope repo mapping)
declare -A MODEL_REPOS=(
    ["en-base"]="iic/CosyVoice2-0.5B"
    ["zh-instruct"]="iic/CosyVoice2-0.5B-Instruct"
)

declare -A MODEL_VERSIONS=(
    ["en-base"]="2.0.5"
    ["zh-instruct"]="2.0.5"
)

declare -A MODEL_LANGUAGES=(
    ["en-base"]="en"
    ["zh-instruct"]="zh"
)

declare -A MODEL_DESCRIPTIONS=(
    ["en-base"]="CosyVoice 2.0 English base model for zero-shot TTS"
    ["zh-instruct"]="CosyVoice 2.0 Chinese instruct-tuned model for zero-shot TTS"
)

# ============================================================================
# Utility Functions
# ============================================================================

log_info() {
    echo "[INFO] $*" >&2
}

log_warn() {
    echo "[WARN] $*" >&2
}

log_error() {
    echo "[ERROR] $*" >&2
}

log_success() {
    echo "[SUCCESS] $*" >&2
}

check_command() {
    local cmd="$1"
    if ! command -v "$cmd" &> /dev/null; then
        log_error "Required command not found: $cmd"
        return 1
    fi
}

check_python_package() {
    local package="$1"
    if ! python3 -c "import $package" 2>/dev/null; then
        log_error "Required Python package not found: $package"
        log_error "Install with: pip install $package"
        return 1
    fi
}

# ============================================================================
# Validation
# ============================================================================

validate_environment() {
    log_info "Validating environment..."

    # Check commands
    check_command python3 || exit 1
    check_command jq || { log_warn "jq not found (optional, for JSON validation)"; }

    # Check Python packages
    check_python_package yaml || exit 1

    if [[ "$SKIP_DOWNLOAD" == "0" ]]; then
        check_python_package modelscope || {
            log_error "modelscope package required for downloading models"
            log_error "Install with: pip install modelscope"
            exit 1
        }
    fi

    log_success "Environment validated"
}

validate_model_variant() {
    local variant="$1"

    if [[ ! -v MODEL_REPOS["$variant"] ]]; then
        log_error "Unknown model variant: $variant"
        log_error "Available variants: ${!MODEL_REPOS[*]}"
        exit 1
    fi

    log_info "Model variant: $variant"
    log_info "ModelScope repo: ${MODEL_REPOS[$variant]}"
}

# ============================================================================
# Download
# ============================================================================

download_model() {
    local variant="$1"
    local repo="${MODEL_REPOS[$variant]}"
    local download_path="$DOWNLOAD_DIR/cosyvoice2-$variant"

    if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
        log_info "Skipping download (SKIP_DOWNLOAD=1)"
        if [[ ! -d "$download_path" ]]; then
            log_error "Download directory not found: $download_path"
            exit 1
        fi
        return 0
    fi

    log_info "Downloading model from ModelScope: $repo"
    log_info "Download path: $download_path"

    # Create download directory
    mkdir -p "$DOWNLOAD_DIR"

    # Download using ModelScope Python SDK
    python3 - <<EOF
from modelscope import snapshot_download
import sys

try:
    model_dir = snapshot_download(
        '$repo',
        local_dir='$download_path',
        cache_dir='$DOWNLOAD_DIR/.cache'
    )
    print(f"Downloaded to: {model_dir}")
except Exception as e:
    print(f"ERROR: Download failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF

    if [[ $? -ne 0 ]]; then
        log_error "Model download failed"
        exit 1
    fi

    log_success "Model downloaded successfully"
}

# ============================================================================
# Conversion
# ============================================================================

convert_config_to_json() {
    local download_path="$1"
    local voicepack_path="$2"

    log_info "Converting config from YAML to JSON..."

    local yaml_config="$download_path/cosyvoice.yaml"
    local json_config="$voicepack_path/config.json"

    if [[ ! -f "$yaml_config" ]]; then
        log_warn "YAML config not found, creating default config.json"
        cat > "$json_config" <<'EOF'
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
EOF
        return 0
    fi

    # Convert YAML to JSON using Python
    python3 - <<EOF
import yaml
import json

with open('$yaml_config') as f:
    config = yaml.safe_load(f)

# Ensure sample_rate is set
if 'audio' not in config:
    config['audio'] = {}
config['audio']['sample_rate'] = 24000

with open('$json_config', 'w') as f:
    json.dump(config, f, indent=2)

print("Config converted successfully")
EOF

    if [[ $? -ne 0 ]]; then
        log_error "Config conversion failed"
        exit 1
    fi

    log_success "Config converted to JSON"
}

create_metadata() {
    local variant="$1"
    local voicepack_path="$2"

    log_info "Creating metadata.yaml..."

    local model_id="cosyvoice2-$variant"
    local version="${MODEL_VERSIONS[$variant]}"
    local language="${MODEL_LANGUAGES[$variant]}"
    local description="${MODEL_DESCRIPTIONS[$variant]}"

    cat > "$voicepack_path/metadata.yaml" <<EOF
model_id: "$model_id"
family: "cosyvoice2"
version: "$version"
language: "$language"
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
model_url: "https://modelscope.cn/models/${MODEL_REPOS[$variant]}"
citation: "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models"

# Description
description: "$description"

# Tags
tags:
  - gpu_required
  - streaming
  - neural
  - zero_shot
  - transformer
EOF

    log_success "Metadata created"
}

copy_model_files() {
    local download_path="$1"
    local voicepack_path="$2"

    log_info "Copying model files..."

    # Find PyTorch checkpoint (.pt or .pth)
    local checkpoint
    if [[ -f "$download_path/llm.pt" ]]; then
        checkpoint="$download_path/llm.pt"
    elif [[ -f "$download_path/model.pt" ]]; then
        checkpoint="$download_path/model.pt"
    else
        checkpoint=$(find "$download_path" -name "*.pt" -o -name "*.pth" | head -n 1)
    fi

    if [[ -z "$checkpoint" ]]; then
        log_error "No PyTorch checkpoint found in $download_path"
        exit 1
    fi

    log_info "Found checkpoint: $checkpoint"
    cp "$checkpoint" "$voicepack_path/model.pt"

    # Copy additional files if they exist
    if [[ -f "$download_path/flow.pt" ]]; then
        log_info "Copying flow model..."
        cp "$download_path/flow.pt" "$voicepack_path/"
    fi

    if [[ -f "$download_path/speech_tokenizer_v2.yaml" ]]; then
        log_info "Copying tokenizer config..."
        cp "$download_path/speech_tokenizer_v2.yaml" "$voicepack_path/"
    fi

    log_success "Model files copied"
}

create_readme() {
    local variant="$1"
    local voicepack_path="$2"

    log_info "Creating README.md..."

    local model_id="cosyvoice2-$variant"
    local version="${MODEL_VERSIONS[$variant]}"
    local description="${MODEL_DESCRIPTIONS[$variant]}"

    cat > "$voicepack_path/README.md" <<EOF
# $model_id

**Version**: $version
**Description**: $description

## Model Information

- **Model ID**: $model_id
- **Family**: CosyVoice 2
- **Language**: ${MODEL_LANGUAGES[$variant]}
- **Sample Rate**: 24000 Hz
- **GPU Required**: Yes (CUDA 12.1, PyTorch 2.3.1)

## Capabilities

- Zero-shot voice cloning with 3-10s reference audio
- Streaming synthesis with incremental text processing
- Multi-speaker support

## Usage

### Docker (Recommended)

\`\`\`bash
docker compose up tts-cosyvoice
\`\`\`

### Python API

\`\`\`python
from src.tts.adapters.adapter_cosyvoice import CosyVoiceAdapter

adapter = CosyVoiceAdapter(
    model_id="$model_id",
    model_path="voicepacks/cosyvoice/$variant"
)

async def text_chunks():
    yield "Hello, world!"

async for frame in adapter.synthesize_stream(text_chunks()):
    # Process 20ms audio frame at 48kHz
    pass
\`\`\`

## References

- **Official Repository**: https://github.com/FunAudioLLM/CosyVoice
- **ModelScope**: https://modelscope.cn/models/${MODEL_REPOS[$variant]}
- **Documentation**: [docs/VOICEPACK_COSYVOICE2.md](../../docs/VOICEPACK_COSYVOICE2.md)

---

**Created**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Setup Script**: scripts/setup_cosyvoice_voicepack.sh
EOF

    log_success "README.md created"
}

# ============================================================================
# Validation
# ============================================================================

validate_voicepack() {
    local voicepack_path="$1"

    if [[ "$SKIP_VALIDATION" == "1" ]]; then
        log_info "Skipping validation (SKIP_VALIDATION=1)"
        return 0
    fi

    log_info "Validating voicepack structure..."

    local errors=0

    # Check required files
    local required_files=(
        "model.pt"
        "config.json"
        "metadata.yaml"
    )

    for file in "${required_files[@]}"; do
        if [[ ! -f "$voicepack_path/$file" ]]; then
            log_error "Missing required file: $file"
            ((errors++))
        fi
    done

    # Validate metadata.yaml
    if [[ -f "$voicepack_path/metadata.yaml" ]]; then
        python3 - <<EOF
import yaml
import sys

try:
    with open('$voicepack_path/metadata.yaml') as f:
        metadata = yaml.safe_load(f)

    # Check required fields
    required_fields = ['model_id', 'family', 'sample_rate', 'gpu_required']
    for field in required_fields:
        if field not in metadata:
            print(f"ERROR: Missing required field in metadata.yaml: {field}", file=sys.stderr)
            sys.exit(1)

    # Validate values
    if metadata['family'] != 'cosyvoice2':
        print("ERROR: metadata.yaml: family must be 'cosyvoice2'", file=sys.stderr)
        sys.exit(1)

    if metadata['sample_rate'] != 24000:
        print("ERROR: metadata.yaml: sample_rate must be 24000", file=sys.stderr)
        sys.exit(1)

    if not metadata['gpu_required']:
        print("WARN: metadata.yaml: gpu_required should be true", file=sys.stderr)

    print("Metadata validation passed")

except Exception as e:
    print(f"ERROR: Metadata validation failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF

        if [[ $? -ne 0 ]]; then
            ((errors++))
        fi
    fi

    # Validate config.json
    if [[ -f "$voicepack_path/config.json" ]]; then
        if command -v jq &> /dev/null; then
            if ! jq empty "$voicepack_path/config.json" 2>/dev/null; then
                log_error "Invalid JSON in config.json"
                ((errors++))
            fi
        fi
    fi

    # Check model file size
    if [[ -f "$voicepack_path/model.pt" ]]; then
        local size_mb=$(($(stat -c%s "$voicepack_path/model.pt" 2>/dev/null || stat -f%z "$voicepack_path/model.pt") / 1024 / 1024))
        if [[ $size_mb -lt 100 ]]; then
            log_warn "Model file suspiciously small: ${size_mb}MB (expected >500MB)"
        fi
        log_info "Model file size: ${size_mb}MB"
    fi

    if [[ $errors -gt 0 ]]; then
        log_error "Validation failed with $errors error(s)"
        exit 1
    fi

    log_success "Voicepack validation passed"
}

# ============================================================================
# Main
# ============================================================================

main() {
    log_info "CosyVoice 2 Voicepack Setup Script"
    log_info "==================================="

    # Validate environment
    validate_environment

    # Validate model variant
    validate_model_variant "$MODEL_VARIANT"

    # Set up paths
    local download_path="$DOWNLOAD_DIR/cosyvoice2-$MODEL_VARIANT"
    local voicepack_path="$VOICEPACK_ROOT/cosyvoice/$MODEL_VARIANT"

    log_info "Download path: $download_path"
    log_info "Voicepack path: $voicepack_path"

    # Download model
    download_model "$MODEL_VARIANT"

    # Create voicepack directory
    log_info "Creating voicepack directory..."
    mkdir -p "$voicepack_path"

    # Copy model files
    copy_model_files "$download_path" "$voicepack_path"

    # Convert config
    convert_config_to_json "$download_path" "$voicepack_path"

    # Create metadata
    create_metadata "$MODEL_VARIANT" "$voicepack_path"

    # Create README
    create_readme "$MODEL_VARIANT" "$voicepack_path"

    # Validate voicepack
    validate_voicepack "$voicepack_path"

    # Success summary
    log_success "=========================================="
    log_success "Voicepack setup complete!"
    log_success "=========================================="
    log_success ""
    log_success "Model ID: cosyvoice2-$MODEL_VARIANT"
    log_success "Location: $voicepack_path"
    log_success ""
    log_success "Next steps:"
    log_success "1. Start TTS worker:"
    log_success "   docker compose up tts-cosyvoice"
    log_success ""
    log_success "2. Test synthesis:"
    log_success "   just test-integration -k cosyvoice"
    log_success ""
    log_success "3. View documentation:"
    log_success "   cat docs/VOICEPACK_COSYVOICE2.md"
}

# ============================================================================
# Entry Point
# ============================================================================

# Show help
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    grep "^#" "$0" | grep -v "^#!/" | cut -c3-
    exit 0
fi

# Run main
main
