# Model-Specific Environment Files

This directory contains environment variable configurations for each TTS model adapter.

## File Structure

```
.env.models/
├── .env.piper          # Piper TTS (CPU, PyTorch 2.7.0)
├── .env.cosyvoice2     # CosyVoice 2 (GPU, PyTorch 2.3.1 - isolated)
├── .env.xtts           # XTTS (GPU, future)
├── .env.sesame         # Sesame (GPU, future)
└── .env.openai         # OpenAI API (cloud, future)
```

## Environment Variable Hierarchy

The project uses a layered approach to environment configuration:

1. **`.env.defaults`** (checked into git)
   - Base configuration shared by all deployment modes
   - Infrastructure settings (Redis, LiveKit, Caddy)
   - Default model selection
   - Generic worker settings

2. **`.env.local`** (gitignored, user-specific)
   - Personal overrides (e.g., different GPU, debug logging)
   - API keys (e.g., OpenAI)
   - Local infrastructure overrides (e.g., native Redis instead of Docker)

3. **`.env.models/.env.{model}`** (gitignored, model-specific)
   - Model-specific worker settings
   - Performance tuning (GPU memory, concurrency)
   - Model-specific paths and options

4. **`docker-compose.yml`** (runtime overrides)
   - Service-specific environment variables
   - Container-specific settings

## Usage

### Development with Default Model (Piper)

```bash
# Uses .env.defaults + .env.models/.env.piper
just dev
```

### Development with CosyVoice 2

```bash
# Uses .env.defaults + .env.models/.env.cosyvoice2
DEFAULT_MODEL=cosyvoice2 just dev
```

### Custom Local Overrides

```bash
# Create .env.local from template
cp .env.local.example .env.local

# Edit .env.local with your overrides
# Example: Use GPU 1 instead of GPU 0
echo "CUDA_VISIBLE_DEVICES=1" >> .env.local

# Start development (will use your overrides)
just dev
```

## Model-Specific Settings

### Piper (CPU, Default)

- **Port**: 7001
- **Metrics**: 9090
- **GPU**: Not required
- **Concurrency**: 4 sessions
- **Quality**: Medium (configurable: low, medium, high)

**Key Variables**:
- `PIPER_VOICE_DIR`: Path to Piper voicepack
- `PIPER_QUALITY`: Voice quality preset
- `PIPER_SPEAKER_ID`: Speaker index for multi-speaker models

### CosyVoice 2 (GPU, Isolated Container)

- **Port**: 7002
- **Metrics**: 9091
- **GPU**: Required (6GB VRAM)
- **Concurrency**: 8 sessions
- **PyTorch**: 2.3.1 + CUDA 12.1 (isolated from main project's 2.7.0)

**Key Variables**:
- `COSYVOICE_VOICEPACK_DIR`: Path to CosyVoice voicepack
- `COSYVOICE_MODEL_PATH`: Specific model checkpoint path
- `USE_FLASH_ATTENTION`: Enable Flash Attention 2 (requires Ampere+ GPU)
- `USE_FP16`: Enable half-precision inference
- `ENABLE_EMOTION_CONTROL`: Enable emotion parameters

**Important**: CosyVoice 2 has a PyTorch version conflict and must run in an isolated Docker container. See [docs/COSYVOICE_PYTORCH_CONFLICT.md](../docs/COSYVOICE_PYTORCH_CONFLICT.md).

### XTTS (GPU, Future)

- **Port**: 7003
- **Metrics**: 9092
- **GPU**: Required (4GB VRAM)
- **Concurrency**: 6 sessions
- **Features**: Voice cloning support

*Coming in M7*

### Sesame (GPU, Future)

- **Port**: 7004
- **Metrics**: 9093
- **GPU**: Required
- **Concurrency**: TBD
- **Features**: Multi-lingual support

*Coming in M8*

### OpenAI API (Cloud, Future)

- **Port**: 7005
- **Metrics**: 9094
- **GPU**: Not required (cloud API)
- **Concurrency**: 10 sessions (API rate limited)
- **Features**: High-quality cloud TTS

**Key Variables**:
- `OPENAI_API_KEY`: API key (must be set in `.env.local`)
- `OPENAI_TTS_MODEL`: Model variant (tts-1, tts-1-hd)
- `OPENAI_VOICE`: Voice preset (alloy, echo, fable, onyx, nova, shimmer)

*Coming in future milestone*

## Adding New Models

To add a new TTS model adapter:

1. **Create model-specific Dockerfile**
   ```bash
   # docker/Dockerfile.tts-{model}
   ```

2. **Create model-specific env file**
   ```bash
   # .env.models/.env.{model}
   WORKER_NAME=tts-{model}
   WORKER_GRPC_PORT=700X  # Increment port number
   DEFAULT_MODEL_ID={model}-variant
   # ... model-specific settings
   ```

3. **Add service to docker-compose.models.yml**
   ```yaml
   services:
     tts-{model}:
       build:
         dockerfile: docker/Dockerfile.tts-{model}
       env_file:
         - .env.defaults
         - .env.models/.env.{model}
       profiles:
         - {model}
   ```

4. **Add justfile command (optional)**
   ```makefile
   dev-{model}:
       just dev {model}
   ```

5. **Update documentation**
   - Add model to this README
   - Update CLAUDE.md
   - Create adapter-specific docs in `.claude/modules/adapters/{model}.md`

## Security Notes

- **Never commit `.env.local`** - contains API keys and personal settings
- **Model env files are gitignored** - can be customized per deployment
- **`.env.defaults` is safe to commit** - contains only default values
- API keys should only go in `.env.local` (gitignored)

## Troubleshooting

### Model Not Starting

Check that:
1. Model-specific env file exists: `.env.models/.env.{model}`
2. Voicepack directory exists and is mounted in docker-compose.yml
3. GPU is available (if GPU model): `nvidia-smi`
4. Correct profile is specified: `just dev {model}` or `DEFAULT_MODEL={model} just dev`

### Port Conflicts

Each model uses a unique gRPC port:
- Piper: 7001
- CosyVoice: 7002
- XTTS: 7003
- Sesame: 7004
- OpenAI: 7005

If you get port conflicts, check that no other services are using these ports:
```bash
sudo lsof -i :7001-7005
```

### Environment Variable Not Applied

Check the precedence order:
1. docker-compose.yml overrides everything
2. .env.models/.env.{model} overrides .env.local
3. .env.local overrides .env.defaults
4. .env.defaults is the base

Debug by checking the container's environment:
```bash
docker exec -it tts-{model} env | grep -i {variable}
```

## References

- **Design Document**: `/tmp/unified_dev_workflow_design.md`
- **Main Documentation**: `CLAUDE.md`
- **Development Guide**: `.claude/modules/development.md`
- **PyTorch Conflict**: `docs/COSYVOICE_PYTORCH_CONFLICT.md`
- **Docker Deployment**: `docs/DOCKER_DEPLOYMENT_COSYVOICE.md`
