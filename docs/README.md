# Documentation Index

**Last Updated**: 2025-10-26
**Status**: Production Ready

Welcome to the Full-Duplex Voice Chat documentation! This index helps you find the right documentation for your needs.

---

## Getting Started

New to the system? Start here:

1. **[QUICK_START.md](QUICK_START.md)** - Get running in 15 minutes
   - Installation and setup
   - CPU (Piper) and GPU (CosyVoice) quick starts
   - Parallel synthesis configuration
   - Troubleshooting common issues

2. **[USER_GUIDE.md](USER_GUIDE.md)** - Complete user journey
   - Model selection guide
   - Test scenarios and validation
   - Production deployment
   - Advanced features

3. **[CONFIGURATION.md](CONFIGURATION.md)** - Configuration reference
   - Environment variables
   - TTS/ASR configuration
   - Parallel synthesis settings
   - Deployment profiles

---

## Performance Optimization

**[PARALLEL_TTS.md](PARALLEL_TTS.md)** - Parallel synthesis deep dive
- Architecture and data flow
- 2x throughput improvement details
- GPU memory planning
- Monitoring and troubleshooting
- Production best practices

**[PERFORMANCE.md](PERFORMANCE.md)** - Performance metrics and targets
- Latency SLAs (FAL, ASR, TTS)
- Benchmarks and profiling
- Optimization techniques

---

## Core Documentation

**Status and Implementation:**
- **[CURRENT_STATUS.md](CURRENT_STATUS.md)** - Project status (M0-M10 complete)
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Developer workflows

**Architecture:**
- **[architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md)** - System design
- **[WEBSOCKET_PROTOCOL.md](WEBSOCKET_PROTOCOL.md)** - WebSocket API reference

**Model Configuration:**
- **[VOICEPACK_COSYVOICE2.md](VOICEPACK_COSYVOICE2.md)** - CosyVoice voicepack setup
- **[VOICE_PACKS.md](VOICE_PACKS.md)** - Piper voicepack reference
- **[PIPER_TECHNICAL_REFERENCE.md](PIPER_TECHNICAL_REFERENCE.md)** - Piper adapter details

---

## Deployment

**Docker:**
- **[setup/DOCKER_SETUP.md](setup/DOCKER_SETUP.md)** - Docker deployment guide
- **[DOCKER_DEPLOYMENT_COSYVOICE.md](DOCKER_DEPLOYMENT_COSYVOICE.md)** - CosyVoice Docker isolation
- **[DOCKER_UNIFIED_WORKFLOW.md](DOCKER_UNIFIED_WORKFLOW.md)** - Unified development mode

**Infrastructure:**
- **[HTTPS_SETUP.md](HTTPS_SETUP.md)** - TLS/SSL configuration
- **[LIVEKIT_SETUP.md](LIVEKIT_SETUP.md)** - LiveKit WebRTC setup
- **[REDIS_CONFIGURATION.md](REDIS_CONFIGURATION.md)** - Redis service discovery
- **[NETWORK_ACCESS.md](NETWORK_ACCESS.md)** - Port configuration and networking

**Scaling:**
- **[MULTI_GPU.md](MULTI_GPU.md)** - Multi-GPU deployment
- **[OBSERVABILITY.md](OBSERVABILITY.md)** - Monitoring and metrics

---

## Testing

**Testing Guides:**
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Test commands and CI/CD
- **[UNIFIED_WORKFLOW_TESTING.md](UNIFIED_WORKFLOW_TESTING.md)** - Dev workflow testing
- **[INFRASTRUCTURE_TESTING.md](INFRASTRUCTURE_TESTING.md)** - Infrastructure validation

---

## Troubleshooting

**Runbooks:**
- **[runbooks/ADVANCED_TROUBLESHOOTING.md](runbooks/ADVANCED_TROUBLESHOOTING.md)** - Advanced debugging
- **[runbooks/AUDIO_QUALITY.md](runbooks/AUDIO_QUALITY.md)** - Audio quality issues
- **[runbooks/AUDIO_BACKPRESSURE.md](runbooks/AUDIO_BACKPRESSURE.md)** - Backpressure debugging
- **[runbooks/GRPC_WORKER.md](runbooks/GRPC_WORKER.md)** - gRPC worker issues
- **[runbooks/WEBSOCKET.md](runbooks/WEBSOCKET.md)** - WebSocket debugging
- **[runbooks/REDIS.md](runbooks/REDIS.md)** - Redis connectivity
- **[runbooks/PORT_CONFLICTS.md](runbooks/PORT_CONFLICTS.md)** - Port conflict resolution
- **[runbooks/TEST_DEBUGGING.md](runbooks/TEST_DEBUGGING.md)** - Test debugging

**Known Issues:**
- **[known-issues/grpc-segfault.md](known-issues/grpc-segfault.md)** - gRPC segfault workaround (WSL2)
- **[COSYVOICE_PYTORCH_CONFLICT.md](COSYVOICE_PYTORCH_CONFLICT.md)** - PyTorch version conflict
- **[CUDA_COMPATIBILITY.md](CUDA_COMPATIBILITY.md)** - CUDA compatibility notes

---

## CLI Tools

**Client Reference:**
- **[CLI_CLIENT_GUIDE.md](CLI_CLIENT_GUIDE.md)** - CLI client usage
- **[ORCHESTRATOR_MODES.md](ORCHESTRATOR_MODES.md)** - Orchestrator mode selection
- **[ORCHESTRATOR_MODE_SELECTION_SUMMARY.md](ORCHESTRATOR_MODE_SELECTION_SUMMARY.md)** - Mode selection summary

---

## Implementation Details

**Milestone Summaries:**
- **[M3_VAD_INTEGRATION_SUMMARY.md](M3_VAD_INTEGRATION_SUMMARY.md)** - VAD integration details

**Implementation Notes:**
- **[implementation/session-protocol-fix.md](implementation/session-protocol-fix.md)** - Session protocol fixes

---

## Reference Material

**Quick References:**
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command cheat sheet
- **[QUICKSTART.md](QUICKSTART.md)** - Alternative quick start

**Configuration:**
- **[CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md)** - Complete config reference
- **[DEFAULT_MODEL_ENV_VAR.md](DEFAULT_MODEL_ENV_VAR.md)** - Model environment variables

**Usage Examples:**
- **[USAGE_DOCKER_PIPER.md](USAGE_DOCKER_PIPER.md)** - Docker + Piper usage

---

## Recommended Reading Order

### For First-Time Users:
1. [QUICK_START.md](QUICK_START.md) - Get system running
2. [USER_GUIDE.md](USER_GUIDE.md) - Learn core features
3. [CONFIGURATION.md](CONFIGURATION.md) - Configure for your needs

### For Production Deployment:
1. [PARALLEL_TTS.md](PARALLEL_TTS.md) - Enable parallel synthesis
2. [PERFORMANCE.md](PERFORMANCE.md) - Performance optimization
3. [setup/DOCKER_SETUP.md](setup/DOCKER_SETUP.md) - Docker deployment
4. [HTTPS_SETUP.md](HTTPS_SETUP.md) - Security configuration
5. [OBSERVABILITY.md](OBSERVABILITY.md) - Set up monitoring

### For Developers:
1. [DEVELOPMENT.md](DEVELOPMENT.md) - Development workflows
2. [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md) - System architecture
3. [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing strategy
4. [CURRENT_STATUS.md](CURRENT_STATUS.md) - Implementation status

---

## Documentation Statistics

- **Total Documents**: 54 markdown files
- **Core Guides**: 3 (QUICK_START, USER_GUIDE, PARALLEL_TTS)
- **Runbooks**: 10 troubleshooting guides
- **Test Coverage**: 790 tests (100% pass rate)
- **Last Updated**: 2025-10-26

---

## Contributing to Documentation

Found an issue or want to improve the docs?

1. Check [CURRENT_STATUS.md](CURRENT_STATUS.md) for latest status
2. Submit issues via GitHub Issues
3. Propose changes via Pull Requests
4. Follow documentation conventions in existing files

---

**Maintained by**: Documentation Engineering Team
**Project**: Full-Duplex Voice Chat
**Repository**: https://github.com/gsornsen/full-duplex-voice-chat
