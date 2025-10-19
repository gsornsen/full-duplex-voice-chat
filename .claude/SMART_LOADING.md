# Smart Loading Guide

**Last Updated**: 2025-10-17
**Phase**: 3 - Context-Aware Module Loading

This guide explains how Claude Code intelligently loads documentation modules based on your current work context.

## Overview

Smart Loading uses **context triggers** to automatically suggest or load relevant documentation based on:
- Files you're currently editing
- Keywords in your queries
- Related file patterns in your workspace

**Benefits**:
- ✅ No manual module selection needed
- ✅ Always have relevant context loaded
- ✅ Minimize token usage (load only what's needed)
- ✅ Faster task completion with right documentation

## How It Works

### 1. Context Detection

When you start working, Claude Code analyzes:

```
User opens: src/orchestrator/vad.py
    ↓
Smart Loading detects: "VAD Implementation" trigger
    ↓
Suggests loading:
  - .claude/modules/features/vad.md
  - .claude/modules/architecture.md#orchestrator-layer
    ↓
User confirms or auto-loads (based on priority)
```

### 2. File Pattern Matching

**Example triggers** (from `.claude/context-triggers.json`):

| Files Editing | Modules Auto-Suggested |
|---------------|------------------------|
| `src/orchestrator/vad.py` | VAD feature module, Architecture |
| `src/asr/adapters/*.py` | ASR module, Architecture, Whisper docs |
| `src/tts/model_manager.py` | Model Manager module, Architecture |
| `src/tts/adapters/adapter_piper.py` | Piper module, Adapter template |
| `tests/**/*.py` | Testing module, Development module |

### 3. Keyword Detection

Claude Code recognizes keywords in your queries:

**Example**:
```
You: "I'm working on VAD improvements"
    ↓
Keywords detected: "VAD", "improvements"
    ↓
Auto-loads: .claude/modules/features/vad.md
```

**Keyword Mapping**:
- "VAD", "barge-in", "noise gate" → VAD module
- "ASR", "whisper", "transcription" → ASR module
- "session", "multi-turn", "timeout" → Session Management module
- "model manager", "TTL", "eviction" → Model Manager module
- "adapter", "synthesize", "framing" → Adapter modules

### 4. Priority Levels

Modules have different priority levels for auto-loading:

**High Priority** (suggest on 2+ file matches):
- VAD Implementation
- ASR Integration
- Session Management
- Model Manager
- Orchestrator

**Medium Priority** (suggest on 3+ file matches):
- TTS Adapters
- Piper Adapter
- CosyVoice Adapter
- Audio Processing
- Testing & CI/CD

**Low Priority** (suggest only on explicit request):
- Development Environment
- Milestone Implementation

## Usage Examples

### Example 1: Working on VAD

```
You: "I need to debug the adaptive noise gate"
```

**Smart Loading Response**:
```
I'll help you debug the adaptive noise gate. Let me load the relevant documentation:

Loading:
✅ .claude/modules/features/vad.md (VAD implementation details)
✅ .claude/modules/architecture.md#orchestrator-layer (VAD architecture)

The adaptive noise gate was implemented in M10 Polish. Key implementation details:
- Location: src/orchestrator/vad_processor.py
- RMS buffer: src/orchestrator/audio/buffer.py
- Configuration: noise_gate section in configs/orchestrator.yaml

[Detailed debugging guidance follows...]
```

### Example 2: Creating New Adapter

```
You: "I want to implement an XTTS adapter"
```

**Smart Loading Response**:
```
I'll help you implement the XTTS adapter. Loading relevant guides:

Loading:
✅ .claude/modules/adapters/template.md (adapter implementation template)
✅ .claude/modules/adapters/README.md (adapter architecture overview)
✅ .claude/modules/features/model-manager.md (ModelManager integration)

Let's follow the adapter template. Here's the implementation checklist:

1. Create src/tts/adapters/adapter_xtts.py
2. Inherit from TTSAdapterBase
3. Implement required methods...

[Step-by-step implementation guidance follows...]
```

### Example 3: Fixing CI Failures

```
Files modified: tests/unit/orchestrator/test_session.py
You: "Why is CI failing?"
```

**Smart Loading Response**:
```
I see you're working on session tests. Let me check the CI failures.

Loading:
✅ .claude/modules/testing.md (CI/CD pipeline and troubleshooting)
✅ .claude/modules/features/session-management.md (session implementation)

Analyzing CI logs...

[CI failure analysis with relevant context from loaded modules...]
```

### Example 4: Milestone Completion

```
You: "I completed M11 implementation"
```

**Smart Loading Response**:
```
Excellent! Let me help you complete the M11 milestone acceptance criteria.

Loading:
✅ .claude/modules/milestones.md (M11 exit criteria)
✅ .claude/agents/COORDINATION.md (multi-agent orchestration)

According to M11 exit criteria, I'll coordinate:
1. @python-pro: Code quality validation
2. @documentation-engineer: Documentation audit
3. @devops-engineer: CI/CD verification

[Multi-agent coordination workflow follows...]
```

## Configuration

### Context Triggers File

**Location**: `.claude/context-triggers.json`

**Structure**:
```json
{
  "triggers": [
    {
      "name": "VAD Implementation",
      "file_patterns": ["src/orchestrator/vad.py", ...],
      "modules": [".claude/modules/features/vad.md", ...],
      "keywords": ["VAD", "barge-in", ...],
      "priority": "high"
    }
  ]
}
```

### Adding Custom Triggers

To add a new context trigger:

1. Edit `.claude/context-triggers.json`
2. Add trigger to `triggers` array:
   ```json
   {
     "name": "My Feature",
     "file_patterns": ["src/my_feature/**/*.py"],
     "modules": [".claude/modules/features/my_feature.md"],
     "keywords": ["my feature", "my keyword"],
     "priority": "high"
   }
   ```
3. Update module metadata if needed

### Module Front-Matter

Each module includes front-matter metadata for smart loading:

```markdown
---
title: "VAD Implementation"
tags: ["vad", "barge-in", "noise-gate", "orchestrator"]
related_files:
  - "src/orchestrator/vad.py"
  - "src/orchestrator/vad_processor.py"
dependencies:
  - ".claude/modules/architecture.md"
estimated_tokens: 1200
priority: "high"
---

# Voice Activity Detection (VAD)
...
```

## Token Budget Management

Smart Loading tracks token usage to stay within budget:

**Session Start**:
- CLAUDE.md core: ~2,000 tokens (always loaded)
- Budget remaining: 198,000 tokens

**Context-Triggered Loading**:
- High priority module (VAD): +1,200 tokens → 196,800 remaining
- Related module (Architecture): +1,800 tokens → 195,000 remaining
- **Total loaded**: ~5,000 tokens (2.5% of budget)

**Optimization**:
- Load high-priority modules immediately
- Load medium-priority on 3+ file matches
- Load low-priority only on explicit request
- Unload modules when no longer relevant

## Best Practices

### For Users

1. **Trust the suggestions**: Smart Loading knows which docs are relevant
2. **Use keywords**: Mention "VAD", "ASR", "adapter" in your queries
3. **Work on related files together**: Triggers activate on file patterns
4. **Review loaded modules**: Check what's loaded with `/context` command

### For Maintainers

1. **Keep triggers updated**: Add new patterns when creating features
2. **Accurate token estimates**: Update `estimated_tokens` in metadata
3. **Test trigger accuracy**: Verify triggers load correct modules
4. **Monitor false positives**: Adjust file patterns if wrong modules load

## Monitoring & Debugging

### Check Active Triggers

```bash
# View all triggers
cat .claude/context-triggers.json | jq '.triggers[] | {name, priority}'

# View module metadata
cat .claude/context-triggers.json | jq '.module_metadata'
```

### Debug Loading Issues

If wrong modules are loading:

1. Check file patterns in `.claude/context-triggers.json`
2. Verify priority levels (high/medium/low)
3. Review keyword matches
4. Update trigger configuration

### Token Usage Tracking

Claude Code automatically tracks:
- Modules loaded per session
- Total tokens consumed
- Budget remaining
- Module load/unload events

## Advanced Features

### Dependency Chain Loading

When loading a module with dependencies:

```
Load: .claude/modules/features/session-management.md
    ↓ (depends on)
Auto-load: .claude/modules/features/vad.md
    ↓ (depends on)
Auto-load: .claude/modules/architecture.md
```

Smart Loading handles dependency chains automatically.

### Multi-File Context

When working on multiple related files:

```
Files open:
  - src/orchestrator/vad.py
  - src/orchestrator/vad_processor.py
  - tests/unit/orchestrator/test_vad.py
    ↓
Smart Loading detects: All VAD-related
    ↓
Loads: VAD module + Architecture module + Testing module
```

### Keyword Combinations

Multiple keywords trigger multiple modules:

```
You: "I'm debugging the VAD noise gate in session timeout handling"
Keywords: "VAD", "noise gate", "session timeout"
    ↓
Loads:
  - .claude/modules/features/vad.md
  - .claude/modules/features/session-management.md
  - .claude/modules/architecture.md
```

## References

- **Context Triggers**: [.claude/context-triggers.json](.claude/context-triggers.json)
- **Module Metadata**: Embedded in trigger file
- **Phase 1 & 2 Overview**: [.claude/README.md](.claude/README.md)
- **Core Documentation**: [CLAUDE.md](../CLAUDE.md)

---

**Phase 3 Status**: ✅ Complete (2025-10-17)
**Smart Loading**: Enabled by default
**Token Optimization**: 87% reduction maintained with intelligent loading
