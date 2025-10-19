# Multi-Agent Coordination

**Last Updated**: 2025-10-17
**Coordinator**: `@agent-mycelium-core:multi-agent-coordinator`
**State Management**: Redis MCP Server (`mcp__RedisMCPServer__*`)

This project uses a multi-agent team orchestrated by the mycelium-core coordinator with Redis-based state management.

## Agent Team Roster

### Core Quality & Infrastructure Agents

**@agent-mycelium-core:documentation-engineer** - Documentation audit and consistency
- **Specialization**: Cross-document validation, API docs, audit reports
- **Invoke for**: Milestone completion, doc audits, cross-reference validation
- **Redis state**: `agent:docs:status`, `agent:docs:audit_results`

**@agent-mycelium-core:devops-engineer** - CI/CD and deployment
- **Specialization**: Pipeline optimization, Docker, deployment automation
- **Invoke for**: CI failures, deployment issues, workflow optimization
- **Redis state**: `agent:devops:ci_status`, `agent:devops:cache_metrics`

**@agent-mycelium-core:python-pro** - Code quality and testing
- **Specialization**: Type safety, test coverage, Python best practices
- **Invoke for**: Type errors, coverage issues, code review
- **Redis state**: `agent:python:coverage`, `agent:python:test_results`

### Feature Development Agents

**@agent-mycelium-core:ml-engineer** - Machine learning and inference
- **Specialization**: Model optimization, TTS/ASR integration, performance tuning
- **Invoke for**: TTS adapter development (M6-M8), ASR optimization, inference profiling
- **Redis state**: `agent:ml:model_metrics`, `agent:ml:inference_stats`

**@agent-mycelium-core:nextjs-developer** - Web client development
- **Specialization**: Next.js 14+, React 18+, frontend architecture
- **Invoke for**: Web client features, LiveKit integration, UI components
- **Redis state**: `agent:nextjs:build_status`, `agent:nextjs:component_registry`

**@agent-mycelium-core:react-tanstack-developer** - React + TanStack ecosystem
- **Specialization**: TanStack Query, Table, Router, Form; clean abstractions
- **Invoke for**: State management, data fetching, table/form components
- **Redis state**: `agent:tanstack:query_cache`, `agent:tanstack:optimizations`

### Planning & Architecture Agents

**@agent-mycelium-core:project-manager** - Project planning and coordination
- **Specialization**: Task breakdown, milestone planning, resource allocation
- **Invoke for**: New milestone planning, scope definition, task prioritization
- **Redis state**: `agent:pm:milestones`, `agent:pm:task_queue`

## Orchestration with Multi-Agent Coordinator

**Primary Orchestrator**: `@agent-mycelium-core:multi-agent-coordinator`

The coordinator handles:
- **Team assembly**: Selects optimal agents for each task
- **Task decomposition**: Breaks complex work into agent-specific subtasks
- **Parallel execution**: Coordinates concurrent agent work
- **State synchronization**: Uses Redis for inter-agent communication
- **Workflow optimization**: Manages handoffs and dependencies

### Redis State Management

**Agent coordination state** (managed by coordinator):
```
agent:coordinator:active_agents     # Set of currently active agents
agent:coordinator:task_queue        # Queue of pending tasks
agent:coordinator:workflow:{id}     # Workflow state (JSON)
agent:coordinator:handoff:{from}:{to} # Agent handoff data
```

**Agent-specific state** (managed by individual agents):
```
agent:{agent_type}:status           # Current status (idle/working/blocked)
agent:{agent_type}:current_task     # Active task description
agent:{agent_type}:results          # Task results (JSON)
agent:{agent_type}:metrics          # Performance metrics
```

**Project state** (shared):
```
project:milestones:current          # Current milestone (e.g., "M11")
project:milestones:{id}:status      # Milestone status (JSON)
project:tests:total_count           # Current test count (649)
project:coverage:percentage         # Current coverage (e.g., 82.5)
project:ci:last_run                 # Last CI run status
```

## Coordination Workflows

### Workflow 1: Milestone Completion (Coordinated)

**Trigger**: Developer completes milestone implementation

**Orchestration**:
```
User: "I completed M11 implementation"
    ↓
@multi-agent-coordinator: Analyze task, assemble team
    ↓
Coordinator → Redis: SET agent:coordinator:workflow:m11_completion {
  "agents": ["python-pro", "documentation-engineer", "devops-engineer"],
  "status": "in_progress",
  "parallel": ["python-pro", "documentation-engineer"]
}
    ↓
┌─────────────────────────────────────────────────────┐
│ Parallel Execution (Phase 1)                       │
├─────────────────────────────────────────────────────┤
│ @python-pro                  @documentation-engineer│
│ - Validate code quality      - Audit all docs       │
│ - Check test coverage        - Verify consistency   │
│ - Review type hints          - Update status docs   │
│ → Redis: agent:python:*      → Redis: agent:docs:*  │
└─────────────────────────────────────────────────────┘
    ↓
Coordinator: Check Redis states, wait for completion
    ↓
@devops-engineer: Verify CI/CD (sequential, after Phase 1)
    ↓
Coordinator: Aggregate results from Redis
    ↓
@documentation-engineer: Generate commit/PR docs
    ↓
Coordinator → User: Present consolidated report
```

**Redis keys used**:
```bash
# Coordinator workflow state
agent:coordinator:workflow:m11_completion

# Agent task states
agent:python:status = "working"
agent:python:current_task = "M11 code quality validation"
agent:python:results = {"coverage": 85.2, "tests_passing": true}

agent:docs:status = "working"
agent:docs:current_task = "M11 documentation audit"
agent:docs:audit_results = {"inconsistencies": 0, "updates_needed": 3}

agent:devops:status = "completed"
agent:devops:ci_status = {"duration": "4m 32s", "cache_hit_rate": 94}

# Project state updates
project:milestones:current = "M11"
project:milestones:m11:status = {"phase": "review", "agents_complete": 3}
project:tests:total_count = 692  # Updated by @python-pro
```

### Workflow 2: New Feature Development (TTS Adapter)

**Trigger**: User requests new TTS adapter implementation

**Orchestration**:
```
User: "Implement CosyVoice 2 adapter (M6)"
    ↓
@multi-agent-coordinator: Analyze requirements
    ↓
@project-manager: Break down milestone into tasks
    ↓
Coordinator → Redis: Create task queue
    ↓
┌─────────────────────────────────────────────────────┐
│ Task Queue (Redis: project:m6:tasks)               │
├─────────────────────────────────────────────────────┤
│ 1. Design adapter architecture (@ml-engineer)       │
│ 2. Implement streaming synthesis (@python-pro)      │
│ 3. Add model loading/unload (@ml-engineer)          │
│ 4. Create unit tests (@python-pro)                  │
│ 5. Integration testing (@devops-engineer)           │
│ 6. Documentation (@documentation-engineer)          │
└─────────────────────────────────────────────────────┘
    ↓
@ml-engineer: Design architecture, update Redis
    ↓
@python-pro: Implement based on ml-engineer's design
    ↓
(Tasks proceed sequentially/parallel as dependencies allow)
    ↓
Coordinator: Track progress via Redis states
    ↓
Coordinator → User: Status updates, completion report
```

### Workflow 3: Web Client Feature (LiveKit Integration)

**Trigger**: User requests web client enhancement

**Orchestration**:
```
User: "Add audio visualization to web client"
    ↓
@multi-agent-coordinator: Identify needed agents
    ↓
┌─────────────────────────────────────────────────────┐
│ Agent Selection                                     │
├─────────────────────────────────────────────────────┤
│ @nextjs-developer    - Next.js integration         │
│ @react-tanstack-dev  - State management             │
│ @devops-engineer     - Build optimization           │
└─────────────────────────────────────────────────────┘
    ↓
Coordinator: Coordinate handoffs via Redis
    ↓
@react-tanstack-developer: Design state layer
    ↓ (handoff via Redis: agent:handoff:tanstack:nextjs)
@nextjs-developer: Implement component
    ↓
@devops-engineer: Optimize build
    ↓
Coordinator → User: Feature complete
```

## Redis State Management Patterns

### Pattern 1: Agent Status Tracking

```python
# Agent updates own status
mcp__RedisMCPServer__hset(
    name="agent:python:status",
    key="state",
    value="working"
)
mcp__RedisMCPServer__hset(
    name="agent:python:status",
    key="task",
    value="M11 code quality validation"
)

# Coordinator checks agent status
status = mcp__RedisMCPServer__hget(
    name="agent:python:status",
    key="state"
)
```

### Pattern 2: Task Queue Management

```python
# Coordinator creates task queue
mcp__RedisMCPServer__rpush(
    name="project:m6:tasks",
    value='{"id": 1, "agent": "ml-engineer", "task": "Design adapter"}'
)

# Agent pops task from queue
task = mcp__RedisMCPServer__lpop(name="project:m6:tasks")

# Agent pushes results
mcp__RedisMCPServer__rpush(
    name="project:m6:results",
    value='{"task_id": 1, "status": "complete", "output": "..."}'
)
```

### Pattern 3: Inter-Agent Handoffs

```python
# Agent 1 completes work, signals handoff
mcp__RedisMCPServer__json_set(
    name="agent:handoff:ml:python",
    path="$",
    value={
        "from": "ml-engineer",
        "to": "python-pro",
        "task": "Implement CosyVoice2 adapter",
        "context": "Architecture designed in docs/m6_design/",
        "status": "ready"
    }
)

# Coordinator notifies Agent 2
# Agent 2 reads handoff context
handoff = mcp__RedisMCPServer__json_get(
    name="agent:handoff:ml:python",
    path="$"
)
```

### Pattern 4: Project State Sync

```python
# Update shared project state
mcp__RedisMCPServer__hset(
    name="project:stats",
    key="total_tests",
    value=692
)
mcp__RedisMCPServer__hset(
    name="project:stats",
    key="coverage",
    value=85.2
)

# All agents can read current state
tests = mcp__RedisMCPServer__hget(name="project:stats", key="total_tests")
```

## When to Invoke Coordinator

**DO invoke coordinator for:**
- ✅ Multi-step milestones requiring 3+ agents
- ✅ Complex features with dependencies between agents
- ✅ Parallel work coordination (e.g., simultaneous doc audit + code review)
- ✅ Workflow optimization (find optimal agent sequencing)

**DON'T invoke coordinator for:**
- ❌ Single-agent tasks (invoke agent directly)
- ❌ Simple sequential work (manual coordination)
- ❌ Exploratory work (main Claude handles)

**Example invocations**:
```
# Good: Complex milestone
"@multi-agent-coordinator: Orchestrate M11 milestone completion with full team"

# Good: Multi-agent feature
"@multi-agent-coordinator: Coordinate TTS adapter implementation (M6) with ml-engineer and python-pro"

# Bad: Single agent task (invoke directly instead)
"@documentation-engineer: Update test counts in docs"
```

## Monitoring Agent Team

**Check coordinator status**:
```bash
# Active agents
redis-cli SMEMBERS agent:coordinator:active_agents

# Workflow state
redis-cli HGETALL agent:coordinator:workflow:m11_completion

# Task queue
redis-cli LRANGE project:m6:tasks 0 -1
```

**Check individual agent status**:
```bash
# Agent state
redis-cli HGET agent:python:status state

# Agent task
redis-cli HGET agent:python:status task

# Agent results
redis-cli GET agent:python:results
```

**Dashboard view** (all states):
```bash
redis-cli KEYS "agent:*"
redis-cli KEYS "project:*"
```

## References

- **Multi-Agent Coordinator**: Mycelium Core plugin agent
- **Redis MCP Server**: `mcp__RedisMCPServer__*` tools for state management
- **Acceptance Criteria**: [CLAUDE.md#mandatory-acceptance-criteria](../../CLAUDE.md#mandatory-acceptance-criteria)
- **Project Status**: Tracked in Redis `project:milestones:*` keys

---

**Note**: Coordinator automatically manages Redis state. Agents self-report status. Main Claude can query Redis for real-time team status.
