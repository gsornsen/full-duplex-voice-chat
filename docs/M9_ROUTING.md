# M9 Routing v1 Documentation

**Status**: Implemented
**Version**: 1.0.0
**Date**: 2025-10-26

## Overview

M9 Routing v1 implements intelligent worker selection for TTS synthesis with capability-based matching, load balancing, and session affinity. This enables efficient multi-worker deployments with voice consistency and optimal resource utilization.

## Features

### 1. Capability-Based Routing

Routes synthesis requests to workers based on their capabilities:
- **Language support**: Match worker language capabilities to request requirements
- **Model support**: Prefer workers with requested model already loaded
- **Feature support**: Match streaming, zero-shot, LoRA capabilities

### 2. Load Balancing Strategies

Multiple strategies for distributing load across workers:
- **Least-loaded** (default): Select worker with lowest queue depth
- **Least-latency**: Select worker with best synthesis performance (RTF)
- **Round-robin**: Simple fair distribution
- **Random**: Random selection (for testing)

### 3. Session Affinity

Maintains voice consistency across multi-turn conversations:
- Same session always routes to same worker
- Configurable TTL (default: 1 hour)
- Graceful fallback when affinity worker unavailable
- Redis-backed for persistence

### 4. Health Filtering

Excludes unhealthy workers from routing:
- Heartbeat-based health checks
- Configurable health threshold (default: 2x TTL)
- Automatic exclusion of stale workers

### 5. Performance Targets

- **Routing overhead**: <1ms p95
- **Session affinity hit rate**: >95%
- **Load distribution variance**: <10%

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Routing Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Get registered workers (from WorkerRegistry)             │
│           ↓                                                   │
│  2. Filter by capability (language, model support)           │
│           ↓                                                   │
│  3. Filter by health (heartbeat check)                       │
│           ↓                                                   │
│  4. Check session affinity (Redis lookup)                    │
│           ↓                                                   │
│  5. Prefer resident models (if configured)                   │
│           ↓                                                   │
│  6. Apply load balancing strategy                            │
│           ↓                                                   │
│  7. Update session affinity (Redis store)                    │
│           ↓                                                   │
│  8. Return selected worker                                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Router (`src/orchestrator/routing.py`)

Main routing engine with capability matching and session affinity:

```python
from src.orchestrator.routing import Router, RoutingStrategy
from src.orchestrator.registry import WorkerRegistry

# Create router
registry = WorkerRegistry(redis_url="redis://localhost:6379")
router = Router(
    registry=registry,
    redis_url="redis://localhost:6379",
    load_balance_strategy="least_loaded",
    affinity_enabled=True,
)

await router.initialize()

# Select worker
worker_addr = await router.select_worker(
    session_id="session-123",
    model_id="cosyvoice2-en-base",
    language="en",
    capabilities={"streaming": True},
    strategy=RoutingStrategy.LEAST_LOADED,
)

# Get routing metrics
metrics = router.get_metrics()
print(f"Affinity hit rate: {metrics['affinity_hit_rate_percent']:.1f}%")
```

### WorkerSelector (`src/orchestrator/worker_selector.py`)

Load balancing algorithms:

```python
from src.orchestrator.worker_selector import WorkerSelector

selector = WorkerSelector()

# Round-robin selection
worker = selector.round_robin(workers)

# Least-loaded selection
worker = selector.least_loaded(workers)

# Least-latency selection
worker = selector.least_latency(workers)

# Weighted selection (custom)
worker = selector.weighted_selection(
    workers,
    queue_weight=0.7,  # 70% weight on queue depth
    latency_weight=0.3,  # 30% weight on latency
)
```

## Configuration

### Environment Variables

Add to `.env` file:

```bash
# Routing Strategy (round_robin, least_loaded, least_latency, random)
ROUTING_STRATEGY=least_loaded

# Session Affinity
ROUTING_AFFINITY_ENABLED=true
ROUTING_AFFINITY_TTL=3600  # 1 hour

# Worker Health
ROUTING_HEALTH_CHECK_INTERVAL=30  # seconds

# Model Preference
ROUTING_PREFER_RESIDENT_MODELS=true

# Fallback Worker (optional)
ROUTING_FALLBACK_WORKER=grpc://localhost:7001
```

### Programmatic Configuration

```python
router = Router(
    registry=registry,
    redis_url="redis://localhost:6379",
    static_worker_addr=None,  # Disable static routing
    prefer_resident_models=True,
    load_balance_strategy="least_loaded",
    affinity_enabled=True,
    affinity_ttl_seconds=3600,
    health_check_interval=30,
    redis_key_prefix="routing:affinity:",
)
```

## Routing Strategies

### Least-Loaded (Default)

Selects worker with lowest queue depth:
- **Best for**: Latency-sensitive applications
- **Pros**: Minimizes queuing delay, automatic load distribution
- **Cons**: May ignore synthesis speed differences

```python
worker = await router.select_worker(
    session_id="session-123",
    strategy=RoutingStrategy.LEAST_LOADED,
)
```

### Least-Latency

Selects worker with best synthesis performance (RTF):
- **Best for**: GPU vs CPU heterogeneous pools
- **Pros**: Optimizes for fastest synthesis
- **Cons**: May overload fast workers

```python
worker = await router.select_worker(
    session_id="session-123",
    strategy=RoutingStrategy.LEAST_LATENCY,
)
```

### Round-Robin

Cycles through workers in order:
- **Best for**: Testing, predictable distribution
- **Pros**: Simple, fair distribution
- **Cons**: Ignores current load and performance

```python
worker = await router.select_worker(
    session_id="session-123",
    strategy=RoutingStrategy.ROUND_ROBIN,
)
```

### Random

Random selection:
- **Best for**: Load testing, chaos engineering
- **Pros**: Simple, unpredictable
- **Cons**: May create imbalanced distribution

```python
worker = await router.select_worker(
    session_id="session-123",
    strategy=RoutingStrategy.RANDOM,
)
```

## Session Affinity

### How It Works

Session affinity ensures voice consistency by routing all messages from the same session to the same worker:

1. **First request**: No affinity exists, route using load balancing
2. **Store affinity**: Save session→worker mapping in Redis with TTL
3. **Subsequent requests**: Lookup affinity, route to same worker
4. **TTL expiry**: Affinity expires after inactivity, new worker selected

### Configuration

```python
# Enable session affinity
router = Router(
    registry=registry,
    redis_url="redis://localhost:6379",
    affinity_enabled=True,
    affinity_ttl_seconds=3600,  # 1 hour
    redis_key_prefix="routing:affinity:",
)

# Disable session affinity
router = Router(
    registry=registry,
    affinity_enabled=False,
)
```

### Graceful Fallback

When affinity worker is unavailable:
1. Check if worker is in available pool
2. If not available, fall back to load balancing
3. Create new affinity mapping
4. Log warning

This ensures service continuity even when workers fail.

## Health Filtering

Workers are excluded from routing if their heartbeat is stale:

```python
router = Router(
    registry=registry,
    health_check_interval=30,  # 30s heartbeat interval
)

# Workers are unhealthy if heartbeat > 60s old (2x interval)
```

### Health Check Logic

```python
current_time = time.time()
health_threshold = health_check_interval * 2  # 60s

for worker in workers:
    age = current_time - worker.last_heartbeat_ts
    if age < health_threshold:
        healthy_workers.append(worker)
    else:
        logger.warning(f"Worker {worker.name} unhealthy ({age:.1f}s old)")
```

## Capability Matching

### Language Filtering

```python
# Request Chinese language support
worker = await router.select_worker(language="zh")

# Matches workers with: capabilities["languages"] = ["zh"] or ["en", "zh"]
```

### Feature Capabilities

```python
# Request streaming + zero-shot
worker = await router.select_worker(
    capabilities={
        "streaming": True,
        "zero_shot": True,
    }
)

# Only matches workers with ALL required capabilities
```

### Resident Model Preference

```python
# Prefer workers with model loaded (avoid 2-5s load latency)
router = Router(
    registry=registry,
    prefer_resident_models=True,
)

worker = await router.select_worker(model_id="cosyvoice2-en-base")

# Filters to workers with: "cosyvoice2-en-base" in resident_models
# Falls back to all workers if no matches
```

## Metrics

### Collected Metrics

```python
metrics = router.get_metrics()
```

Returns:
- `total_decisions`: Total routing decisions made
- `affinity_hits`: Session affinity cache hits
- `affinity_misses`: Session affinity cache misses
- `affinity_hit_rate_percent`: Hit rate percentage (>95% target)
- `no_workers_errors`: Errors due to no workers available
- `capability_filters`: Requests filtered by capability
- `health_filters`: Requests filtered by health
- `avg_latency_ms`: Average routing decision latency (<1ms target)

### Monitoring

```python
# Example metrics output
{
    "total_decisions": 1000,
    "affinity_hits": 950,
    "affinity_misses": 50,
    "affinity_hit_rate_percent": 95.0,  # ✅ Target: >95%
    "no_workers_errors": 0,
    "capability_filters": 10,
    "health_filters": 2,
    "avg_latency_ms": 0.5,  # ✅ Target: <1ms
}
```

## Multi-Worker Deployment

### Same-Host Multi-GPU

```bash
# Terminal 1: Worker 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 WORKER_NAME=worker-0 WORKER_GRPC_PORT=7001 \
  uv run python -m src.tts.worker

# Terminal 2: Worker 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 WORKER_NAME=worker-1 WORKER_GRPC_PORT=7002 \
  uv run python -m src.tts.worker

# Terminal 3: Orchestrator
ROUTING_STRATEGY=least_loaded ROUTING_AFFINITY_ENABLED=true \
  uv run python -m src.orchestrator.agent
```

### Multi-Host Deployment

```bash
# Host 1: Worker + Redis
docker run -d --name redis -p 6379:6379 redis:7-alpine
docker run --gpus all -e REDIS_URL=redis://host1:6379 tts-worker

# Host 2: Worker
docker run --gpus all -e REDIS_URL=redis://host1:6379 tts-worker

# Host 3: Orchestrator
docker run -e REDIS_URL=redis://host1:6379 \
           -e ROUTING_STRATEGY=least_loaded \
           orchestrator
```

## Error Handling

### No Workers Available

```python
try:
    worker = await router.select_worker()
except NoWorkersAvailableError:
    # All workers down or no workers registered
    if fallback_worker:
        worker = fallback_worker
    else:
        raise ServiceUnavailable("No TTS workers available")
```

### All Workers Unhealthy

```python
try:
    worker = await router.select_worker()
except RuntimeError as e:
    if "unhealthy" in str(e):
        # Wait for workers to recover
        await asyncio.sleep(5)
        worker = await router.select_worker()
```

### Capability Mismatch

```python
try:
    worker = await router.select_worker(language="fr")
except RuntimeError as e:
    if "No workers support" in str(e):
        # Fall back to default language
        worker = await router.select_worker(language="en")
```

## Testing

### Unit Tests

```bash
# Run routing tests (27 tests)
uv run pytest tests/unit/orchestrator/test_routing.py -v

# Run worker selector tests (22 tests)
uv run pytest tests/unit/orchestrator/test_worker_selector.py -v

# All tests
uv run pytest tests/unit/orchestrator/ -k "routing or selector" -v
```

### Performance Benchmarking

```python
import pytest_benchmark

def test_routing_latency_benchmark(router, sample_workers, benchmark):
    """Benchmark routing decision latency."""

    async def route():
        return await router.select_worker()

    # Run benchmark
    result = benchmark(asyncio.run, route())

    # Verify <1ms p95
    assert benchmark.stats.mean < 0.001  # 1ms
```

### Load Testing

```bash
# Generate 1000 requests with session affinity
for i in {1..1000}; do
    SESSION_ID=$((i % 100))  # 100 unique sessions
    curl -X POST http://localhost:8080/synthesize \
         -H "Session-ID: session-$SESSION_ID" \
         -d '{"text": "Hello world"}'
done

# Check affinity hit rate (should be >95%)
curl http://localhost:8080/metrics | grep affinity_hit_rate
```

## Migration from M2

### M2 Static Routing

```python
# M2: Static routing (still supported)
router = Router(
    registry=registry,
    static_worker_addr="grpc://localhost:7001",
)

# Always routes to static worker
worker = await router.select_worker()
```

### M9 Dynamic Routing

```python
# M9: Dynamic routing with affinity
router = Router(
    registry=registry,
    redis_url="redis://localhost:6379",
    static_worker_addr=None,  # Disable static routing
    affinity_enabled=True,
)

# Routes based on capabilities and affinity
worker = await router.select_worker(
    session_id="session-123",
    model_id="cosyvoice2-en-base",
)
```

### Gradual Migration

1. **Phase 1**: Deploy with static routing (M2 mode)
2. **Phase 2**: Enable dynamic discovery, keep static fallback
3. **Phase 3**: Enable session affinity
4. **Phase 4**: Remove static fallback, full M9

```python
# Phase 2: Dynamic with fallback
router = Router(
    registry=registry,
    redis_url="redis://localhost:6379",
    static_worker_addr="grpc://localhost:7001",  # Fallback
    affinity_enabled=False,  # Not yet
)

# Phase 4: Full M9
router = Router(
    registry=registry,
    redis_url="redis://localhost:6379",
    static_worker_addr=None,  # No fallback
    affinity_enabled=True,  # Full affinity
)
```

## Troubleshooting

### Low Affinity Hit Rate (<95%)

**Symptoms**: `affinity_hit_rate_percent < 95`

**Causes**:
- Session IDs not consistent across requests
- TTL too short (sessions expire before next message)
- Workers frequently restarting (affinity invalidated)

**Solutions**:
```python
# Increase TTL
router = Router(..., affinity_ttl_seconds=7200)  # 2 hours

# Check session ID consistency
logger.info(f"Session ID: {session_id}")  # Same across requests?
```

### High Routing Latency (>1ms)

**Symptoms**: `avg_latency_ms > 1.0`

**Causes**:
- Redis network latency
- Large worker pool (>100 workers)
- Complex capability filtering

**Solutions**:
```python
# Use local Redis
router = Router(redis_url="redis://localhost:6379")

# Reduce worker pool size
# Optimize capability filters
```

### Uneven Load Distribution

**Symptoms**: Some workers overloaded, others idle

**Causes**:
- Round-robin with varying request sizes
- Session affinity with hot sessions
- Health filtering excluding many workers

**Solutions**:
```python
# Switch to least-loaded strategy
router = Router(load_balance_strategy="least_loaded")

# Reduce affinity TTL
router = Router(affinity_ttl_seconds=1800)  # 30 minutes

# Check worker health
workers = await registry.get_workers()
for w in workers:
    print(f"{w.name}: queue={w.metrics['queue_depth']}")
```

## Performance Optimization

### Redis Connection Pooling

```python
# Increase Redis connection pool size for high concurrency
router = Router(
    redis_url="redis://localhost:6379?max_connections=50"
)
```

### Capability Filter Caching

```python
# Cache capability results for repeated queries
from functools import lru_cache

@lru_cache(maxsize=128)
def check_capabilities(worker_name: str, model_id: str) -> bool:
    worker = await registry.get_worker_by_name(worker_name)
    return model_id in worker.resident_models
```

### Parallel Worker Queries

```python
# Query multiple workers in parallel
import asyncio

workers = await registry.get_workers()
tasks = [check_worker_health(w) for w in workers]
results = await asyncio.gather(*tasks)
healthy_workers = [w for w, healthy in zip(workers, results) if healthy]
```

## Future Enhancements

### M10+: Advanced Features

- **Weighted load balancing**: Combine queue depth + latency
- **Geographic routing**: Route to closest worker (latency-aware)
- **Cost-based routing**: Route to cheapest worker (cloud costs)
- **Predictive routing**: Route based on historical performance
- **Circuit breaker**: Temporarily exclude failing workers

### Integration with Metrics

```python
# Export routing metrics to Prometheus
from src.orchestrator.metrics import get_metrics_collector

metrics = get_metrics_collector()
router_metrics = router.get_metrics()

# Record routing decision
metrics.record_routing_decision(
    latency_ms=router_metrics["avg_latency_ms"],
    affinity_hit=True,
)
```

## References

- [M9 Milestone Specification](.claude/modules/milestones.md#m9-routing-v1)
- [Worker Registry Documentation](docs/WORKER_REGISTRY.md)
- [Redis Configuration](docs/REDIS.md)
- [Testing Guide](docs/TESTING_GUIDE.md)

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review [unit tests](../tests/unit/orchestrator/test_routing.py) for examples
3. Check routing logs: `logs/orchestrator.log`
4. File GitHub issue with routing metrics and logs
