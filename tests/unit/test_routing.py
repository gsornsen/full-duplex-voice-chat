"""Unit tests for worker routing logic.

Tests static routing, capability matching, load balancing, and graceful degradation.
"""

import time
from unittest.mock import AsyncMock

import pytest

from src.orchestrator.registry import WorkerRegistration, WorkerRegistry
from src.orchestrator.routing import Router


# Test data fixtures
@pytest.fixture
def mock_worker_cpu() -> WorkerRegistration:
    """Create mock CPU worker registration."""
    return WorkerRegistration(
        name="tts-piper@0",
        addr="grpc://localhost:7001",
        capabilities={
            "streaming": True,
            "zero_shot": False,
            "lora": False,
            "cpu_ok": True,
            "languages": ["en"],
            "emotive_zero_prompt": False,
        },
        resident_models=["piper-en-us"],
        metrics={"rtf": 0.8, "queue_depth": 2},
        last_heartbeat_ts=time.time(),
    )


@pytest.fixture
def mock_worker_gpu_en() -> WorkerRegistration:
    """Create mock GPU worker (English only)."""
    return WorkerRegistration(
        name="tts-xtts@0",
        addr="grpc://localhost:7002",
        capabilities={
            "streaming": True,
            "zero_shot": True,
            "lora": False,
            "cpu_ok": False,
            "languages": ["en"],
            "emotive_zero_prompt": True,
        },
        resident_models=["xtts-v2-en"],
        metrics={"rtf": 0.3, "queue_depth": 1},
        last_heartbeat_ts=time.time(),
    )


@pytest.fixture
def mock_worker_gpu_multilang() -> WorkerRegistration:
    """Create mock GPU worker (multilingual)."""
    return WorkerRegistration(
        name="tts-cosyvoice2@0",
        addr="grpc://localhost:7003",
        capabilities={
            "streaming": True,
            "zero_shot": True,
            "lora": False,
            "cpu_ok": False,
            "languages": ["en", "zh", "ja"],
            "emotive_zero_prompt": True,
        },
        resident_models=["cosyvoice2-multilang"],
        metrics={"rtf": 0.2, "queue_depth": 0},
        last_heartbeat_ts=time.time(),
    )


@pytest.fixture
def mock_worker_lora() -> WorkerRegistration:
    """Create mock LoRA-capable worker."""
    return WorkerRegistration(
        name="tts-unsloth-sesame@0",
        addr="grpc://localhost:7004",
        capabilities={
            "streaming": True,
            "zero_shot": False,
            "lora": True,
            "cpu_ok": False,
            "languages": ["en"],
            "emotive_zero_prompt": False,
        },
        resident_models=["sesame-lora-v1"],
        metrics={"rtf": 0.4, "queue_depth": 3},
        last_heartbeat_ts=time.time(),
    )


@pytest.fixture
def mock_registry() -> AsyncMock:
    """Create mock worker registry."""
    registry = AsyncMock(spec=WorkerRegistry)
    registry.health_check = AsyncMock(return_value=True)
    registry.get_workers = AsyncMock(return_value=[])
    registry.get_worker_by_name = AsyncMock(return_value=None)
    return registry


class TestRouterStaticRouting:
    """Test static routing behavior (M2 scope)."""

    async def test_static_routing_configured(self, mock_registry: AsyncMock) -> None:
        """Test static routing when static_worker_addr is configured."""
        router = Router(
            registry=mock_registry,
            static_worker_addr="grpc://static-worker:7001",
        )

        addr = await router.select_worker()

        # Should return static address without querying registry
        assert addr == "grpc://static-worker:7001"
        mock_registry.get_workers.assert_not_awaited()

    async def test_static_routing_with_params_ignored(
        self, mock_registry: AsyncMock
    ) -> None:
        """Test that static routing ignores capability parameters."""
        router = Router(
            registry=mock_registry,
            static_worker_addr="grpc://static-worker:7001",
        )

        addr = await router.select_worker(
            language="zh",
            model_id="some-model",
            capabilities={"streaming": True},
        )

        # Should still return static address (params ignored in M2)
        assert addr == "grpc://static-worker:7001"

    async def test_static_routing_health_check_always_healthy(
        self, mock_registry: AsyncMock
    ) -> None:
        """Test that static routing health check always returns True."""
        router = Router(
            registry=mock_registry,
            static_worker_addr="grpc://static-worker:7001",
        )

        # Even if registry is unhealthy
        mock_registry.health_check.return_value = False

        is_healthy = await router.health_check()
        assert is_healthy is True


class TestRouterDynamicDiscovery:
    """Test dynamic worker discovery (M2+ fallback)."""

    async def test_dynamic_discovery_single_worker(
        self,
        mock_registry: AsyncMock,
        mock_worker_cpu: WorkerRegistration,
    ) -> None:
        """Test dynamic discovery with single worker."""
        mock_registry.get_workers.return_value = [mock_worker_cpu]

        router = Router(
            registry=mock_registry,
            static_worker_addr=None,  # No static routing
        )

        addr = await router.select_worker()

        # Should return first (only) worker
        assert addr == "grpc://localhost:7001"
        mock_registry.get_workers.assert_awaited_once()

    async def test_dynamic_discovery_multiple_workers_returns_least_loaded(
        self,
        mock_registry: AsyncMock,
        mock_worker_cpu: WorkerRegistration,
        mock_worker_gpu_en: WorkerRegistration,
    ) -> None:
        """Test dynamic discovery with multiple workers uses least_loaded by default."""
        # mock_worker_cpu has queue_depth=2
        # mock_worker_gpu_en has queue_depth=1 (should be selected)
        mock_registry.get_workers.return_value = [
            mock_worker_cpu,
            mock_worker_gpu_en,
        ]

        router = Router(
            registry=mock_registry,
            static_worker_addr=None,
        )

        addr = await router.select_worker()

        # Default behavior: select least loaded worker (lowest queue depth)
        assert addr == "grpc://localhost:7002"  # gpu_en has queue_depth=1

    async def test_dynamic_discovery_no_workers(
        self, mock_registry: AsyncMock
    ) -> None:
        """Test dynamic discovery when no workers available."""
        mock_registry.get_workers.return_value = []

        router = Router(
            registry=mock_registry,
            static_worker_addr=None,
        )

        with pytest.raises(
            RuntimeError,
            match="No workers available in registry",
        ):
            await router.select_worker()

    async def test_dynamic_discovery_redis_unavailable_no_fallback(
        self, mock_registry: AsyncMock
    ) -> None:
        """Test dynamic discovery when Redis unavailable and no static fallback."""
        mock_registry.get_workers.side_effect = ConnectionError("Redis down")

        router = Router(
            registry=mock_registry,
            static_worker_addr=None,
        )

        with pytest.raises(
            RuntimeError, match="Dynamic worker selection failed"
        ):
            await router.select_worker()

    async def test_dynamic_discovery_redis_unavailable_with_fallback(
        self, mock_registry: AsyncMock
    ) -> None:
        """Test graceful degradation to static worker when Redis unavailable."""
        mock_registry.get_workers.side_effect = ConnectionError("Redis down")

        router = Router(
            registry=mock_registry,
            static_worker_addr="grpc://fallback:7001",
        )

        addr = await router.select_worker()

        # Should fall back to static address
        assert addr == "grpc://fallback:7001"

    async def test_health_check_redis_healthy(self, mock_registry: AsyncMock) -> None:
        """Test health check when Redis is healthy."""
        mock_registry.health_check.return_value = True

        router = Router(
            registry=mock_registry,
            static_worker_addr=None,
        )

        is_healthy = await router.health_check()
        assert is_healthy is True

    async def test_health_check_redis_unhealthy(
        self, mock_registry: AsyncMock
    ) -> None:
        """Test health check when Redis is unhealthy."""
        mock_registry.health_check.return_value = False

        router = Router(
            registry=mock_registry,
            static_worker_addr=None,
        )

        is_healthy = await router.health_check()
        assert is_healthy is False


class TestRouterDynamicRouting:
    """Test dynamic routing with capability matching (M9+ scope)."""

    async def test_filter_by_language(
        self,
        mock_registry: AsyncMock,
        mock_worker_cpu: WorkerRegistration,
        mock_worker_gpu_en: WorkerRegistration,
        mock_worker_gpu_multilang: WorkerRegistration,
    ) -> None:
        """Test filtering workers by language capability."""
        mock_registry.get_workers.return_value = [
            mock_worker_cpu,  # en only
            mock_worker_gpu_en,  # en only
            mock_worker_gpu_multilang,  # en, zh, ja
        ]

        router = Router(registry=mock_registry, static_worker_addr=None)

        # Request Chinese
        worker = await router.select_worker_dynamic(language="zh")

        # Should select only multilang worker
        assert worker.name == "tts-cosyvoice2@0"

    async def test_filter_by_language_no_match(
        self,
        mock_registry: AsyncMock,
        mock_worker_cpu: WorkerRegistration,
    ) -> None:
        """Test filtering by language when no worker supports it."""
        mock_registry.get_workers.return_value = [mock_worker_cpu]  # en only

        router = Router(registry=mock_registry, static_worker_addr=None)

        with pytest.raises(RuntimeError, match="No workers support language 'zh'"):
            await router.select_worker_dynamic(language="zh")

    async def test_filter_by_capabilities_streaming(
        self,
        mock_registry: AsyncMock,
        mock_worker_cpu: WorkerRegistration,
        mock_worker_gpu_en: WorkerRegistration,
    ) -> None:
        """Test filtering by boolean capabilities."""
        mock_registry.get_workers.return_value = [
            mock_worker_cpu,
            mock_worker_gpu_en,
        ]

        router = Router(registry=mock_registry, static_worker_addr=None)

        # Request zero-shot capability
        worker = await router.select_worker_dynamic(
            capabilities={"zero_shot": True}
        )

        # Should select GPU worker (has zero-shot)
        assert worker.name == "tts-xtts@0"

    async def test_filter_by_capabilities_multiple(
        self,
        mock_registry: AsyncMock,
        mock_worker_cpu: WorkerRegistration,
        mock_worker_lora: WorkerRegistration,
    ) -> None:
        """Test filtering by multiple capabilities."""
        mock_registry.get_workers.return_value = [
            mock_worker_cpu,
            mock_worker_lora,
        ]

        router = Router(registry=mock_registry, static_worker_addr=None)

        # Request LoRA + streaming
        worker = await router.select_worker_dynamic(
            capabilities={"lora": True, "streaming": True}
        )

        # Should select LoRA worker
        assert worker.name == "tts-unsloth-sesame@0"

    async def test_filter_by_capabilities_no_match(
        self,
        mock_registry: AsyncMock,
        mock_worker_cpu: WorkerRegistration,
    ) -> None:
        """Test filtering when no worker matches capabilities."""
        mock_registry.get_workers.return_value = [mock_worker_cpu]

        router = Router(registry=mock_registry, static_worker_addr=None)

        with pytest.raises(RuntimeError, match="No workers match required capabilities"):
            await router.select_worker_dynamic(capabilities={"lora": True})

    async def test_prefer_resident_models(
        self,
        mock_registry: AsyncMock,
        mock_worker_gpu_en: WorkerRegistration,
        mock_worker_gpu_multilang: WorkerRegistration,
    ) -> None:
        """Test preference for workers with resident model."""
        # Both workers support English
        mock_registry.get_workers.return_value = [
            mock_worker_gpu_en,  # resident: xtts-v2-en
            mock_worker_gpu_multilang,  # resident: cosyvoice2-multilang
        ]

        router = Router(
            registry=mock_registry,
            static_worker_addr=None,
            prefer_resident_models=True,
        )

        # Request specific model
        worker = await router.select_worker_dynamic(
            language="en", model_id="xtts-v2-en"
        )

        # Should select worker with resident model
        assert worker.name == "tts-xtts@0"

    async def test_prefer_resident_models_fallback(
        self,
        mock_registry: AsyncMock,
        mock_worker_gpu_en: WorkerRegistration,
        mock_worker_gpu_multilang: WorkerRegistration,
    ) -> None:
        """Test fallback when requested model not resident anywhere."""
        mock_registry.get_workers.return_value = [
            mock_worker_gpu_en,
            mock_worker_gpu_multilang,
        ]

        router = Router(
            registry=mock_registry,
            static_worker_addr=None,
            prefer_resident_models=True,
        )

        # Request model not resident on any worker
        worker = await router.select_worker_dynamic(
            language="en", model_id="unknown-model"
        )

        # Should still return a worker (lowest queue depth by default)
        assert worker.name == "tts-cosyvoice2@0"  # queue_depth=0

    async def test_no_workers_available(self, mock_registry: AsyncMock) -> None:
        """Test dynamic routing when no workers available."""
        mock_registry.get_workers.return_value = []

        router = Router(registry=mock_registry, static_worker_addr=None)

        with pytest.raises(RuntimeError, match="No workers available in registry"):
            await router.select_worker_dynamic()


class TestRouterLoadBalancing:
    """Test load balancing strategies (M9+ scope)."""

    async def test_load_balance_queue_depth(
        self,
        mock_registry: AsyncMock,
        mock_worker_cpu: WorkerRegistration,
        mock_worker_gpu_en: WorkerRegistration,
        mock_worker_gpu_multilang: WorkerRegistration,
    ) -> None:
        """Test queue depth load balancing."""
        # queue_depth: cpu=2, gpu_en=1, gpu_multi=0
        mock_registry.get_workers.return_value = [
            mock_worker_cpu,
            mock_worker_gpu_en,
            mock_worker_gpu_multilang,
        ]

        router = Router(
            registry=mock_registry,
            static_worker_addr=None,
            load_balance_strategy="least_loaded",  # Fixed: was "queue_depth"
        )

        worker = await router.select_worker_dynamic()

        # Should select worker with lowest queue depth
        assert worker.name == "tts-cosyvoice2@0"  # queue_depth=0

    async def test_load_balance_latency(
        self,
        mock_registry: AsyncMock,
        mock_worker_cpu: WorkerRegistration,
        mock_worker_gpu_en: WorkerRegistration,
        mock_worker_gpu_multilang: WorkerRegistration,
    ) -> None:
        """Test latency (RTF) load balancing."""
        # rtf: cpu=0.8, gpu_en=0.3, gpu_multi=0.2
        mock_registry.get_workers.return_value = [
            mock_worker_cpu,
            mock_worker_gpu_en,
            mock_worker_gpu_multilang,
        ]

        router = Router(
            registry=mock_registry,
            static_worker_addr=None,
            load_balance_strategy="least_latency",  # Fixed: was "latency"
        )

        worker = await router.select_worker_dynamic()

        # Should select worker with best (lowest) RTF
        assert worker.name == "tts-cosyvoice2@0"  # rtf=0.2

    async def test_load_balance_round_robin(
        self,
        mock_registry: AsyncMock,
        mock_worker_cpu: WorkerRegistration,
        mock_worker_gpu_en: WorkerRegistration,
        mock_worker_gpu_multilang: WorkerRegistration,
    ) -> None:
        """Test round-robin load balancing."""
        workers = [
            mock_worker_cpu,
            mock_worker_gpu_en,
            mock_worker_gpu_multilang,
        ]
        mock_registry.get_workers.return_value = workers

        router = Router(
            registry=mock_registry,
            static_worker_addr=None,
            load_balance_strategy="round_robin",
        )

        # Make multiple selections
        selected_workers = []
        for _ in range(5):
            worker = await router.select_worker_dynamic()
            selected_workers.append(worker.name)

        # Should cycle through workers
        expected = [
            "tts-piper@0",  # index 0
            "tts-xtts@0",  # index 1
            "tts-cosyvoice2@0",  # index 2
            "tts-piper@0",  # index 0 again
            "tts-xtts@0",  # index 1 again
        ]
        assert selected_workers == expected

    async def test_load_balance_unknown_strategy(
        self,
        mock_registry: AsyncMock,
        mock_worker_cpu: WorkerRegistration,
        mock_worker_gpu_en: WorkerRegistration,
    ) -> None:
        """Test fallback behavior for unknown load balancing strategy."""
        # mock_worker_cpu has queue_depth=2
        # mock_worker_gpu_en has queue_depth=1 (should be selected)
        mock_registry.get_workers.return_value = [
            mock_worker_cpu,
            mock_worker_gpu_en,
        ]

        router = Router(
            registry=mock_registry,
            static_worker_addr=None,
            load_balance_strategy="invalid_strategy",
        )

        worker = await router.select_worker_dynamic()

        # Should fall back to LEAST_LOADED strategy and select least loaded worker
        assert worker.name == "tts-xtts@0"  # queue_depth=1 (lower than cpu's 2)

    async def test_load_balance_single_worker(
        self,
        mock_registry: AsyncMock,
        mock_worker_cpu: WorkerRegistration,
    ) -> None:
        """Test load balancing with single worker."""
        mock_registry.get_workers.return_value = [mock_worker_cpu]

        router = Router(
            registry=mock_registry,
            static_worker_addr=None,
            load_balance_strategy="least_loaded",  # Fixed: was "queue_depth"
        )

        worker = await router.select_worker_dynamic()

        # Should return the only worker
        assert worker.name == "tts-piper@0"


class TestRouterUtilityMethods:
    """Test utility methods on Router."""

    async def test_get_worker_info_found(
        self,
        mock_registry: AsyncMock,
        mock_worker_cpu: WorkerRegistration,
    ) -> None:
        """Test retrieving worker info when found."""
        mock_registry.get_worker_by_name.return_value = mock_worker_cpu

        router = Router(registry=mock_registry)

        worker = await router.get_worker_info("tts-piper@0")

        assert worker is not None
        assert worker.name == "tts-piper@0"
        mock_registry.get_worker_by_name.assert_awaited_once_with("tts-piper@0")

    async def test_get_worker_info_not_found(
        self, mock_registry: AsyncMock
    ) -> None:
        """Test retrieving worker info when not found."""
        mock_registry.get_worker_by_name.return_value = None

        router = Router(registry=mock_registry)

        worker = await router.get_worker_info("nonexistent")

        assert worker is None

    async def test_get_worker_info_connection_error(
        self, mock_registry: AsyncMock
    ) -> None:
        """Test retrieving worker info when connection error occurs."""
        mock_registry.get_worker_by_name.side_effect = ConnectionError("Redis down")

        router = Router(registry=mock_registry)

        worker = await router.get_worker_info("test-worker")

        # Should return None on error
        assert worker is None


class TestRouterConfiguration:
    """Test router configuration and initialization."""

    def test_router_initialization_defaults(
        self, mock_registry: AsyncMock
    ) -> None:
        """Test router initialization with defaults."""
        router = Router(registry=mock_registry)

        assert router.registry is mock_registry
        assert router.static_worker_addr is None
        assert router.prefer_resident_models is True
        assert router.load_balance_strategy.value == "least_loaded"  # Fixed: was "queue_depth"
        assert router._round_robin_index == 0

    def test_router_initialization_custom(
        self, mock_registry: AsyncMock
    ) -> None:
        """Test router initialization with custom configuration."""
        router = Router(
            registry=mock_registry,
            static_worker_addr="grpc://custom:7001",
            prefer_resident_models=False,
            load_balance_strategy="least_latency",  # Fixed: was "latency"
        )

        assert router.static_worker_addr == "grpc://custom:7001"
        assert router.prefer_resident_models is False
        assert router.load_balance_strategy.value == "least_latency"  # Fixed: was "latency"
