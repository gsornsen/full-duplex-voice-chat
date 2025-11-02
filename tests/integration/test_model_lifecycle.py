"""Integration tests for model lifecycle via gRPC.

Tests end-to-end model management through the worker gRPC interface,
including dynamic loading, session management, and eviction.

NOTE: These tests use gRPC and are marked as infrastructure to skip in CI.
Run locally with: pytest -m infrastructure
"""

import asyncio
import uuid
from collections.abc import AsyncGenerator

import pytest
from rpc.generated import tts_pb2
from tts.model_manager import ModelManager
from tts.worker import TTSWorkerServicer

# Mark all tests in this module as infrastructure (skip in CI - uses gRPC)
pytestmark = [pytest.mark.grpc, pytest.mark.infrastructure]


@pytest.fixture
async def model_manager() -> AsyncGenerator[ModelManager, None]:
    """Create a ModelManager for testing."""
    mm = ModelManager(
        default_model_id="mock-440hz",
        warmup_enabled=False,  # Faster tests
        ttl_ms=500,  # Short TTL for tests
        min_residency_ms=100,
        evict_check_interval_ms=100,
        resident_cap=3,
    )
    await mm.initialize()
    yield mm
    await mm.shutdown()


@pytest.fixture
async def servicer(model_manager: ModelManager) -> TTSWorkerServicer:
    """Create a TTSWorkerServicer for testing."""
    return TTSWorkerServicer(model_manager)


@pytest.mark.asyncio
async def test_worker_loads_default_model(
    servicer: TTSWorkerServicer, model_manager: ModelManager
) -> None:
    """Test that worker loads default model on startup."""
    # Check that default model is loaded
    loaded = await model_manager.list_models()
    assert "mock-440hz" in loaded
    assert len(loaded) >= 1


@pytest.mark.asyncio
async def test_worker_preloads_models() -> None:
    """Test that worker preloads specified models."""
    mm = ModelManager(
        default_model_id="default",
        preload_model_ids=["preload-1", "preload-2"],
        warmup_enabled=False,
        evict_check_interval_ms=1000,
    )
    await mm.initialize()

    # Check all models loaded
    loaded = await mm.list_models()
    assert "default" in loaded
    assert "preload-1" in loaded
    assert "preload-2" in loaded

    await mm.shutdown()

@pytest.mark.grpc
@pytest.mark.asyncio
async def test_dynamic_model_load_via_grpc(
    servicer: TTSWorkerServicer, model_manager: ModelManager
) -> None:
    """Test dynamic model loading via LoadModel RPC."""
    # Create LoadModel request
    request = tts_pb2.LoadModelRequest(model_id="new-model")

    # Call LoadModel
    response = await servicer.LoadModel(request, None)

    # Check response
    assert response.success is True
    assert response.load_duration_ms >= 0

    # Verify model is loaded
    loaded = await model_manager.list_models()
    assert "new-model" in loaded

@pytest.mark.grpc_unsafe
@pytest.mark.grpc
@pytest.mark.asyncio
async def test_dynamic_model_unload_via_grpc(
    servicer: TTSWorkerServicer, model_manager: ModelManager
) -> None:
    """Test model unloading via UnloadModel RPC."""
    # Load a model first
    await model_manager.load("test-model")

    # Create UnloadModel request
    request = tts_pb2.UnloadModelRequest(model_id="test-model")

    # Call UnloadModel
    response = await servicer.UnloadModel(request, None)

    # Check response
    assert response.success is True


@pytest.mark.grpc
@pytest.mark.asyncio
async def test_ttl_eviction_end_to_end(
    servicer: TTSWorkerServicer, model_manager: ModelManager
) -> None:
    """Test TTL eviction works end-to-end."""
    # Load a model via LoadModel RPC
    request = tts_pb2.LoadModelRequest(model_id="ttl-test")
    response = await servicer.LoadModel(request, None)
    assert response.success is True

    # Release it
    await model_manager.release("ttl-test")
    await model_manager.release("mock-440hz")  # Release default too

    # Wait for TTL + min_residency + eviction interval
    await asyncio.sleep(1.0)

    # Force eviction
    await model_manager.evict_idle()

    # Model should be evicted
    loaded = await model_manager.list_models()
    assert "ttl-test" not in loaded


@pytest.mark.grpc
@pytest.mark.asyncio
async def test_lru_eviction_end_to_end(
    servicer: TTSWorkerServicer, model_manager: ModelManager
) -> None:
    """Test LRU eviction works end-to-end."""
    # Release default
    await model_manager.release("mock-440hz")

    # Load models up to capacity
    await model_manager.load("model-1")
    await model_manager.load("model-2")
    await model_manager.load("model-3")

    # Release all
    await model_manager.release("model-1")
    await model_manager.release("model-2")
    await model_manager.release("model-3")

    # Load one more (should trigger LRU eviction)
    await model_manager.load("model-4")

    # Should have at most resident_cap models
    loaded = await model_manager.list_models()
    assert len(loaded) <= 3


@pytest.mark.grpc
@pytest.mark.asyncio
async def test_session_uses_correct_model(
    servicer: TTSWorkerServicer, model_manager: ModelManager
) -> None:
    """Test that sessions use the correct model."""
    session_id = str(uuid.uuid4())

    # Start session with specific model
    request = tts_pb2.StartSessionRequest(
        session_id=session_id,
        model_id="session-model",
    )
    response = await servicer.StartSession(request, None)
    assert response.success is True

    # Check that model is loaded
    loaded = await model_manager.list_models()
    assert "session-model" in loaded

    # End session
    end_request = tts_pb2.EndSessionRequest(session_id=session_id)
    await servicer.EndSession(end_request, None)


@pytest.mark.grpc
@pytest.mark.asyncio
async def test_model_switch_between_sessions(
    servicer: TTSWorkerServicer, model_manager: ModelManager
) -> None:
    """Test that different sessions can use different models."""
    session1_id = str(uuid.uuid4())
    session2_id = str(uuid.uuid4())

    # Start session 1 with model A
    request1 = tts_pb2.StartSessionRequest(
        session_id=session1_id,
        model_id="model-a",
    )
    response1 = await servicer.StartSession(request1, None)
    assert response1.success is True

    # Start session 2 with model B
    request2 = tts_pb2.StartSessionRequest(
        session_id=session2_id,
        model_id="model-b",
    )
    response2 = await servicer.StartSession(request2, None)
    assert response2.success is True

    # Both models should be loaded
    loaded = await model_manager.list_models()
    assert "model-a" in loaded
    assert "model-b" in loaded

    # End sessions
    await servicer.EndSession(tts_pb2.EndSessionRequest(session_id=session1_id), None)
    await servicer.EndSession(tts_pb2.EndSessionRequest(session_id=session2_id), None)


@pytest.mark.grpc
@pytest.mark.asyncio
async def test_concurrent_sessions_different_models(
    servicer: TTSWorkerServicer, model_manager: ModelManager
) -> None:
    """Test concurrent sessions with different models."""
    sessions = []

    # Start multiple concurrent sessions with different models
    for i in range(3):
        session_id = str(uuid.uuid4())
        request = tts_pb2.StartSessionRequest(
            session_id=session_id,
            model_id=f"model-{i}",
        )
        response = await servicer.StartSession(request, None)
        assert response.success is True
        sessions.append(session_id)

    # All models should be loaded
    loaded = await model_manager.list_models()
    for i in range(3):
        assert f"model-{i}" in loaded

    # End all sessions
    for session_id in sessions:
        await servicer.EndSession(tts_pb2.EndSessionRequest(session_id=session_id), None)



@pytest.mark.asyncio
async def test_warmup_performance(model_manager: ModelManager) -> None:
    """Test that warmup completes within performance target."""
    import time

    # Create new manager with warmup enabled
    mm = ModelManager(
        default_model_id="warmup-test",
        warmup_enabled=True,
        warmup_text="This is a warmup test.",
        evict_check_interval_ms=1000,
    )

    start_time = time.time()
    await mm.initialize()
    warmup_duration = time.time() - start_time

    # Warmup should complete quickly (< 500ms for mock adapter)
    assert warmup_duration < 0.5

    await mm.shutdown()

@pytest.mark.grpc
@pytest.mark.asyncio
async def test_list_models_via_grpc(
    servicer: TTSWorkerServicer, model_manager: ModelManager
) -> None:
    """Test ListModels RPC returns correct model list."""
    # Load some models
    await model_manager.load("list-test-1")
    await model_manager.load("list-test-2")

    # Call ListModels
    request = tts_pb2.ListModelsRequest()
    response = await servicer.ListModels(request, None)

    # Check response
    assert len(response.models) >= 3  # At least default + 2 loaded
    model_ids = [m.model_id for m in response.models]
    assert "list-test-1" in model_ids
    assert "list-test-2" in model_ids


@pytest.mark.grpc
@pytest.mark.asyncio
async def test_get_capabilities_includes_resident_models(
    servicer: TTSWorkerServicer, model_manager: ModelManager
) -> None:
    """Test GetCapabilities RPC includes resident models."""
    # Load some models
    await model_manager.load("cap-test-1")
    await model_manager.load("cap-test-2")

    # Call GetCapabilities
    request = tts_pb2.GetCapabilitiesRequest()
    response = await servicer.GetCapabilities(request, None)

    # Check resident models
    assert "cap-test-1" in response.resident_models
    assert "cap-test-2" in response.resident_models


@pytest.mark.grpc
@pytest.mark.asyncio
async def test_session_increments_model_refcount(
    servicer: TTSWorkerServicer, model_manager: ModelManager
) -> None:
    """Test that starting a session increments model refcount."""
    session_id = str(uuid.uuid4())

    # Get initial refcount
    info_before = await model_manager.get_model_info()
    initial_refcount = info_before.get("refcount-test", {}).get("refcount", 0)

    # Start session
    request = tts_pb2.StartSessionRequest(
        session_id=session_id,
        model_id="refcount-test",
    )
    await servicer.StartSession(request, None)

    # Get new refcount
    info_after = await model_manager.get_model_info()
    new_refcount = info_after["refcount-test"]["refcount"]

    # Refcount should be incremented
    assert new_refcount == initial_refcount + 1

    # End session
    await servicer.EndSession(tts_pb2.EndSessionRequest(session_id=session_id), None)


@pytest.mark.grpc
@pytest.mark.asyncio
async def test_session_end_decrements_model_refcount(
    servicer: TTSWorkerServicer, model_manager: ModelManager
) -> None:
    """Test that ending a session decrements model refcount."""
    session_id = str(uuid.uuid4())

    # Start session
    request = tts_pb2.StartSessionRequest(
        session_id=session_id,
        model_id="refcount-test-2",
    )
    await servicer.StartSession(request, None)

    # Get refcount after start
    info_after_start = await model_manager.get_model_info()
    refcount_after_start = info_after_start["refcount-test-2"]["refcount"]

    # End session
    await servicer.EndSession(tts_pb2.EndSessionRequest(session_id=session_id), None)

    # Get refcount after end
    info_after_end = await model_manager.get_model_info()
    refcount_after_end = info_after_end.get("refcount-test-2", {}).get("refcount", 0)

    # Refcount should be decremented
    assert refcount_after_end == refcount_after_start - 1


@pytest.mark.grpc
@pytest.mark.asyncio
async def test_multiple_sessions_same_model_share_instance(
    servicer: TTSWorkerServicer, model_manager: ModelManager
) -> None:
    """Test that multiple sessions using same model share one instance."""
    session1_id = str(uuid.uuid4())
    session2_id = str(uuid.uuid4())

    # Start two sessions with same model
    await servicer.StartSession(
        tts_pb2.StartSessionRequest(session_id=session1_id, model_id="shared-model"), None
    )
    await servicer.StartSession(
        tts_pb2.StartSessionRequest(session_id=session2_id, model_id="shared-model"), None
    )

    # Should only have one instance of the model
    loaded = await model_manager.list_models()
    shared_count = sum(1 for m in loaded if m == "shared-model")
    assert shared_count == 1

    # Refcount should be 2
    info = await model_manager.get_model_info()
    assert info["shared-model"]["refcount"] == 2

    # End sessions
    await servicer.EndSession(tts_pb2.EndSessionRequest(session_id=session1_id), None)
    await servicer.EndSession(tts_pb2.EndSessionRequest(session_id=session2_id), None)
