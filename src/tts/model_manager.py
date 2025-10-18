"""Model lifecycle management with load/unload, TTL eviction, and LRU caching.

This module implements the ModelManager class which handles all TTS model lifecycle
operations including loading, unloading, reference counting, TTL-based eviction,
and LRU-based capacity management.

Adapter imports are lazy to support container isolation (e.g., CosyVoice container
doesn't need Piper dependencies).
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

# Only import MockTTSAdapter at module level (no external dependencies)
from src.tts.adapters.adapter_mock import MockTTSAdapter

logger = logging.getLogger(__name__)


class ModelManagerError(Exception):
    """Base exception for model manager errors."""

    pass


class ModelNotFoundError(ModelManagerError):
    """Raised when a requested model is not found."""

    pass


class ModelAlreadyLoadedError(ModelManagerError):
    """Raised when attempting to load an already loaded model."""

    pass


class ModelManager:
    """Manages TTS model lifecycle including loading, unloading, and eviction.

    The ModelManager ensures efficient use of GPU memory by:
    - Loading default and preload models on startup
    - Warming up models with synthetic utterances
    - Reference counting to prevent unload during active use
    - TTL-based eviction for idle models
    - LRU-based eviction when resident capacity is exceeded
    - Thread-safe operations with semaphore control

    Adapter imports are lazy (inside _load_model_impl) to support container
    isolation. Each TTS worker container only needs its specific adapter
    dependencies installed.

    Attributes:
        default_model_id: Required default model loaded on startup
        preload_model_ids: Optional list of models to preload
        ttl_ms: Time-to-live for idle models in milliseconds
        min_residency_ms: Minimum residency time before eviction eligible
        resident_cap: Maximum number of resident models
        max_parallel_loads: Maximum parallel model loads (prevents OOM)
        warmup_enabled: Whether to warmup models on load
        warmup_text: Text to use for warmup synthesis
        loaded_models: Dictionary of model_id -> model instance
        refcounts: Dictionary of model_id -> reference count
        last_used_ts: Dictionary of model_id -> last use timestamp
        load_ts: Dictionary of model_id -> load timestamp
        load_semaphore: Semaphore limiting parallel loads
        eviction_task: Background task handle for eviction loop
    """

    def __init__(
        self,
        default_model_id: str,
        preload_model_ids: list[str] | None = None,
        ttl_ms: int = 600000,
        min_residency_ms: int = 120000,
        resident_cap: int = 3,
        max_parallel_loads: int = 1,
        warmup_enabled: bool = True,
        warmup_text: str = "This is a warmup test.",
        evict_check_interval_ms: int = 30000,
    ) -> None:
        """Initialize model manager.

        Args:
            default_model_id: Required default model to load on startup
            preload_model_ids: Optional list of models to preload
            ttl_ms: Time-to-live for idle models in milliseconds
            min_residency_ms: Minimum residency time before eviction eligible
            resident_cap: Maximum number of resident models
            max_parallel_loads: Maximum parallel model loads
            warmup_enabled: Whether to warmup models on load
            warmup_text: Text to use for warmup synthesis
            evict_check_interval_ms: How often to run eviction check
        """
        self.default_model_id = default_model_id
        self.preload_model_ids = preload_model_ids or []
        self.ttl_ms = ttl_ms
        self.min_residency_ms = min_residency_ms
        self.resident_cap = resident_cap
        self.max_parallel_loads = max_parallel_loads
        self.warmup_enabled = warmup_enabled
        self.warmup_text = warmup_text
        self.evict_check_interval_ms = evict_check_interval_ms

        # Internal state
        self.loaded_models: dict[str, Any] = {}
        self.refcounts: dict[str, int] = {}
        self.last_used_ts: dict[str, float] = {}
        self.load_ts: dict[str, float] = {}

        # Synchronization
        self.load_semaphore = asyncio.Semaphore(max_parallel_loads)
        self.state_lock = asyncio.Lock()

        # Background eviction task
        self.eviction_task: asyncio.Task[None] | None = None
        self._shutdown = False

        logger.info(
            "ModelManager initialized",
            extra={
                "default_model": default_model_id,
                "preload_models": self.preload_model_ids,
                "ttl_ms": ttl_ms,
                "resident_cap": resident_cap,
                "warmup_enabled": warmup_enabled,
            },
        )

    async def initialize(self) -> None:
        """Initialize the model manager by loading default and preload models.

        This should be called once after construction to:
        1. Load the default model (required)
        2. Load any preload models (optional)
        3. Warmup all loaded models
        4. Start the eviction background task

        Raises:
            ModelManagerError: If default model fails to load
        """
        logger.info("Initializing ModelManager")

        # Load default model
        try:
            await self.load(self.default_model_id)
            logger.info(
                "Default model loaded successfully",
                extra={"model_id": self.default_model_id},
            )
        except Exception as e:
            logger.error(
                "Failed to load default model",
                extra={"model_id": self.default_model_id, "error": str(e)},
            )
            raise ModelManagerError(f"Failed to load default model: {e}") from e

        # Load preload models
        for model_id in self.preload_model_ids:
            try:
                await self.load(model_id)
                logger.info("Preload model loaded", extra={"model_id": model_id})
            except Exception as e:
                logger.warning(
                    "Failed to load preload model (continuing)",
                    extra={"model_id": model_id, "error": str(e)},
                )

        # Start eviction task
        self.eviction_task = asyncio.create_task(self._eviction_loop())
        logger.info("ModelManager initialization complete")

    async def shutdown(self) -> None:
        """Shutdown the model manager and clean up resources.

        Stops the eviction task and unloads all models.
        """
        logger.info("Shutting down ModelManager")
        self._shutdown = True

        # Cancel eviction task
        if self.eviction_task:
            self.eviction_task.cancel()
            try:
                await self.eviction_task
            except asyncio.CancelledError:
                pass

        # Unload all models
        async with self.state_lock:
            model_ids = list(self.loaded_models.keys())
            for model_id in model_ids:
                try:
                    await self._unload_model(model_id)
                except Exception as e:
                    logger.error(
                        "Error unloading model during shutdown",
                        extra={"model_id": model_id, "error": str(e)},
                    )

        logger.info("ModelManager shutdown complete")

    async def load(self, model_id: str) -> Any:
        """Load model and increment reference count.

        If the model is already loaded, increments the reference count and
        returns the existing instance. Otherwise, loads the model using the
        semaphore to limit parallel loads.

        Args:
            model_id: Model identifier

        Returns:
            Model instance

        Raises:
            ModelManagerError: If model load fails
        """
        async with self.state_lock:
            # Check if already loaded
            if model_id in self.loaded_models:
                self.refcounts[model_id] += 1
                self.last_used_ts[model_id] = time.time()
                logger.info(
                    "Model already loaded, incrementing refcount",
                    extra={"model_id": model_id, "refcount": self.refcounts[model_id]},
                )
                return self.loaded_models[model_id]

            # Initialize refcount
            self.refcounts[model_id] = 1

        # Load model (outside state lock, but with semaphore)
        async with self.load_semaphore:
            try:
                start_time = time.time()
                model = await self._load_model_impl(model_id)
                load_duration_ms = (time.time() - start_time) * 1000

                # Warmup if enabled
                if self.warmup_enabled:
                    await self._warmup_model(model, model_id)

                # Update state
                async with self.state_lock:
                    self.loaded_models[model_id] = model
                    self.load_ts[model_id] = time.time()
                    self.last_used_ts[model_id] = time.time()

                    # Enforce resident cap (LRU eviction)
                    await self._enforce_resident_cap()

                logger.info(
                    "Model loaded successfully",
                    extra={
                        "model_id": model_id,
                        "load_duration_ms": load_duration_ms,
                        "refcount": self.refcounts[model_id],
                        "resident_count": len(self.loaded_models),
                    },
                )
                return model

            except Exception as e:
                # Cleanup refcount on failure
                async with self.state_lock:
                    if model_id in self.refcounts:
                        del self.refcounts[model_id]

                logger.error(
                    "Failed to load model",
                    extra={"model_id": model_id, "error": str(e)},
                )
                raise ModelManagerError(f"Failed to load model {model_id}: {e}") from e

    async def release(self, model_id: str) -> None:
        """Release model and decrement reference count.

        Decrements the reference count for the model and updates the last used
        timestamp. The model remains loaded but becomes eligible for eviction
        when the reference count reaches zero.

        Args:
            model_id: Model identifier

        Raises:
            ModelNotFoundError: If model is not loaded
            ModelManagerError: If refcount goes negative
        """
        async with self.state_lock:
            if model_id not in self.loaded_models:
                raise ModelNotFoundError(f"Model {model_id} is not loaded")

            if model_id not in self.refcounts:
                raise ModelManagerError(f"Model {model_id} has no refcount entry")

            self.refcounts[model_id] -= 1
            self.last_used_ts[model_id] = time.time()

            if self.refcounts[model_id] < 0:
                raise ModelManagerError(
                    f"Model {model_id} refcount went negative: {self.refcounts[model_id]}"
                )

            logger.info(
                "Model reference released",
                extra={"model_id": model_id, "refcount": self.refcounts[model_id]},
            )

    async def evict_idle(self) -> None:
        """Evict idle models based on TTL and LRU policy.

        This is called by the background eviction task periodically. It:
        1. Checks for models idle longer than TTL
        2. Respects min_residency_ms before eviction
        3. Never evicts models with refcount > 0
        """
        now = time.time()
        ttl_seconds = self.ttl_ms / 1000.0
        min_residency_seconds = self.min_residency_ms / 1000.0

        async with self.state_lock:
            model_ids = list(self.loaded_models.keys())

            for model_id in model_ids:
                # Skip if in use
                if self.refcounts.get(model_id, 0) > 0:
                    continue

                # Skip if not loaded long enough
                load_time = self.load_ts.get(model_id, now)
                if (now - load_time) < min_residency_seconds:
                    logger.debug(
                        "Model not resident long enough for eviction",
                        extra={
                            "model_id": model_id,
                            "residency_seconds": now - load_time,
                            "min_residency_seconds": min_residency_seconds,
                        },
                    )
                    continue

                # Check TTL
                last_used = self.last_used_ts.get(model_id, now)
                idle_time = now - last_used

                if idle_time >= ttl_seconds:
                    logger.info(
                        "Evicting idle model (TTL)",
                        extra={
                            "model_id": model_id,
                            "idle_seconds": idle_time,
                            "ttl_seconds": ttl_seconds,
                        },
                    )
                    await self._unload_model(model_id)

    async def list_models(self) -> list[str]:
        """List all loaded models.

        Returns:
            List of model IDs currently loaded
        """
        async with self.state_lock:
            return list(self.loaded_models.keys())

    async def get_model_info(self) -> dict[str, dict[str, Any]]:
        """Get detailed information about all loaded models.

        Returns:
            Dictionary of model_id -> info dict with refcount, last_used, etc.
        """
        async with self.state_lock:
            now = time.time()
            info = {}
            for model_id in self.loaded_models:
                info[model_id] = {
                    "refcount": self.refcounts.get(model_id, 0),
                    "last_used_ts": self.last_used_ts.get(model_id, 0),
                    "load_ts": self.load_ts.get(model_id, 0),
                    "idle_seconds": now - self.last_used_ts.get(model_id, now),
                    "resident_seconds": now - self.load_ts.get(model_id, now),
                }
            return info

    async def _load_model_impl(self, model_id: str) -> Any:
        """Internal implementation of model loading with lazy adapter imports.

        Routes to the appropriate adapter based on model_id prefix:
        - "piper-*": PiperTTSAdapter (M5) - lazy import to avoid piper-tts dependency
        - "cosyvoice2-*": CosyVoiceAdapter (M6) - lazy import for PyTorch 2.3.1 isolation
        - Others: MockTTSAdapter (M1-M4) - imported at module level (no dependencies)

        Lazy imports enable container isolation: each TTS worker container only needs
        its specific adapter's dependencies installed. For example, the CosyVoice
        container (PyTorch 2.3.1 + CUDA 12.1) doesn't need piper-tts package.

        Future milestones (M7-M8) will add more adapter types with lazy imports.

        Args:
            model_id: Model identifier

        Returns:
            Model instance

        Raises:
            ModelNotFoundError: If voicepack not found
        """
        logger.info("Loading model implementation", extra={"model_id": model_id})

        # Route to Piper adapter for piper-* models (M5)
        # Lazy import to avoid requiring piper-tts in all containers
        if model_id.startswith("piper-"):
            from src.tts.adapters.adapter_piper import PiperTTSAdapter

            # Extract voice name from model_id
            # (e.g., "piper-en-us-lessac-medium" -> "en-us-lessac-medium")
            voice_name = model_id.replace("piper-", "", 1)
            voicepack_path = Path(f"voicepacks/piper/{voice_name}")

            if not voicepack_path.exists():
                raise ModelNotFoundError(
                    f"Voicepack not found for model {model_id}: {voicepack_path}"
                )

            piper_adapter: Any = PiperTTSAdapter(model_id=model_id, model_path=voicepack_path)
            return piper_adapter

        # Route to CosyVoice 2 adapter for cosyvoice2-* models (M6)
        # Lazy import to support PyTorch 2.3.1 isolation in CosyVoice container
        if model_id.startswith("cosyvoice2-"):
            from src.tts.adapters.adapter_cosyvoice import CosyVoiceAdapter

            # Extract voice name from model_id
            # (e.g., "cosyvoice2-en-base" -> "en-base")
            voice_name = model_id.replace("cosyvoice2-", "", 1)
            voicepack_path = Path(f"voicepacks/cosyvoice/{voice_name}")

            if not voicepack_path.exists():
                raise ModelNotFoundError(
                    f"Voicepack not found for model {model_id}: {voicepack_path}"
                )

            cosyvoice_adapter: Any = CosyVoiceAdapter(
                model_id=model_id, model_path=voicepack_path
            )
            return cosyvoice_adapter

        # Default to mock adapter for testing (imported at module level)
        mock_adapter: Any = MockTTSAdapter(model_id=model_id)
        return mock_adapter

    async def _warmup_model(self, model: Any, model_id: str) -> None:
        """Warmup model with synthetic utterance.

        Runs a quick synthesis to warm up the model and ensure it's ready
        for real-time inference.

        Args:
            model: Model instance
            model_id: Model identifier
        """
        logger.info("Warming up model", extra={"model_id": model_id})
        start_time = time.time()

        try:
            # Generate warmup text stream
            async def warmup_text() -> Any:
                yield self.warmup_text

            # Run synthesis (consume all frames)
            frame_count = 0
            async for _frame in model.synthesize_stream(warmup_text()):
                frame_count += 1

            warmup_duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "Model warmup complete",
                extra={
                    "model_id": model_id,
                    "warmup_duration_ms": warmup_duration_ms,
                    "frames_generated": frame_count,
                },
            )

        except Exception as e:
            logger.warning(
                "Model warmup failed (continuing)",
                extra={"model_id": model_id, "error": str(e)},
            )

    async def _unload_model(self, model_id: str) -> None:
        """Internal implementation of model unloading.

        Removes model from loaded_models and cleans up metadata. Does NOT
        check refcount - caller must ensure model is not in use.

        Args:
            model_id: Model identifier
        """
        if model_id not in self.loaded_models:
            return

        logger.info("Unloading model", extra={"model_id": model_id})

        # Remove from loaded models
        model = self.loaded_models.pop(model_id)

        # Clean up metadata
        self.refcounts.pop(model_id, None)
        self.last_used_ts.pop(model_id, None)
        self.load_ts.pop(model_id, None)

        # Call unload on the model if it has the method
        if hasattr(model, "unload_model"):
            try:
                await model.unload_model(model_id)
            except Exception as e:
                logger.warning(
                    "Error calling model.unload_model",
                    extra={"model_id": model_id, "error": str(e)},
                )

        logger.info(
            "Model unloaded successfully",
            extra={"model_id": model_id, "resident_count": len(self.loaded_models)},
        )

    async def _enforce_resident_cap(self) -> None:
        """Enforce resident capacity limit using LRU eviction.

        If the number of loaded models exceeds resident_cap, evicts the
        least recently used models (with refcount == 0) until under the cap.

        Must be called with state_lock held.
        """
        if len(self.loaded_models) <= self.resident_cap:
            return

        logger.info(
            "Resident capacity exceeded, applying LRU eviction",
            extra={
                "resident_count": len(self.loaded_models),
                "resident_cap": self.resident_cap,
            },
        )

        # Build list of evictable models (refcount == 0)
        evictable = []
        for model_id in self.loaded_models:
            if self.refcounts.get(model_id, 0) == 0:
                last_used = self.last_used_ts.get(model_id, 0)
                evictable.append((last_used, model_id))

        # Sort by last used time (oldest first)
        evictable.sort()

        # Evict until under cap
        models_to_evict = len(self.loaded_models) - self.resident_cap
        for i in range(min(models_to_evict, len(evictable))):
            _last_used, model_id = evictable[i]
            logger.info(
                "Evicting model (LRU)",
                extra={"model_id": model_id, "resident_count": len(self.loaded_models)},
            )
            await self._unload_model(model_id)

        logger.info(
            "LRU eviction complete",
            extra={"resident_count": len(self.loaded_models)},
        )

    async def _eviction_loop(self) -> None:
        """Background task that periodically runs eviction.

        Runs every evict_check_interval_ms and calls evict_idle().
        """
        logger.info(
            "Starting eviction loop",
            extra={"check_interval_ms": self.evict_check_interval_ms},
        )

        interval_seconds = self.evict_check_interval_ms / 1000.0

        while not self._shutdown:
            try:
                await asyncio.sleep(interval_seconds)
                await self.evict_idle()
            except asyncio.CancelledError:
                logger.info("Eviction loop cancelled")
                break
            except Exception as e:
                logger.error(
                    "Error in eviction loop",
                    extra={"error": str(e)},
                )

        logger.info("Eviction loop stopped")
