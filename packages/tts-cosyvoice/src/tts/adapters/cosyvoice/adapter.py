"""CosyVoice 2 TTS adapter - GPU-based neural TTS for realtime synthesis.

This adapter integrates CosyVoice 2 (https://github.com/FunAudioLLM/CosyVoice) as the
first GPU-accelerated TTS model in the system (M6). CosyVoice 2 is a high-quality,
streaming-capable neural TTS system optimized for low-latency realtime synthesis.

Key features:
- Zero-shot voice cloning with 3-10s reference audio
- Streaming synthesis with incremental text processing
- GPU acceleration for sub-300ms First Audio Latency (FAL)
- Multi-speaker support with voice embeddings
"""

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import torch
from numpy.typing import NDArray

from tts.audio.framing import repacketize_to_20ms
from tts.audio.processing import process_audio_for_streaming
from tts.audio.resampling import resample_audio
from tts.tts_base import AdapterState

# Import CosyVoice 2 with graceful fallback for testing
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
except ImportError:
    CosyVoice2 = None  # Graceful degradation for testing

# Import Hugging Face Hub for model download
if TYPE_CHECKING:
    from huggingface_hub import snapshot_download as _snapshot_download
else:
    try:
        from huggingface_hub import snapshot_download as _snapshot_download
    except ImportError:
        _snapshot_download = None  # type: ignore[assignment]  # Graceful degradation

# Type alias for snapshot_download (can be None if not installed)
snapshot_download: Any = _snapshot_download
# Constants
TARGET_SAMPLE_RATE_HZ: Final[int] = 48000  # Required output sample rate
FRAME_DURATION_MS: Final[int] = 20  # 20ms frames
SAMPLES_PER_FRAME: Final[int] = 960  # 48000 Hz * 0.020 sec = 960 samples
WARMUP_TEXT: Final[str] = "Testing warmup synthesis for model initialization."

# CosyVoice 2 specific constants
COSYVOICE_NATIVE_SAMPLE_RATE: Final[int] = 24000  # CosyVoice 2 native rate
COSYVOICE_DEFAULT_SPEAKER: Final[str] = "default"  # Default speaker ID
COSYVOICE_HF_REPO_ID: Final[str] = "FunAudioLLM/CosyVoice2-0.5B"  # Official Hugging Face repo
# Silence prompt duration: 2 seconds at 16kHz provides sufficient samples for
# mel spectrogram padding. CosyVoice requires >=1440 samples after resampling
# for 720-sample padding. Using 2s (32000@16kHz → 48000@24kHz) provides margin.
SILENCE_PROMPT_DURATION_SAMPLES_16KHZ: Final[int] = 32000  # 2 seconds at 16kHz

# Audio normalization: Reduced from 0.95 to 0.85 to prevent resampling overshoots
# CosyVoice outputs conservative amplitudes (~0.5 peak) to avoid clipping,
# so we normalize to 85% of full scale to match Piper's loudness levels
# while leaving 15% headroom for resampling/processing artifacts and fades.
AUDIO_NORMALIZATION_TARGET_PEAK: Final[float] = 0.85  # 85% of full scale (-1.4 dBFS)

logger = logging.getLogger(__name__)


def _download_cosyvoice_model_if_needed(model_path: Path) -> None:
    """Download CosyVoice2 model from Hugging Face if not already present.

    Args:
        model_path: Path to the voicepack directory

    Raises:
        ImportError: If huggingface_hub is not available
        RuntimeError: If download fails
    """
    # Check if model already exists (look for key files)
    required_files = ["cosyvoice.yaml", "llm.pt", "flow.pt", "hift.pt"]
    model_exists = all((model_path / f).exists() for f in required_files)

    if model_exists:
        logger.info(
            "CosyVoice2 model already exists, skipping download",
            extra={"model_path": str(model_path)},
        )
        return

    logger.info(
        "CosyVoice2 model not found, downloading from Hugging Face",
        extra={"model_path": str(model_path), "repo_id": COSYVOICE_HF_REPO_ID},
    )

    if snapshot_download is None:
        raise ImportError(
            "huggingface_hub is required to download CosyVoice2 model. "
            "Install with: pip install huggingface-hub"
        )

    try:
        # Create parent directory if it doesn't exist
        model_path.mkdir(parents=True, exist_ok=True)

        # Download model from Hugging Face
        logger.info(
            "Downloading CosyVoice2 model from Hugging Face (this may take a few minutes)...",
            extra={"repo_id": COSYVOICE_HF_REPO_ID, "local_dir": str(model_path)},
        )

        downloaded_path = snapshot_download(
            repo_id=COSYVOICE_HF_REPO_ID,
            local_dir=str(model_path),
            # Resume is now automatic in huggingface_hub 1.0+
        )

        logger.info(
            "CosyVoice2 model downloaded successfully",
            extra={"downloaded_path": downloaded_path, "model_path": str(model_path)},
        )

    except Exception as e:
        logger.error(
            "Failed to download CosyVoice2 model from Hugging Face",
            extra={"error": str(e), "repo_id": COSYVOICE_HF_REPO_ID},
            exc_info=True,
        )
        raise RuntimeError(
            f"Failed to download CosyVoice2 model: {e}. "
            "Please check your internet connection and try again."
        ) from e


class CosyVoiceAdapter:
    """CosyVoice 2 TTS adapter implementing the TTSAdapter protocol.

    This adapter loads CosyVoice 2 models from voicepacks and provides streaming
    synthesis with 20ms PCM frames at 48kHz. It supports pause/resume/stop control
    commands with <50ms response time.

    GPU acceleration enables:
    - First Audio Latency (FAL): p95 < 300ms
    - Streaming synthesis: incremental text processing
    - Zero-shot voice cloning: 3-10s reference audio
    - Multi-speaker support: voice embeddings

    Attributes:
        model_id: Identifier for the model instance
        model_path: Path to the voicepack directory
        device: CUDA device for model inference
        model: Loaded CosyVoice model instance
        native_sample_rate: Sample rate of the CosyVoice model (24000 Hz)
        state: Current adapter state (IDLE, SYNTHESIZING, PAUSED, STOPPED)
        pause_event: Event for pause/resume signaling
        stop_event: Event for stop signaling
        lock: Async lock for protecting state transitions

    Example:
        >>> adapter = CosyVoiceAdapter(
        ...     model_id="cosyvoice2-en-base",
        ...     model_path="voicepacks/cosyvoice/en-base"
        ... )
        >>> async def text_gen():
        ...     yield "Hello, world!"
        >>> async for frame in adapter.synthesize_stream(text_gen()):
        ...     # Process 20ms audio frame at 48kHz
        ...     pass
    """

    def __init__(self, model_id: str, model_path: str | Path) -> None:
        """Initialize the CosyVoice adapter.

        Args:
            model_id: Model identifier (e.g., "cosyvoice2-en-base")
            model_path: Path to the voicepack directory containing model files

        Raises:
            FileNotFoundError: If model files are missing
            ValueError: If model configuration is invalid
            ImportError: If CosyVoice package is not installed
        """
        self.model_id = model_id
        self.model_path = Path(model_path)
        self.state = AdapterState.IDLE
        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Start unpaused
        self.stop_event = asyncio.Event()
        self.lock = asyncio.Lock()  # For async state management
        self.model_lock = threading.Lock()  # For thread-safe model access

        # Check CUDA availability (warning only, not fatal)
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA is not available - CosyVoice requires GPU acceleration",
                extra={"model_id": model_id},
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-create silence prompt (cached for reuse across all synthesis calls)
        # Avoids repeated 128KB allocations + zero-filling on every synthesis
        # This is a CPU tensor - GPU acceleration happens later in model inference
        self._silence_prompt_cache = torch.zeros(
            (1, SILENCE_PROMPT_DURATION_SAMPLES_16KHZ),
            dtype=torch.float32,
            device="cpu",
        )
        logger.info(
            "Silence prompt pre-allocated",
            extra={"samples": SILENCE_PROMPT_DURATION_SAMPLES_16KHZ, "shape": [1, 32000]},
        )

        # Synthesis counter for telemetry (track first vs subsequent)
        self._synthesis_count = 0

        logger.info(
            "Initializing CosyVoice adapter",
            extra={
                "model_id": model_id,
                "model_path": str(self.model_path),
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
                ),
            },
        )

        # Download model from Hugging Face if not already present
        logger.info(
            "Checking if CosyVoice2 model needs to be downloaded",
            extra={"model_path": str(self.model_path)},
        )
        _download_cosyvoice_model_if_needed(self.model_path)

        # Load CosyVoice 2 model
        if CosyVoice2 is None:
            logger.error(
                "CosyVoice2 package not installed - cannot load model",
                extra={"model_id": model_id},
            )
            raise ImportError(
                "CosyVoice2 package not installed. "
                "Install with: pip install cosyvoice"
            )

        # Enable TensorRT if requested (provides ~2x speedup on flow matching stage)
        # TensorRT requires initial compilation but dramatically improves RTF
        use_tensorrt = os.getenv("USE_TENSORRT", "false").lower() == "true"
        use_vllm = os.getenv("USE_VLLM", "false").lower() == "true"
        
        try:
            self.model = CosyVoice2(
                str(self.model_path),
                load_jit=False,  # Optional optimization
                load_trt=use_tensorrt,  # TensorRT optimization (~2x speedup on RTX 4090)
                load_vllm=use_vllm,  # vLLM acceleration for LLM stage (~2x speedup)
                fp16=True,  # FP16 for faster inference
            )
            # Verify TensorRT compilation status if enabled
            tensorrt_status = "not_enabled"
            if use_tensorrt:
                try:
                    # Check if TensorRT engines exist (indicates compilation)
                    trt_dir = Path(self.model_path) / "trt"
                    if trt_dir.exists():
                        trt_files = list(trt_dir.glob("*.engine"))
                        if trt_files:
                            tensorrt_status = f"compiled ({len(trt_files)} engines)"
                        else:
                            tensorrt_status = "compiling (first run)"
                    else:
                        tensorrt_status = "compiling (first run)"
                except Exception:
                    tensorrt_status = "unknown"
            
            logger.info(
                "CosyVoice 2 model loaded successfully",
                extra={
                    "model_id": model_id,
                    "model_path": str(self.model_path),
                    "tensorrt_enabled": use_tensorrt,
                    "tensorrt_status": tensorrt_status,
                    "vllm_enabled": use_vllm,
                    "fp16_enabled": True,
                },
            )
        except Exception as e:
            logger.error(
                "Failed to load CosyVoice 2 model",
                extra={
                    "model_id": model_id,
                    "model_path": str(self.model_path),
                    "error": str(e),
                },
            )
            raise

        # CosyVoice 2 has fixed 24kHz sample rate
        self.native_sample_rate = COSYVOICE_NATIVE_SAMPLE_RATE

        logger.info(
            "CosyVoiceAdapter initialized",
            extra={
                "model_id": model_id,
                "native_sample_rate": self.native_sample_rate,
                "target_sample_rate": TARGET_SAMPLE_RATE_HZ,
                "device": str(self.device),
            },
        )

    async def synthesize_stream(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Generate TTS audio for each text chunk.

        For each text chunk, synthesizes audio using CosyVoice 2, resamples to 48kHz,
        and repacketizes into 20ms PCM frames. Yields frames as fast as possible to
        minimize latency.

        Args:
            text_chunks: Async iterator of text chunks to synthesize

        Yields:
            20ms PCM audio frames at 48kHz (bytes, int16 little-endian)

        Notes:
            - Respects PAUSE commands (stops yielding frames immediately)
            - Respects RESUME commands (continues yielding frames)
            - Respects STOP commands (terminates streaming)
            - All control commands respond within < 50ms
            - Uses GPU acceleration for synthesis (CUDA)
        """
        async with self.lock:
            self.state = AdapterState.SYNTHESIZING
            logger.info(
                "Starting synthesis stream",
                extra={"state": self.state.value, "model_id": self.model_id},
            )

        try:
            chunk_count = 0
            async for text in text_chunks:
                chunk_count += 1
                logger.debug(
                    "Processing text chunk",
                    extra={
                        "chunk_id": chunk_count,
                        "text_length": len(text),
                        "text_preview": text[:50],
                        "model_id": self.model_id,
                    },
                )

                # Synthesize audio using CosyVoice (blocking call, run in executor)
                audio = await asyncio.to_thread(self._synthesize_cosyvoice, text)

                # Resample to 48kHz if needed
                if self.native_sample_rate != TARGET_SAMPLE_RATE_HZ:
                    audio = await asyncio.to_thread(
                        resample_audio, audio, self.native_sample_rate, TARGET_SAMPLE_RATE_HZ
                    )

                # Repacketize to 20ms frames
                frames = await asyncio.to_thread(
                    repacketize_to_20ms, audio, sample_rate=TARGET_SAMPLE_RATE_HZ
                )

                logger.debug(
                    "Generated frames for chunk",
                    extra={
                        "chunk_id": chunk_count,
                        "frame_count": len(frames),
                        "audio_duration_ms": len(audio) / TARGET_SAMPLE_RATE_HZ * 1000,
                        "model_id": self.model_id,
                    },
                )

                # Yield frames immediately without artificial delay
                for frame_idx, frame in enumerate(frames):
                    # Check if stopped (immediate termination)
                    if self.stop_event.is_set():
                        logger.info(
                            "Synthesis stopped by STOP command",
                            extra={
                                "chunk_id": chunk_count,
                                "frame_idx": frame_idx,
                                "model_id": self.model_id,
                            },
                        )
                        return

                    # Wait if paused (blocks until RESUME)
                    await self.pause_event.wait()

                    # Check again before yielding (race condition fix)
                    if self.stop_event.is_set():
                        return
                    await self.pause_event.wait()

                    yield frame

            logger.info(
                "Synthesis stream completed",
                extra={"total_chunks": chunk_count, "model_id": self.model_id},
            )

        finally:
            async with self.lock:
                # Only reset to IDLE if we weren't stopped
                if self.state != AdapterState.STOPPED:
                    self.state = AdapterState.IDLE
                    logger.info(
                        "Synthesis stream ended",
                        extra={"state": self.state.value, "model_id": self.model_id},
                    )

    def _synthesize_cosyvoice(self, text: str) -> NDArray[np.int16]:
        """Synthesize audio using CosyVoice 2 (synchronous).

        Args:
            text: Text to synthesize

        Returns:
            Audio samples as int16 numpy array at native sample rate (24kHz)

        Raises:
            RuntimeError: If synthesis fails

        Notes:
            This is a blocking call that runs on GPU. It should be called via
            asyncio.to_thread() to avoid blocking the event loop.

            For M6, uses batch mode with default voice. Future enhancements:
            - Voice cloning with reference audio
            - True streaming mode (stream=True)
            - Multi-speaker support
        """
        if self.model is None:
            logger.error(
                "Cannot synthesize - model not loaded",
                extra={"model_id": self.model_id},
            )
            raise RuntimeError("Model not loaded")

        try:
            # Performance tracking for Phase 1 optimization analysis
            timings = {}
            overall_start = time.perf_counter()

            with torch.no_grad():
                # CosyVoice 2 streaming API
                # inference_zero_shot() returns a generator of audio chunks
                audio_chunks: list[torch.Tensor] = []

                # For M6: Use batch mode with default voice (no reference audio)
                # Empty prompt_text and silence reference audio for default voice
                # CRITICAL FIX: CosyVoice expects torch.Tensor, not numpy.ndarray
                # Convert silence reference to torch.Tensor to avoid AttributeError
                # CosyVoice frontend preprocessing expects CPU tensors (GPU is used later)
                # GPU acceleration happens later in the model inference pipeline
                # PADDING FIX: Use 2-second silence (32000 samples) vs 1 second (16000)
                # After resampling to 24kHz (48000 samples), provides sufficient length
                # for mel spectrogram padding (requires >= 1440 samples for 720 padding)
                # SHAPE FIX: Add batch dimension [1, samples] to match CosyVoice convention
                # CosyVoice expects batch-first tensors: (batch, time) not just (time)
                # Without batch dim, library incorrectly reshapes to [samples, 1] vs [1, samples]
                # Use pre-allocated cached silence prompt (avoid repeated allocation)
                silence_prompt = self._silence_prompt_cache

                # Stage 1: CosyVoice inference (GPU)
                inference_start = time.perf_counter()

                # Retry logic for numerical instability in CosyVoice sampling
                # Uses exponential backoff to allow GPU state to stabilize between retries
                max_retries = 3
                initial_delay = 0.1  # 100ms
                backoff_factor = 2.0
                
                # Acquire thread lock for thread-safe model access
                # This prevents concurrent access from parallel workers
                with self.model_lock:
                    for attempt in range(max_retries):
                        try:
                            for chunk in self.model.inference_zero_shot(
                                tts_text=text,
                                prompt_text="",  # Empty for default voice
                                prompt_speech_16k=silence_prompt,  # torch.Tensor
                                stream=False,  # Use batch mode for simplicity (M6)
                                speed=1.0,
                            ):
                                # Extract audio tensor from chunk dict
                                # Expected format: {'tts_speech': torch.Tensor}
                                audio_tensor = chunk["tts_speech"]  # Shape: (1, samples)
                                audio_chunks.append(audio_tensor)
                            break  # Success!
                        except RuntimeError as e:
                            error_str = str(e)
                            
                            # Handle probability tensor errors (numerical instability)
                            if "probability tensor" in error_str and attempt < max_retries - 1:
                                # Calculate exponential backoff delay
                                delay = initial_delay * (backoff_factor ** attempt)
                                
                                logger.warning(
                                    f"CosyVoice sampling error (attempt {attempt + 1}/{max_retries}), "
                                    f"retrying in {delay:.3f}s... Error: {e}",
                                    extra={
                                        "model_id": self.model_id,
                                        "text_length": len(text),
                                        "attempt": attempt + 1,
                                        "delay": delay,
                                    },
                                )
                                audio_chunks.clear()  # Clear partial results
                                
                                # Synchronize CUDA and clear cache before retry
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()
                                
                                # Wait before retry to allow GPU state to stabilize
                                time.sleep(delay)
                                continue
                            
                            # Handle tensor type corruption errors (state corruption from concurrent access)
                            elif ("indices" in error_str and "FloatTensor" in error_str) or (
                                "Expected tensor for argument" in error_str
                                and "Long, Int" in error_str
                            ):
                                if attempt < max_retries - 1:
                                    delay = initial_delay * (backoff_factor ** attempt) * 2  # Longer delay for corruption
                                    
                                    logger.warning(
                                        f"Tensor type corruption detected (attempt {attempt + 1}/{max_retries}), "
                                        f"resetting model state and retrying in {delay:.3f}s... Error: {e}",
                                        extra={
                                            "model_id": self.model_id,
                                            "text_length": len(text),
                                            "attempt": attempt + 1,
                                            "delay": delay,
                                        },
                                    )
                                    audio_chunks.clear()
                                    
                                    # Reset model state and clear GPU cache
                                    if torch.cuda.is_available():
                                        torch.cuda.synchronize()
                                        torch.cuda.empty_cache()
                                    
                                    time.sleep(delay)
                                    continue
                                else:
                                    logger.error(
                                        f"Tensor type corruption after {max_retries} attempts",
                                        extra={"model_id": self.model_id, "error": str(e)},
                                    )
                            
                            # Handle attention dimension mismatch (state corruption from concurrent access)
                            elif ("expanded size" in error_str and "must match" in error_str) or (
                                "dimension" in error_str and "non-singleton" in error_str
                            ):
                                if attempt < max_retries - 1:
                                    delay = initial_delay * (backoff_factor ** attempt) * 2  # Longer delay for corruption
                                    
                                    logger.warning(
                                        f"Attention dimension mismatch detected (attempt {attempt + 1}/{max_retries}), "
                                        f"resetting model state and retrying in {delay:.3f}s... Error: {e}",
                                        extra={
                                            "model_id": self.model_id,
                                            "text_length": len(text),
                                            "attempt": attempt + 1,
                                            "delay": delay,
                                        },
                                    )
                                    audio_chunks.clear()
                                    
                                    # Reset model state and clear GPU cache
                                    if torch.cuda.is_available():
                                        torch.cuda.synchronize()
                                        torch.cuda.empty_cache()
                                    
                                    time.sleep(delay)
                                    continue
                                else:
                                    logger.error(
                                        f"Attention dimension mismatch after {max_retries} attempts",
                                        extra={"model_id": self.model_id, "error": str(e)},
                                    )
                            
                            # Re-raise if not handled above and this is the last attempt
                            if attempt == max_retries - 1:
                                # If all retries failed, try fallback: synthesize text in chunks (by sentences)
                                if "probability tensor" in error_str:
                                    logger.warning(
                                        f"CosyVoice synthesis failed after {max_retries} attempts, "
                                        "trying fallback: chunked synthesis",
                                        extra={
                                            "model_id": self.model_id,
                                            "text_length": len(text),
                                            "error": str(e),
                                        },
                                    )
                                
                                # Fallback: split text into sentences and synthesize separately
                                import re
                                sentences = re.split(r'([.!?]+[\s\n]*)', text)
                                # Filter empty sentences and merge punctuation with previous sentence
                                processed_sentences: list[str] = []
                                for i, sentence in enumerate(sentences):
                                    if sentence.strip():
                                        if i > 0 and processed_sentences and re.match(r'^[.!?]+\s*$', sentence):
                                            processed_sentences[-1] += sentence
                                        else:
                                            processed_sentences.append(sentence)
                                
                                if processed_sentences:
                                    logger.debug(
                                        f"Synthesizing {len(processed_sentences)} chunks separately",
                                        extra={"model_id": self.model_id},
                                    )
                                    
                                    chunked_audio: list[torch.Tensor] = []
                                    for chunk_idx, chunk_text in enumerate(processed_sentences):
                                        try:
                                            # Fallback chunked synthesis also needs thread lock
                                            for chunk_result in self.model.inference_zero_shot(
                                                tts_text=chunk_text,
                                                prompt_text="",
                                                prompt_speech_16k=silence_prompt,
                                                stream=False,
                                                speed=1.0,
                                            ):
                                                chunked_audio.append(chunk_result["tts_speech"])
                                        except RuntimeError as chunk_error:
                                            chunk_error_str = str(chunk_error)
                                            if (
                                                "probability tensor" in chunk_error_str
                                                or ("indices" in chunk_error_str and "FloatTensor" in chunk_error_str)
                                                or ("expanded size" in chunk_error_str and "must match" in chunk_error_str)
                                            ):
                                                logger.warning(
                                                    f"Chunk {chunk_idx + 1}/{len(processed_sentences)} "
                                                    f"failed with error, skipping: {chunk_error_str[:100]}",
                                                    extra={
                                                        "model_id": self.model_id,
                                                        "chunk_text": chunk_text[:50],
                                                    },
                                                )
                                                # Skip this chunk but continue with others
                                                continue
                                            else:
                                                raise
                                    
                                    if chunked_audio:
                                        audio_chunks = chunked_audio
                                        logger.info(
                                            f"Fallback chunked synthesis succeeded "
                                            f"({len(chunked_audio)} chunks)",
                                            extra={"model_id": self.model_id},
                                        )
                                        break  # Success with fallback!
                                
                                # If fallback didn't work, raise the error
                                logger.error(
                                    f"CosyVoice synthesis failed after {max_retries} attempts "
                                    f"and fallback strategies",
                                    extra={"model_id": self.model_id, "error": str(e)},
                                )
                                raise

                timings["inference_ms"] = (time.perf_counter() - inference_start) * 1000

                # Concatenate chunks
                if not audio_chunks:
                    logger.warning(
                        "No audio generated for text",
                        extra={"model_id": self.model_id, "text_length": len(text)},
                    )
                    return np.zeros(0, dtype=np.int16)

                # Stage 2: Tensor concatenation
                concat_start = time.perf_counter()
                audio_tensor = torch.cat(audio_chunks, dim=1)
                timings["concat_ms"] = (time.perf_counter() - concat_start) * 1000

                # Stage 3: Audio processing and type conversion
                convert_start = time.perf_counter()

                # Convert to float32 numpy for processing
                audio_float = audio_tensor.squeeze(0).cpu().numpy()

                # Apply comprehensive audio processing pipeline
                # This eliminates pops, clicks, static, and resampling artifacts
                audio_processed = process_audio_for_streaming(
                    audio_float,
                    sample_rate=self.native_sample_rate,
                    target_peak=AUDIO_NORMALIZATION_TARGET_PEAK,
                    apply_fades=True,  # 5ms fade-in/out prevents boundary pops
                    apply_dithering=True,  # Reduces quantization noise
                )

                # Convert to int16 with proper clipping
                # Clipping prevents overflow from any remaining processing artifacts
                audio_int16 = np.clip(audio_processed * 32767.0, -32768, 32767).astype(np.int16)

                timings["convert_ms"] = (time.perf_counter() - convert_start) * 1000

                # Calculate overall performance metrics
                total_ms = (time.perf_counter() - overall_start) * 1000
                audio_duration_s = len(audio_int16) / self.native_sample_rate
                rtf = (total_ms / 1000) / audio_duration_s if audio_duration_s > 0 else 0

                # Measure peak and RMS for quality validation
                peak = np.abs(audio_int16).max() / 32767.0
                rms = np.sqrt(np.mean((audio_int16 / 32767.0) ** 2))

                # TELEMETRY: Track first vs subsequent synthesis for performance analysis
                is_first_synthesis = (self._synthesis_count == 0)
                self._synthesis_count += 1

                # Log comprehensive performance breakdown for Phase 1 analysis
                logger.info(
                    "CosyVoice synthesis performance",
                    extra={
                        "model_id": self.model_id,
                        "text_length": len(text),
                        "audio_duration_s": round(audio_duration_s, 2),
                        "audio_samples": len(audio_int16),
                        "total_ms": round(total_ms, 1),
                        "rtf": round(rtf, 3),
                        "is_first_synthesis": is_first_synthesis,  # Track first vs subsequent
                        "synthesis_number": self._synthesis_count,
                        "inference_ms": round(timings["inference_ms"], 1),
                        "concat_ms": round(timings["concat_ms"], 1),
                        "convert_ms": round(timings["convert_ms"], 1),
                        "inference_pct": round(timings["inference_ms"] / total_ms * 100, 1),
                        "peak_level": round(peak, 3),
                        "rms_level": round(rms, 3),
                        "target_peak": AUDIO_NORMALIZATION_TARGET_PEAK,
                    },
                )

                return audio_int16

        except Exception as e:
            logger.error(
                "CosyVoice synthesis failed",
                extra={
                    "model_id": self.model_id,
                    "text_length": len(text),
                    "error": str(e),
                },
            )
            raise RuntimeError(f"CosyVoice synthesis failed: {e}") from e

    async def control(self, command: str) -> None:
        """Handle control commands with < 50ms response time.

        Args:
            command: Control command string (PAUSE, RESUME, STOP)

        Raises:
            ValueError: If command is not recognized

        Notes:
            - PAUSE: Stops yielding frames immediately, state → PAUSED
            - RESUME: Continues yielding frames, state → SYNTHESIZING
            - STOP: Terminates streaming, state → STOPPED
            - All commands use asyncio.Event for immediate response
        """
        async with self.lock:
            previous_state = self.state

            if command == "PAUSE":
                if self.state == AdapterState.SYNTHESIZING:
                    self.state = AdapterState.PAUSED
                    self.pause_event.clear()  # Block synthesize_stream
                    logger.info(
                        "Adapter paused",
                        extra={
                            "command": command,
                            "previous_state": previous_state.value,
                            "model_id": self.model_id,
                        },
                    )
                else:
                    logger.warning(
                        "PAUSE command ignored (not synthesizing)",
                        extra={"current_state": self.state.value, "model_id": self.model_id},
                    )

            elif command == "RESUME":
                if self.state == AdapterState.PAUSED:
                    self.state = AdapterState.SYNTHESIZING
                    self.pause_event.set()  # Unblock synthesize_stream
                    logger.info(
                        "Adapter resumed",
                        extra={
                            "command": command,
                            "previous_state": previous_state.value,
                            "model_id": self.model_id,
                        },
                    )
                else:
                    logger.warning(
                        "RESUME command ignored (not paused)",
                        extra={"current_state": self.state.value, "model_id": self.model_id},
                    )

            elif command == "STOP":
                self.state = AdapterState.STOPPED
                self.stop_event.set()
                self.pause_event.set()  # Unblock if paused
                logger.info(
                    "Adapter stopped",
                    extra={
                        "command": command,
                        "previous_state": previous_state.value,
                        "model_id": self.model_id,
                    },
                )

            else:
                logger.error(
                    "Unknown control command",
                    extra={"command": command, "model_id": self.model_id},
                )
                raise ValueError(f"Unknown control command: {command}")

    async def load_model(self, model_id: str) -> None:
        """Load a specific CosyVoice model.

        Args:
            model_id: Model identifier

        Notes:
            For M6, this is simplified - the model is loaded at initialization.
            Future implementations may support dynamic model loading for the
            Model Manager (M4) to manage multiple models with TTL-based eviction.

            GPU memory management:
            - Model size: ~1-2GB VRAM (typical for CosyVoice)
            - Consider unloading unused models to free VRAM
            - Use torch.cuda.empty_cache() after unloading
        """
        logger.info(
            "CosyVoice load_model called (model already loaded at init)",
            extra={"model_id": model_id, "adapter_model_id": self.model_id},
        )

        # TODO: Implement dynamic model loading for M4+ if needed
        # if self.model is None:
        #     self.model = CosyVoice2(
        #         str(self.model_path),
        #         load_jit=False,
        #         load_trt=False,
        #         load_vllm=False,
        #         fp16=True,
        #     )

    async def unload_model(self, model_id: str) -> None:
        """Unload a specific CosyVoice model.

        Args:
            model_id: Model identifier

        Notes:
            For M6, this is simplified - model lifecycle is managed by adapter
            instance lifecycle. Future implementations may support dynamic unloading
            for GPU memory management (M4+).

            GPU memory cleanup:
            - Delete model reference
            - Call torch.cuda.empty_cache() to free VRAM
            - Monitor VRAM usage with torch.cuda.memory_allocated()
        """
        logger.info(
            "CosyVoice unload_model called (model lifecycle managed by instance)",
            extra={"model_id": model_id, "adapter_model_id": self.model_id},
        )

        # TODO: Implement dynamic model unloading for M4+ if needed
        # if self.model is not None:
        #     del self.model
        #     self.model = None
        #     torch.cuda.empty_cache()

    async def warm_up(self) -> None:
        """Warm up the model by synthesizing a test utterance.

        This method synthesizes a short test sentence to ensure the model
        is fully loaded, cached, and GPU kernels are compiled for faster
        first-real-synthesis latency.

        Notes:
            Target: <1s warmup time on modern GPU (RTX 3090/4090, A100)
            Discards output audio, measures duration for telemetry
            GPU-specific optimizations:
            - Compile CUDA kernels on first run
            - Cache embeddings for default speaker
            - Allocate GPU memory upfront
        """
        import time

        logger.info("Starting warmup synthesis", extra={"model_id": self.model_id})

        start_time = time.perf_counter()

        # Synthesize warmup text
        audio = await asyncio.to_thread(self._synthesize_cosyvoice, WARMUP_TEXT)

        # Resample and repacketize (same as normal synthesis)
        if self.native_sample_rate != TARGET_SAMPLE_RATE_HZ:
            audio = await asyncio.to_thread(
                resample_audio, audio, self.native_sample_rate, TARGET_SAMPLE_RATE_HZ
            )

        # Just measure, don't repacketize
        warmup_duration_ms = (time.perf_counter() - start_time) * 1000

        # Log GPU memory usage
        if torch.cuda.is_available():
            gpu_memory_allocated_mb = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            gpu_memory_reserved_mb = torch.cuda.memory_reserved(self.device) / 1024 / 1024
        else:
            gpu_memory_allocated_mb = 0.0
            gpu_memory_reserved_mb = 0.0

        logger.info(
            "Warmup synthesis complete",
            extra={
                "model_id": self.model_id,
                "warmup_duration_ms": warmup_duration_ms,
                "audio_duration_ms": len(audio) / TARGET_SAMPLE_RATE_HZ * 1000,
                "gpu_memory_allocated_mb": gpu_memory_allocated_mb,
                "gpu_memory_reserved_mb": gpu_memory_reserved_mb,
            },
        )

    def get_state(self) -> AdapterState:
        """Get the current adapter state.

        Returns:
            Current adapter state

        Notes:
            This method is provided for testing purposes to inspect adapter state
            without accessing internal attributes directly.
        """
        return self.state

    async def reset(self) -> None:
        """Reset the adapter to initial state.

        This is a testing utility method to reset the adapter between test cases.
        It clears all events and resets the state to IDLE.

        Notes:
            Not part of the TTSAdapter protocol - for testing only.
            Does NOT clear GPU memory or reload the model.
        """
        async with self.lock:
            self.state = AdapterState.IDLE
            self.pause_event.set()
            self.stop_event.clear()
            # Reset synthesis counter
            self._synthesis_count = 0
            logger.info("Adapter reset to initial state", extra={"model_id": self.model_id})
