"""Streaming audio processing utilities for TTS adapters.

This module provides utilities for processing audio in chunks while maintaining
quality at boundaries. Key components:

- StreamingResampler: Progressive resampling with overlap buffering
- FrameBuffer: Accumulate samples and emit fixed-size frames

These utilities enable low-latency streaming TTS with smooth audio output.
"""

from __future__ import annotations

import torch
import torchaudio.functional as F


class StreamingResampler:
    """Progressive audio resampler for chunk-based processing.

    Handles incremental resampling while maintaining quality at chunk
    boundaries using an overlap buffer to ensure smooth transitions.

    The overlap buffer prevents artifacts (clicks, pops) at chunk boundaries
    by maintaining context from the previous chunk during resampling.

    Args:
        input_rate: Source sample rate (Hz)
        output_rate: Target sample rate (Hz)
        overlap_samples: Overlap buffer size for smooth boundaries
            Default is 441 samples (20ms at 22050Hz)
        device: Torch device ("cpu" or "cuda")

    Example:
        >>> resampler = StreamingResampler(24000, 48000, device="cuda")
        >>> for chunk in audio_chunks:
        ...     resampled = resampler.process(chunk)
        ...     yield resampled
        >>> final = resampler.flush()  # Process residual samples
    """

    def __init__(
        self,
        input_rate: int,
        output_rate: int,
        overlap_samples: int = 441,  # 20ms at 22050Hz
        device: str = "cpu",
    ) -> None:
        """Initialize streaming resampler.

        Args:
            input_rate: Source sample rate in Hz
            output_rate: Target sample rate in Hz
            overlap_samples: Number of samples to overlap between chunks
            device: Torch device for tensor operations
        """
        if input_rate <= 0:
            raise ValueError(f"input_rate must be positive, got {input_rate}")
        if output_rate <= 0:
            raise ValueError(f"output_rate must be positive, got {output_rate}")
        if overlap_samples < 0:
            raise ValueError(f"overlap_samples must be non-negative, got {overlap_samples}")

        self.input_rate = input_rate
        self.output_rate = output_rate
        self.overlap_samples = overlap_samples
        self.device = device
        self.ratio = output_rate / input_rate

        # Overlap buffer for smooth chunk transitions
        self.overlap_buffer: torch.Tensor | None = None

    def process(self, chunk: torch.Tensor) -> torch.Tensor:
        """Resample audio chunk with boundary handling.

        Args:
            chunk: Input audio samples (1D or 2D tensor)
                Shape: (samples,) or (channels, samples)

        Returns:
            Resampled audio at output_rate with same shape as input

        Raises:
            ValueError: If chunk is empty or has invalid dimensions
        """
        if chunk.numel() == 0:
            raise ValueError("Cannot process empty chunk")

        if chunk.dim() > 2:
            raise ValueError(f"Chunk must be 1D or 2D, got {chunk.dim()}D")

        # Ensure chunk is on correct device
        if chunk.device.type != self.device:
            chunk = chunk.to(self.device)

        # Prepend overlap from previous chunk for smooth transition
        if self.overlap_buffer is not None:
            chunk = torch.cat([self.overlap_buffer, chunk], dim=-1)

        # Resample using torchaudio (GPU-accelerated if device="cuda")
        resampled: torch.Tensor = F.resample(
            chunk,
            orig_freq=self.input_rate,
            new_freq=self.output_rate,
            resampling_method="sinc_interp_kaiser",  # High quality
        )

        # Save overlap for next chunk (last N samples of input)
        if chunk.shape[-1] > self.overlap_samples:
            self.overlap_buffer = chunk[..., -self.overlap_samples:].clone()
        else:
            # If chunk is smaller than overlap, save entire chunk
            self.overlap_buffer = chunk.clone()

        return resampled

    def flush(self) -> torch.Tensor | None:
        """Process final residual samples in overlap buffer.

        Call this after processing all chunks to emit any remaining
        samples in the overlap buffer.

        Returns:
            Final resampled audio or None if no residual samples
        """
        if self.overlap_buffer is not None and self.overlap_buffer.numel() > 0:
            final_resampled: torch.Tensor = F.resample(
                self.overlap_buffer,
                orig_freq=self.input_rate,
                new_freq=self.output_rate,
                resampling_method="sinc_interp_kaiser",
            )
            self.overlap_buffer = None
            return final_resampled
        return None

    def reset(self) -> None:
        """Reset resampler state, clearing overlap buffer.

        Use this when starting a new audio stream to prevent
        contamination from previous streams.
        """
        self.overlap_buffer = None


class FrameBuffer:
    """Accumulate audio samples and emit fixed-size frames.

    This buffer collects incoming audio samples and yields frames of exactly
    `frame_size` samples. Partial frames are held until enough samples arrive.

    Args:
        frame_size: Target frame size in samples
        sample_rate: Audio sample rate (Hz), used for time calculations
        device: Torch device for tensor operations

    Example:
        >>> buffer = FrameBuffer(frame_size=960, sample_rate=48000)
        >>> for chunk in audio_chunks:
        ...     frames = buffer.push(chunk)
        ...     for frame in frames:
        ...         yield frame  # Each frame is exactly 960 samples
        >>> final = buffer.flush()  # Get partial frame at end
    """

    def __init__(
        self,
        frame_size: int,
        sample_rate: int,
        device: str = "cpu",
    ) -> None:
        """Initialize frame buffer.

        Args:
            frame_size: Target frame size in samples
            sample_rate: Audio sample rate in Hz
            device: Torch device for tensor operations
        """
        if frame_size <= 0:
            raise ValueError(f"frame_size must be positive, got {frame_size}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")

        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.device = device

        # Internal buffer for accumulating samples
        self.buffer: torch.Tensor | None = None
        self._total_samples_emitted = 0

    def push(self, samples: torch.Tensor) -> list[torch.Tensor]:
        """Push samples and emit complete frames.

        Args:
            samples: Input audio samples (1D or 2D tensor)
                Shape: (samples,) or (channels, samples)

        Returns:
            List of complete frames, each with shape matching input
            but with frame_size samples. May be empty if not enough
            samples accumulated yet.

        Raises:
            ValueError: If samples is empty or has invalid dimensions
        """
        if samples.numel() == 0:
            raise ValueError("Cannot push empty samples")

        if samples.dim() > 2:
            raise ValueError(f"Samples must be 1D or 2D, got {samples.dim()}D")

        # Ensure samples are on correct device
        if samples.device.type != self.device:
            samples = samples.to(self.device)

        # Append to internal buffer
        if self.buffer is None:
            self.buffer = samples
        else:
            self.buffer = torch.cat([self.buffer, samples], dim=-1)

        # Emit complete frames
        frames: list[torch.Tensor] = []
        while self.buffer.shape[-1] >= self.frame_size:
            frame = self.buffer[..., :self.frame_size]
            frames.append(frame)
            self.buffer = self.buffer[..., self.frame_size:]
            self._total_samples_emitted += self.frame_size

        return frames

    def flush(self) -> torch.Tensor | None:
        """Emit final partial frame if any samples remain.

        Returns:
            Partial frame with remaining samples, or None if buffer is empty
        """
        if self.buffer is not None and self.buffer.numel() > 0:
            final_frame = self.buffer
            samples_count = final_frame.shape[-1]
            self.buffer = None
            self._total_samples_emitted += samples_count
            return final_frame
        return None

    def reset(self) -> None:
        """Reset buffer state, discarding any accumulated samples."""
        self.buffer = None
        self._total_samples_emitted = 0

    @property
    def buffered_samples(self) -> int:
        """Number of samples currently in buffer (not yet emitted)."""
        if self.buffer is None:
            return 0
        return self.buffer.shape[-1]

    @property
    def buffered_duration_ms(self) -> float:
        """Duration of buffered samples in milliseconds."""
        return (self.buffered_samples / self.sample_rate) * 1000.0

    @property
    def total_samples_emitted(self) -> int:
        """Total number of samples emitted since creation or last reset."""
        return self._total_samples_emitted
