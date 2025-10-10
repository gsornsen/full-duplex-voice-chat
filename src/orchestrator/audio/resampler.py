"""Audio resampling utilities for VAD preprocessing.

Provides efficient resampling from 48kHz (client audio) to 16kHz (VAD requirement)
using scipy's high-quality signal processing functions.

Key features:
- Resample 48kHz → 16kHz for webrtcvad processing
- Maintain audio quality for accurate VAD
- Efficient processing for real-time streaming
- Support for arbitrary sample rates
"""

import logging
import struct
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import signal

logger = logging.getLogger(__name__)


class AudioResampler:
    """High-quality audio resampler using scipy.

    Resamples PCM audio frames between different sample rates while
    maintaining quality for speech processing.

    Thread-safety: This class is thread-safe for read operations after
    initialization, but process_frame() should be called from a single thread.

    Example:
        ```python
        # Resample 48kHz → 16kHz for VAD
        resampler = AudioResampler(source_rate=48000, target_rate=16000)
        frame_48k = get_audio_frame()  # 1920 bytes @ 48kHz, 20ms
        frame_16k = resampler.process_frame(frame_48k)  # 640 bytes @ 16kHz, 20ms
        ```
    """

    def __init__(self, source_rate: int, target_rate: int) -> None:
        """Initialize resampler.

        Args:
            source_rate: Source sample rate in Hz (e.g., 48000)
            target_rate: Target sample rate in Hz (e.g., 16000)

        Raises:
            ValueError: If sample rates are invalid
        """
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError(
                f"Sample rates must be positive: source={source_rate}, target={target_rate}"
            )

        self._source_rate = source_rate
        self._target_rate = target_rate
        self._ratio = target_rate / source_rate

        logger.info(
            f"Resampler initialized: {source_rate}Hz → {target_rate}Hz (ratio={self._ratio:.4f})"
        )

    def process_frame(self, frame: bytes) -> bytes:
        """Resample a PCM audio frame.

        Args:
            frame: Raw PCM audio (16-bit signed int, little endian, mono)

        Returns:
            Resampled PCM audio frame

        Raises:
            ValueError: If frame size is invalid (not a multiple of 2 bytes)
        """
        if len(frame) % 2 != 0:
            raise ValueError(
                f"Frame size must be a multiple of 2 bytes (16-bit samples), got {len(frame)} bytes"
            )

        # Convert bytes to numpy array
        audio_int16 = self._bytes_to_int16(frame)

        # Resample using scipy
        resampled = self._resample_signal(audio_int16)

        # Convert back to bytes
        return self._int16_to_bytes(resampled)

    def calculate_output_size(self, input_size_bytes: int) -> int:
        """Calculate expected output size after resampling.

        Args:
            input_size_bytes: Input frame size in bytes

        Returns:
            Expected output size in bytes

        Raises:
            ValueError: If input size is invalid
        """
        if input_size_bytes % 2 != 0:
            raise ValueError(f"Input size must be a multiple of 2 bytes, got {input_size_bytes}")

        input_samples = input_size_bytes // 2
        output_samples = int(input_samples * self._ratio)
        return output_samples * 2

    def _bytes_to_int16(self, data: bytes) -> NDArray[np.int16]:
        """Convert PCM bytes to int16 numpy array.

        Args:
            data: Raw PCM bytes (16-bit signed int, little endian)

        Returns:
            Numpy array of int16 samples
        """
        # Unpack bytes as signed 16-bit integers (little endian)
        sample_count = len(data) // 2
        return np.array(
            struct.unpack(f"<{sample_count}h", data),
            dtype=np.int16,
        )

    def _int16_to_bytes(self, data: NDArray[np.int16]) -> bytes:
        """Convert int16 numpy array to PCM bytes.

        Args:
            data: Numpy array of int16 samples

        Returns:
            Raw PCM bytes (16-bit signed int, little endian)
        """
        # Pack int16 array as signed 16-bit integers (little endian)
        return struct.pack(f"<{len(data)}h", *data)

    def _resample_signal(self, audio: NDArray[np.int16]) -> NDArray[np.int16]:
        """Resample audio signal using scipy.

        Uses scipy.signal.resample for high-quality resampling.
        This uses FFT-based resampling which provides good quality
        for speech signals.

        Args:
            audio: Input audio as int16 samples

        Returns:
            Resampled audio as int16 samples
        """
        # Calculate output length
        num_samples = len(audio)
        output_length = int(num_samples * self._ratio)

        # Convert to float for processing
        audio_float = audio.astype(np.float32)

        # Resample using scipy (FFT-based method)
        # scipy.signal.resample returns ndarray with Any dtype, so we cast it
        resampled_float: NDArray[Any] = cast(
            NDArray[Any], signal.resample(audio_float, output_length)
        )

        # Clip to int16 range and convert back
        resampled_int16: NDArray[np.int16] = np.clip(
            resampled_float,
            np.iinfo(np.int16).min,
            np.iinfo(np.int16).max,
        ).astype(np.int16)

        return resampled_int16

    @property
    def source_rate(self) -> int:
        """Get source sample rate."""
        return self._source_rate

    @property
    def target_rate(self) -> int:
        """Get target sample rate."""
        return self._target_rate

    @property
    def ratio(self) -> float:
        """Get resampling ratio (target/source)."""
        return self._ratio


def create_vad_resampler() -> AudioResampler:
    """Create a resampler for VAD preprocessing.

    Returns a preconfigured resampler that converts from 48kHz (standard
    client audio) to 16kHz (webrtcvad requirement).

    Returns:
        AudioResampler configured for 48kHz → 16kHz

    Example:
        ```python
        resampler = create_vad_resampler()
        frame_48k = get_client_audio()  # 1920 bytes @ 48kHz, 20ms
        frame_16k = resampler.process_frame(frame_48k)  # 640 bytes @ 16kHz, 20ms
        vad.process_frame(frame_16k)  # Feed to VAD
        ```
    """
    return AudioResampler(source_rate=48000, target_rate=16000)
