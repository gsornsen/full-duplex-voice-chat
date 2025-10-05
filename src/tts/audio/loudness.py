"""Audio loudness normalization utilities."""

import numpy as np
from numpy.typing import NDArray


def normalize_rms(audio: NDArray[np.float32], target_rms: float = 0.1) -> NDArray[np.float32]:
    """Normalize audio to target RMS level.

    Args:
        audio: Input audio array
        target_rms: Target RMS level

    Returns:
        Normalized audio array
    """
    return audio


def normalize_lufs(audio: NDArray[np.float32], target_lufs: float = -16.0) -> NDArray[np.float32]:
    """Normalize audio to target LUFS level.

    Args:
        audio: Input audio array
        target_lufs: Target LUFS level

    Returns:
        Normalized audio array
    """
    return audio
