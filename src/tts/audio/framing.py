"""Audio framing utilities for 20ms frames at 48kHz."""

import numpy as np
from numpy.typing import NDArray


def repacketize_to_20ms(
    audio: NDArray[np.float32], sample_rate: int, target_sample_rate: int = 48000
) -> list[bytes]:
    """Repacketize audio to 20ms frames at target sample rate.

    Args:
        audio: Input audio array
        sample_rate: Input sample rate
        target_sample_rate: Target sample rate (default 48kHz)

    Returns:
        List of 20ms PCM frame bytes
    """
    return []
