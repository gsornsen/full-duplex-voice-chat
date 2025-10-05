#!/usr/bin/env python3
"""Example usage of audio synthesis utilities."""

from src.tts.audio.synthesis import (
    calculate_frame_count,
    calculate_pcm_byte_size,
    generate_silence,
    generate_sine_wave,
    generate_sine_wave_frames,
)


def main() -> None:
    """Demonstrate audio synthesis utilities."""
    print("Audio Synthesis Examples\n" + "=" * 50)

    # Example 1: Generate a single sine wave
    print("\n1. Generate 100ms of 440Hz sine wave at 48kHz:")
    audio_bytes = generate_sine_wave(frequency=440, duration_ms=100, sample_rate=48000)
    print(f"   Generated {len(audio_bytes)} bytes")
    print(f"   Expected: {100 * 48 * 2} bytes (100ms × 48kHz × 2 bytes/sample)")

    # Example 2: Generate framed audio
    print("\n2. Generate 100ms framed into 20ms chunks:")
    frames = generate_sine_wave_frames(
        frequency=440, duration_ms=100, sample_rate=48000, frame_duration_ms=20
    )
    print(f"   Generated {len(frames)} frames")
    print(f"   Each frame: {len(frames[0])} bytes")
    print("   Expected: 5 frames of 1920 bytes each")

    # Example 3: Standard 20ms frame at 48kHz
    print("\n3. Standard 20ms frame at 48kHz:")
    frame = generate_sine_wave(frequency=440, duration_ms=20, sample_rate=48000)
    print(f"   Frame size: {len(frame)} bytes")
    print("   Expected: 1920 bytes (960 samples × 2 bytes)")

    # Example 4: Generate silence
    print("\n4. Generate 20ms of silence:")
    silence = generate_silence(duration_ms=20, sample_rate=48000)
    print(f"   Silence size: {len(silence)} bytes")
    print(f"   All zeros: {silence == b'\\x00' * 1920}")

    # Example 5: Calculate frame counts
    print("\n5. Calculate frame counts:")
    count_100 = calculate_frame_count(duration_ms=100, frame_duration_ms=20)
    count_105 = calculate_frame_count(duration_ms=105, frame_duration_ms=20)
    print(f"   100ms / 20ms = {count_100} frames")
    print(f"   105ms / 20ms = {count_105} frames (partial frame discarded)")

    # Example 6: Calculate PCM sizes
    print("\n6. Calculate PCM byte sizes:")
    mono_size = calculate_pcm_byte_size(duration_ms=20, sample_rate=48000, channels=1)
    stereo_size = calculate_pcm_byte_size(duration_ms=20, sample_rate=48000, channels=2)
    print(f"   20ms mono at 48kHz: {mono_size} bytes")
    print(f"   20ms stereo at 48kHz: {stereo_size} bytes")

    # Example 7: Different frequencies
    print("\n7. Generate different frequencies:")
    for freq in [220, 440, 880, 1760]:
        audio = generate_sine_wave(frequency=freq, duration_ms=20, sample_rate=48000)
        print(f"   {freq}Hz: {len(audio)} bytes (all same size)")

    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()
