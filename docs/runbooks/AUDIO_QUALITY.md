# Runbook: Audio Quality Troubleshooting

**Time to Resolution:** 15-30 minutes
**Severity:** High (affects user experience)
**Related:** [Audio Backpressure](AUDIO_BACKPRESSURE.md), [Log Debugging](LOG_DEBUGGING.md), [Advanced Troubleshooting](ADVANCED_TROUBLESHOOTING.md)

---

## Overview

This runbook addresses audio quality issues in the M2 TTS system, including distortion, clipping, noise, and other artifacts that affect the listening experience.

**Audio Pipeline:**
```
Model → Raw samples → Normalization → Resampling → Framing → WebSocket → Client Playback
```

Each stage can introduce quality issues.

---

## Quick Diagnostic Checklist

```bash
# 1. Check for clipping in logs
docker logs tts-worker | grep -i "clip\|distort"

# 2. Verify sample rate configuration
grep "output_sample_rate" configs/worker.yaml
# Should be: 48000

# 3. Check normalization settings
grep "normalization" configs/worker.yaml

# 4. Look for frame drops
docker logs orchestrator | grep -i "frame.*drop"

# 5. Verify no resampling errors
docker logs tts-worker | grep -i "resample.*error"

# 6. Check loudness measurements
docker logs tts-worker | jq 'select(.loudness_lufs?) | {timestamp, loudness_lufs}'
```

---

## Common Issues

### 1. Distorted/Clipped Audio

**Symptom:** Audio sounds harsh, distorted, or "crunchy"

**Cause:** Sample values exceeding int16 range (-32768 to 32767)

**Diagnostic:**

```python
# Check for clipping in audio samples
import numpy as np

def check_clipping(samples: np.ndarray, threshold: float = 0.99):
    """Check if audio is clipping.

    Args:
        samples: Audio samples (int16 or float)
        threshold: Clipping threshold (0.99 = 99% of max value)

    Returns:
        Clipping statistics
    """
    if samples.dtype == np.int16:
        max_val = 32767
        samples_norm = samples.astype(np.float32) / max_val
    else:
        max_val = 1.0
        samples_norm = samples

    clipped = np.abs(samples_norm) >= threshold
    clipping_pct = (np.sum(clipped) / len(samples)) * 100

    return {
        'clipping_pct': clipping_pct,
        'max_value': np.max(np.abs(samples_norm)),
        'samples_clipped': np.sum(clipped),
        'total_samples': len(samples)
    }

# Example usage
stats = check_clipping(audio_samples)
print(f"Clipping: {stats['clipping_pct']:.2f}%")
```

**Resolution:**

**Option A: Adjust normalization target**

```yaml
# configs/worker.yaml
audio:
  loudness_target_lufs: -18.0  # Reduce from -16.0 (more headroom)
  normalization_enabled: true
```

**Option B: Add soft clipping**

```python
def soft_clip(samples: np.ndarray, threshold: float = 0.9):
    """Apply soft clipping to prevent hard distortion.

    Args:
        samples: Audio samples (normalized -1 to 1)
        threshold: Soft clipping threshold

    Returns:
        Soft-clipped samples
    """
    # Tanh-based soft clipping
    clipped = np.where(
        np.abs(samples) > threshold,
        np.tanh((samples - threshold * np.sign(samples)) / (1 - threshold)) * (1 - threshold) + threshold * np.sign(samples),
        samples
    )
    return clipped
```

**Option C: Limiter**

```python
def apply_limiter(samples: np.ndarray, threshold: float = -1.0, release: int = 100):
    """Apply dynamic range limiter.

    Args:
        samples: Audio samples
        threshold: Threshold in dB
        release: Release time in samples

    Returns:
        Limited samples
    ```
    import numpy as np

    threshold_linear = 10 ** (threshold / 20)
    gain = 1.0
    output = np.zeros_like(samples)

    for i, sample in enumerate(samples):
        if abs(sample) > threshold_linear:
            target_gain = threshold_linear / abs(sample)
            gain = min(gain, target_gain)
        else:
            gain = min(1.0, gain + (1.0 - gain) / release)

        output[i] = sample * gain

    return output
```

---

### 2. Choppy/Robotic Audio

**Symptom:** Audio sounds stuttery, robotic, or has gaps

**Causes:**
- Frame drops
- Timing jitter
- Resampling artifacts
- Playback buffer underruns

**Diagnostic:**

```bash
# Check for frame drops
docker logs orchestrator | grep -E "drop|gap|skip"

# Check frame timing
docker logs orchestrator | jq 'select(.frame_interval_ms?) | {timestamp, interval: .frame_interval_ms}'
# Should be ~20ms consistently

# Check for resampling warnings
docker logs tts-worker | grep -i resample
```

**Client-side diagnostics:**

```javascript
// Browser console: Check playback timing
let lastPlayTime = null;

audioContext.addEventListener('statechange', () => {
    const now = audioContext.currentTime;
    if (lastPlayTime) {
        const gap = (now - lastPlayTime) * 1000;
        if (gap > 25) {  // > 25ms gap
            console.warn(`Playback gap: ${gap.toFixed(1)}ms`);
        }
    }
    lastPlayTime = now;
});
```

**Resolution:**

**Option A: Increase client buffer**

```javascript
// Client: Add playback buffer
const MIN_BUFFER_MS = 200;  // Buffer 200ms before playback

class AudioBufferManager {
    constructor(minBufferMs = 200) {
        this.buffer = [];
        this.minBufferMs = minBufferMs;
        this.playing = false;
    }

    addFrame(frame) {
        this.buffer.push(frame);
        if (!this.playing && this.bufferDuration() >= this.minBufferMs) {
            this.startPlayback();
        }
    }

    bufferDuration() {
        return this.buffer.length * 20;  // 20ms per frame
    }

    startPlayback() {
        this.playing = true;
        this.playNext();
    }

    playNext() {
        if (this.buffer.length === 0) {
            console.warn('Buffer underrun');
            this.playing = false;
            return;
        }

        const frame = this.buffer.shift();
        playAudioFrame(frame);

        setTimeout(() => this.playNext(), 20);
    }
}
```

**Option B: Fix frame timing on server**

See [Audio Backpressure Runbook](AUDIO_BACKPRESSURE.md)

---

### 3. Noisy/Hissy Audio

**Symptom:** Background hiss, static, or noise

**Causes:**
- Model inference noise
- Quantization errors
- Electrical interference (rare in digital)
- Low-quality model

**Diagnostic:**

```python
def analyze_noise_floor(samples: np.ndarray, sample_rate: int = 48000):
    """Measure noise floor in silent portions.

    Args:
        samples: Audio samples
        sample_rate: Sample rate in Hz

    Returns:
        Noise floor statistics
    """
    import numpy as np

    # Find silent regions (RMS < -40 dBFS)
    frame_size = int(sample_rate * 0.02)  # 20ms frames
    frames = samples.reshape(-1, frame_size)

    rms_db = []
    for frame in frames:
        rms = np.sqrt(np.mean(frame ** 2))
        rms_db.append(20 * np.log10(rms + 1e-10))

    silent_frames = [db for db in rms_db if db < -40]

    if silent_frames:
        noise_floor = np.mean(silent_frames)
        print(f"Noise floor: {noise_floor:.1f} dBFS")
        print(f"Silent frames: {len(silent_frames)}/{len(rms_db)}")
    else:
        print("No silent frames detected")

    return {'noise_floor_db': noise_floor if silent_frames else None}
```

**Resolution:**

**Option A: Noise gate**

```python
def apply_noise_gate(samples: np.ndarray, threshold_db: float = -50, attack: int = 10, release: int = 100):
    """Apply noise gate to reduce background noise.

    Args:
        samples: Audio samples
        threshold_db: Gate threshold in dB
        attack: Attack time in samples
        release: Release time in samples

    Returns:
        Gated samples
    """
    import numpy as np

    threshold_linear = 10 ** (threshold_db / 20)
    gate_gain = 0.0
    output = np.zeros_like(samples, dtype=np.float32)

    for i, sample in enumerate(samples):
        if abs(sample) > threshold_linear:
            # Attack: Open gate quickly
            gate_gain = min(1.0, gate_gain + (1.0 / attack))
        else:
            # Release: Close gate slowly
            gate_gain = max(0.0, gate_gain - (1.0 / release))

        output[i] = sample * gate_gain

    return output
```

**Option B: Spectral subtraction**

```python
def spectral_subtraction(samples: np.ndarray, sample_rate: int = 48000, noise_profile: np.ndarray = None):
    """Reduce noise using spectral subtraction.

    Args:
        samples: Audio samples
        sample_rate: Sample rate
        noise_profile: Noise spectrum (from silent portion)

    Returns:
        Denoised samples
    """
    import numpy as np
    from scipy import signal

    # Compute STFT
    f, t, Zxx = signal.stft(samples, fs=sample_rate, nperseg=1024)

    if noise_profile is None:
        # Estimate noise from first 100ms
        noise_frames = int(0.1 * sample_rate / 1024)
        noise_profile = np.mean(np.abs(Zxx[:, :noise_frames]), axis=1)

    # Subtract noise
    Zxx_clean = np.maximum(np.abs(Zxx) - noise_profile[:, np.newaxis], 0) * np.exp(1j * np.angle(Zxx))

    # Inverse STFT
    _, samples_clean = signal.istft(Zxx_clean, fs=sample_rate, nperseg=1024)

    return samples_clean
```

---

### 4. Volume Too Low/High

**Symptom:** Audio too quiet or too loud

**Diagnostic:**

```bash
# Check loudness normalization
docker logs tts-worker | jq 'select(.loudness_lufs?) | {timestamp, model_id, loudness_lufs, target_lufs}'

# Verify normalization enabled
grep "normalization_enabled" configs/worker.yaml
```

**Measure loudness:**

```python
def measure_loudness_lufs(samples: np.ndarray, sample_rate: int = 48000):
    """Measure integrated loudness in LUFS (ITU-R BS.1770).

    Args:
        samples: Audio samples
        sample_rate: Sample rate in Hz

    Returns:
        LUFS measurement
    """
    import pyloudnorm as pyln

    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(samples)

    return loudness
```

**Resolution:**

**Adjust normalization target:**

```yaml
# configs/worker.yaml
audio:
  loudness_target_lufs: -16.0  # Standard: -16 LUFS
  # -23 LUFS: Broadcasting standard (quieter)
  # -14 LUFS: Spotify/streaming (louder)
  normalization_enabled: true
```

**Client-side volume control:**

```javascript
// Browser: Audio gain node
const gainNode = audioContext.createGain();
gainNode.gain.value = 0.8;  // 80% volume

audioSource.connect(gainNode);
gainNode.connect(audioContext.destination);
```

---

### 5. Resampling Artifacts

**Symptom:** Metallic, aliased, or "underwater" sound

**Cause:** Poor resampling quality or incorrect sample rate

**Diagnostic:**

```bash
# Check sample rate config
grep "output_sample_rate" configs/worker.yaml
# Must be: 48000

# Check resampling logs
docker logs tts-worker | grep -i resample

# Verify no rate mismatches
docker logs tts-worker | jq 'select(.sample_rate?) | {model_id, model_rate: .sample_rate, target_rate: .output_sample_rate}'
```

**Resolution:**

**Use high-quality resampler:**

```python
import librosa
import numpy as np

def resample_high_quality(samples: np.ndarray, orig_sr: int, target_sr: int):
    """Resample using high-quality sinc interpolation.

    Args:
        samples: Input samples
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled samples
    """
    # Use librosa's high-quality resampler (Kaiser windowed sinc)
    resampled = librosa.resample(
        samples,
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type='kaiser_best'  # Highest quality
    )
    return resampled
```

**Or use scipy:**

```python
from scipy import signal

def resample_scipy(samples: np.ndarray, orig_sr: int, target_sr: int):
    """Resample using scipy's polyphase filter.

    Args:
        samples: Input samples
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled samples
    """
    num_samples = int(len(samples) * target_sr / orig_sr)
    resampled = signal.resample_poly(samples, target_sr, orig_sr)
    return resampled
```

---

### 6. Audio-Visual Desync

**Symptom:** Audio doesn't match visual cues (if video present)

**Cause:** Cumulative latency, buffering delays

**Diagnostic:**

```javascript
// Browser: Measure A/V sync
const visualEventTime = performance.now();

// When corresponding audio plays
audioContext.addEventListener('play', () => {
    const audioEventTime = performance.now();
    const avSync = audioEventTime - visualEventTime;
    console.log(`A/V sync: ${avSync}ms`);
    // Should be < 100ms for good sync
});
```

**Resolution:**

**Timestamp-based sync:**

```javascript
// Server sends timestamps
{
    type: 'audio_frame',
    data: audioData,
    timestamp: serverTime,  // Server timestamp
    sequence: frameNumber
}

// Client adjusts playback timing
const clientReceiveTime = performance.now();
const latency = clientReceiveTime - frame.timestamp;

// Adjust playback to compensate
const playTime = audioContext.currentTime + (TARGET_LATENCY - latency) / 1000;
scheduleAudioFrame(frame.data, playTime);
```

---

## Audio Quality Metrics

### Objective Metrics

**PESQ (Perceptual Evaluation of Speech Quality):**

```python
from pesq import pesq

def measure_pesq(reference: np.ndarray, degraded: np.ndarray, sample_rate: int = 48000):
    """Measure PESQ score.

    Args:
        reference: Reference audio
        degraded: Degraded audio
        sample_rate: Sample rate (8000 or 16000 for PESQ)

    Returns:
        PESQ score (1.0 to 4.5, higher is better)
    """
    # PESQ requires 8kHz or 16kHz
    if sample_rate not in [8000, 16000]:
        from scipy import signal
        degraded = signal.resample_poly(degraded, 16000, sample_rate)
        reference = signal.resample_poly(reference, 16000, sample_rate)
        sample_rate = 16000

    score = pesq(sample_rate, reference, degraded, 'wb')  # Wideband
    return score
```

**STOI (Short-Time Objective Intelligibility):**

```python
from pystoi import stoi

def measure_stoi(reference: np.ndarray, degraded: np.ndarray, sample_rate: int = 48000):
    """Measure STOI score.

    Args:
        reference: Reference audio (clean speech)
        degraded: Degraded audio
        sample_rate: Sample rate

    Returns:
        STOI score (0 to 1, higher is better)
    """
    score = stoi(reference, degraded, sample_rate, extended=False)
    return score
```

**SNR (Signal-to-Noise Ratio):**

```python
def measure_snr(clean: np.ndarray, noisy: np.ndarray):
    """Measure SNR in dB.

    Args:
        clean: Clean signal
        noisy: Noisy signal

    Returns:
        SNR in dB
    """
    import numpy as np

    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr_db
```

---

### Subjective Metrics

**MOS (Mean Opinion Score):**

Requires human listeners to rate quality on 1-5 scale:
- 5: Excellent
- 4: Good
- 3: Fair
- 2: Poor
- 1: Bad

**Automated MOS estimation:**

```python
# Use ViSQOL (Virtual Speech Quality Objective Listener)
# https://github.com/google/visqol

from visqol import visqol_lib_py

def estimate_mos(reference: np.ndarray, degraded: np.ndarray, sample_rate: int = 48000):
    """Estimate MOS using ViSQOL.

    Args:
        reference: Reference audio
        degraded: Degraded audio
        sample_rate: Sample rate

    Returns:
        MOS estimate (1-5)
    """
    config = visqol_lib_py.VisqolConfig()
    config.audio.sample_rate = sample_rate
    config.options.use_speech_scoring = True

    visqol = visqol_lib_py.VisqolApi()
    visqol.Create(config)

    result = visqol.Measure(reference, degraded)
    return result.moslqo  # MOS-LQO score
```

---

## Spectral Analysis

### Visualize Audio Spectrum

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def plot_spectrogram(samples: np.ndarray, sample_rate: int = 48000):
    """Plot spectrogram of audio.

    Args:
        samples: Audio samples
        sample_rate: Sample rate
    """
    f, t, Sxx = signal.spectrogram(samples, sample_rate, nperseg=1024)

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Power [dB]')
    plt.title('Spectrogram')
    plt.ylim(0, 8000)  # Focus on speech range
    plt.show()
```

**Identify issues from spectrogram:**

- **Clipping:** Horizontal bands at high frequencies
- **Noise:** Consistent low-level noise floor across all frequencies
- **Aliasing:** Repetitive patterns above Nyquist frequency
- **Resampling artifacts:** Gaps or distortions in high frequencies

---

## Testing Audio Quality

### Automated Quality Tests

```python
import pytest
import numpy as np

def test_no_clipping():
    """Verify synthesized audio doesn't clip."""
    audio = synthesize_text("Test audio quality")
    max_value = np.max(np.abs(audio))
    assert max_value < 0.99, f"Audio clipping detected: {max_value}"

def test_loudness_normalized():
    """Verify loudness normalization."""
    import pyloudnorm as pyln

    audio = synthesize_text("Test loudness normalization")
    meter = pyln.Meter(48000)
    loudness = meter.integrated_loudness(audio)

    # Target: -16 LUFS ± 1 dB
    assert -17 <= loudness <= -15, f"Loudness {loudness} LUFS outside target range"

def test_no_silence():
    """Verify audio contains actual signal."""
    audio = synthesize_text("Test audio presence")
    rms = np.sqrt(np.mean(audio ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)

    assert rms_db > -40, f"Audio too quiet: {rms_db} dBFS"

def test_proper_sample_rate():
    """Verify output sample rate."""
    audio, sample_rate = synthesize_with_sr("Test sample rate")
    assert sample_rate == 48000, f"Wrong sample rate: {sample_rate}"

def test_stereo_correlation():
    """Verify stereo channels are properly correlated (if stereo)."""
    audio_stereo = synthesize_stereo("Test stereo correlation")

    left = audio_stereo[:, 0]
    right = audio_stereo[:, 1]

    correlation = np.corrcoef(left, right)[0, 1]
    assert 0.9 <= correlation <= 1.0, f"Poor stereo correlation: {correlation}"
```

---

## Troubleshooting Workflow

**Step-by-step quality investigation:**

```bash
# 1. Generate test audio
echo "Testing audio quality" | uv run python src/client/cli_client.py > test-audio.wav

# 2. Analyze with script
python << 'EOF'
import numpy as np
import soundfile as sf

# Load audio
audio, sr = sf.read('test-audio.wav')

# Check clipping
clipping = np.sum(np.abs(audio) > 0.99) / len(audio) * 100
print(f"Clipping: {clipping:.2f}%")

# Check RMS level
rms = np.sqrt(np.mean(audio ** 2))
rms_db = 20 * np.log10(rms)
print(f"RMS level: {rms_db:.1f} dBFS")

# Check peak level
peak = np.max(np.abs(audio))
peak_db = 20 * np.log10(peak)
print(f"Peak level: {peak_db:.1f} dBFS")

# Check crest factor (peak/rms ratio)
crest_factor = peak / (rms + 1e-10)
print(f"Crest factor: {crest_factor:.1f}")

# Frequency analysis
from scipy import fft
spectrum = np.abs(fft.rfft(audio))
freqs = fft.rfftfreq(len(audio), 1/sr)

# Check frequency range
energy_by_band = {
    'sub-bass (20-60 Hz)': np.sum(spectrum[(freqs >= 20) & (freqs < 60)]),
    'bass (60-250 Hz)': np.sum(spectrum[(freqs >= 60) & (freqs < 250)]),
    'low-mid (250-500 Hz)': np.sum(spectrum[(freqs >= 250) & (freqs < 500)]),
    'mid (500-2k Hz)': np.sum(spectrum[(freqs >= 500) & (freqs < 2000)]),
    'high-mid (2k-4k Hz)': np.sum(spectrum[(freqs >= 2000) & (freqs < 4000)]),
    'presence (4k-8k Hz)': np.sum(spectrum[(freqs >= 4000) & (freqs < 8000)]),
    'brilliance (8k+ Hz)': np.sum(spectrum[freqs >= 8000])
}

for band, energy in energy_by_band.items():
    print(f"{band}: {energy:.0f}")
EOF
```

---

## Related Runbooks

- **[Audio Backpressure](AUDIO_BACKPRESSURE.md)** - Frame delivery issues
- **[Log Debugging](LOG_DEBUGGING.md)** - Log analysis
- **[Monitoring](MONITORING.md)** - Quality metrics tracking
- **[Advanced Troubleshooting](ADVANCED_TROUBLESHOOTING.md)** - Deep diagnostics

---

## Further Help

**Quick quality check:**

```bash
# Generate test audio
just cli << EOF
Testing one two three
EOF

# Analyze quality
uv run python scripts/analyze-audio.py test-output.wav

# Compare with reference
uv run python scripts/compare-audio.py reference.wav test-output.wav
```

**Best practices:**

1. Enable loudness normalization
2. Monitor for clipping
3. Use high-quality resampling
4. Buffer audio client-side
5. Measure objective metrics (PESQ, STOI)
6. Conduct listening tests

**Still experiencing quality issues?**

1. Check logs: `docker logs tts-worker | grep -i "audio\|clip\|loudness"`
2. Analyze spectrum: Use spectral analysis tools
3. Measure metrics: PESQ, STOI, SNR
4. Compare models: Test different TTS models
5. File issue with: Audio samples, logs, configuration
