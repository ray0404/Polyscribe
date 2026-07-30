"""
Classical Non-ML DSP engine for Polyphonic pitch estimation.
Uses Short-Time Fourier Transform (STFT) spectral peak analysis to estimate notes without neural network weights.
"""

import numpy as np
from typing import Dict

MIDI_OFFSET = 21
N_PITCHES = 88
FFT_HOP = 256
N_FFT = 2048
AUDIO_SAMPLE_RATE = 22050


def midi_to_freq(midi_pitch: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))


class DSPEngine:
    """
    Classical DSP polyphonic pitch estimator using STFT spectral energy mapping.
    """

    def __init__(self):
        # Precompute target frequencies for 88 MIDI keys (21..108)
        self.pitches = np.arange(MIDI_OFFSET, MIDI_OFFSET + N_PITCHES)
        self.target_freqs = np.array([midi_to_freq(p) for p in self.pitches], dtype=np.float32)

        # Precompute FFT frequency bin indices
        fft_freqs = np.fft.rfftfreq(N_FFT, d=1.0 / AUDIO_SAMPLE_RATE)
        self.pitch_bin_map = []
        for freq in self.target_freqs:
            bin_idx = np.argmin(np.abs(fft_freqs - freq))
            self.pitch_bin_map.append(bin_idx)
        self.pitch_bin_map = np.array(self.pitch_bin_map, dtype=int)

    def run(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute classical DSP STFT polyphonic pitch estimation.

        Args:
            audio: 1D float32 audio array at 22,050 Hz

        Returns:
            Dictionary with 'note' (T, 88), 'onset' (T, 88), 'contour' None
        """
        if audio.ndim > 1:
            audio = np.squeeze(audio)

        n_samples = len(audio)
        n_frames = max(1, (n_samples - N_FFT) // FFT_HOP + 1)

        window = np.hanning(N_FFT).astype(np.float32)
        note_frames = np.zeros((n_frames, N_PITCHES), dtype=np.float32)
        onset_frames = np.zeros((n_frames, N_PITCHES), dtype=np.float32)

        for f in range(n_frames):
            start = f * FFT_HOP
            end = start + N_FFT
            if end > n_samples:
                break
            frame_audio = audio[start:end] * window
            magnitude = np.abs(np.fft.rfft(frame_audio))

            # Normalize magnitude spectrum
            mag_max = np.max(magnitude) + 1e-6
            norm_mag = magnitude / mag_max

            # Map spectrum energy to 88 MIDI pitches
            pitch_salience = norm_mag[self.pitch_bin_map]
            note_frames[f] = pitch_salience

            # Compute onset energy jump relative to previous frame
            if f == 0:
                onset_frames[f] = pitch_salience
            else:
                delta = np.maximum(0.0, pitch_salience - note_frames[f - 1])
                onset_frames[f] = delta

        return {
            'note': note_frames,
            'onset': onset_frames,
            'contour': None
        }
