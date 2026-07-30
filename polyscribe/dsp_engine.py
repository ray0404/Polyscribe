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
        Execute vectorized classical DSP STFT polyphonic pitch estimation.

        Args:
            audio: 1D float32 audio array at 22,050 Hz

        Returns:
            Dictionary with 'note' (T, 88), 'onset' (T, 88), 'contour' None
        """
        if audio.ndim > 1:
            audio = np.squeeze(audio)

        n_samples = len(audio)
        if n_samples < N_FFT:
            audio = np.pad(audio, (0, N_FFT - n_samples))
            n_samples = len(audio)

        n_frames = max(1, (n_samples - N_FFT) // FFT_HOP + 1)
        valid_len = (n_frames - 1) * FFT_HOP + N_FFT
        audio_crop = audio[:valid_len]

        window = np.hanning(N_FFT).astype(np.float32)

        # Vectorized strided 2D frame view: shape (n_frames, N_FFT)
        shape = (n_frames, N_FFT)
        strides = (audio_crop.strides[0] * FFT_HOP, audio_crop.strides[0])
        frames_matrix = np.lib.stride_tricks.as_strided(audio_crop, shape=shape, strides=strides)

        # Compute vectorized RFFT across all frames simultaneously
        windowed_frames = frames_matrix * window
        magnitudes = np.abs(np.fft.rfft(windowed_frames, axis=-1))

        # Normalize magnitude spectrum per frame
        mag_max = np.max(magnitudes, axis=-1, keepdims=True) + 1e-6
        norm_mag = magnitudes / mag_max

        # Map spectral energy to 88 MIDI pitch bins: shape (n_frames, 88)
        note_frames = norm_mag[:, self.pitch_bin_map].astype(np.float32)

        # Compute vectorized onset energy deltas
        onset_frames = np.zeros_like(note_frames)
        onset_frames[0] = note_frames[0]
        if n_frames > 1:
            onset_frames[1:] = np.maximum(0.0, note_frames[1:] - note_frames[:-1])

        return {
            'note': note_frames,
            'onset': onset_frames,
            'contour': None
        }
