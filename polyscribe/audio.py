"""
Audio processing and decoding utilities for Polyscribe.
"""

import sys
import numpy as np
from typing import Tuple, List, Generator

TARGET_SAMPLE_RATE = 22050
# Basic Pitch expects 43,844 samples (~1.98 seconds per frame batch window)
MODEL_WINDOW_SAMPLES = 43844
MODEL_HOP_SAMPLES = 11025  # ~0.5s hop for inference alignment


def load_audio_ffmpeg(file_path: str, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Decode audio file using ffmpeg CLI pipe to 16-bit mono PCM stream at target_sr Hz.
    """
    import subprocess

    cmd = [
        'ffmpeg', '-nostdin', '-loglevel', 'quiet',
        '-i', file_path,
        '-f', 's16le',
        '-ac', '1',
        '-ar', str(target_sr),
        '-'
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_bytes, stderr = process.communicate()

    if process.returncode != 0 or len(raw_bytes) == 0:
        raise RuntimeError(f"FFmpeg decoding failed for '{file_path}': {stderr.decode('utf-8', errors='ignore')}")

    data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return data, target_sr


def load_audio(file_path: str, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, downmix to mono, and resample to target_sr Hz.

    Args:
        file_path: Path to input audio file (.wav, .flac, .ogg, etc.)
        target_sr: Target sample rate in Hz (default 22050)

    Returns:
        Tuple of (mono_float32_audio_array, target_sample_rate)
    """
    data = None
    orig_sr = target_sr

    try:
        import soundfile as sf
        data, orig_sr = sf.read(file_path, dtype='float32')
    except (ImportError, OSError):
        try:
            from scipy.io import wavfile
            orig_sr, raw_data = wavfile.read(file_path)
            if raw_data.dtype == np.int16:
                data = raw_data.astype(np.float32) / 32768.0
            elif raw_data.dtype == np.int32:
                data = raw_data.astype(np.float32) / 2147483648.0
            elif raw_data.dtype == np.uint8:
                data = (raw_data.astype(np.float32) - 128.0) / 128.0
            else:
                data = raw_data.astype(np.float32)
        except Exception:
            try:
                # Native Python wave module fallback for PCM WAV
                import wave
                with wave.open(file_path, 'rb') as wf:
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    orig_sr = wf.getframerate()
                    n_frames = wf.getnframes()
                    raw_bytes = wf.readframes(n_frames)

                    if sampwidth == 2:
                        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sampwidth == 4:
                        data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
                    elif sampwidth == 1:
                        data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                    else:
                        raise ValueError(f"Unsupported sample width: {sampwidth}")

                    if n_channels > 1:
                        data = data.reshape(-1, n_channels).mean(axis=1)
            except Exception:
                # Universal fallback using ffmpeg CLI pipe
                try:
                    data, orig_sr = load_audio_ffmpeg(file_path, target_sr)
                except Exception as e:
                    raise RuntimeError(
                        f"Could not decode audio file '{file_path}'. "
                        f"Please ensure soundfile, scipy, or ffmpeg is installed."
                    ) from e

    if data is None:
        raise RuntimeError(f"Could not decode audio file '{file_path}'. Please install soundfile or ffmpeg.")

    # Downmix multi-channel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if orig_sr != target_sr:
        data = resample_audio(data, orig_sr, target_sr)

    return data.astype(np.float32), target_sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio array using scipy resample_poly or numpy linear interpolation fallback.
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    try:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        resampled = resample_poly(audio, up, down)
        return resampled.astype(np.float32)
    except ImportError:
        # Fallback to linear interpolation if scipy is not available
        num_target_samples = int(len(audio) * target_sr / orig_sr)
        orig_indices = np.linspace(0, len(audio) - 1, num=len(audio))
        target_indices = np.linspace(0, len(audio) - 1, num=num_target_samples)
        return np.interp(target_indices, orig_indices, audio).astype(np.float32)


def chunk_audio(
    audio: np.ndarray, chunk_seconds: float = 30.0, overlap_seconds: float = 1.0, sr: int = TARGET_SAMPLE_RATE
) -> Generator[Tuple[np.ndarray, float, float], None, None]:
    """
    Generator that splits long audio into overlapping chunks for memory-safe processing.

    Yields:
        Tuple of (chunk_audio_array, start_time_seconds, end_time_seconds)
    """
    chunk_samples = int(chunk_seconds * sr)
    overlap_samples = int(overlap_seconds * sr)
    step_samples = chunk_samples - overlap_samples

    total_samples = len(audio)
    if total_samples <= chunk_samples:
        yield audio, 0.0, total_samples / sr
        return

    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        start_sec = start / sr
        end_sec = end / sr

        yield chunk, start_sec, end_sec

        if end == total_samples:
            break
        start += step_samples
