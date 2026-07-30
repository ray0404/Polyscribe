"""
Unit test and sample audio generator for Polyscribe.
Generates a synthetic polyphonic C-major chord audio file (C4 + E4 + G4) and tests conversion.
"""

import os
import sys
import tempfile
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from polyscribe.audio import load_audio, chunk_audio
from polyscribe.decoder import decode_output_to_notes
from polyscribe.midi_writer import export_notes_to_midi


def generate_chord_wav(file_path: str, duration: float = 2.0, sr: int = 44100):
    """Generate a synthetic C-major triad (C4=261.63Hz, E4=329.63Hz, G4=392.00Hz)."""
    t = np.linspace(0, duration, int(sr * duration), False)

    c4 = 0.3 * np.sin(2 * np.pi * 261.63 * t)
    e4 = 0.3 * np.sin(2 * np.pi * 329.63 * t)
    g4 = 0.3 * np.sin(2 * np.pi * 392.00 * t)

    chord = (c4 + e4 + g4).astype(np.float32)

    # Apply fade in and fade out envelope
    envelope = np.ones_like(t)
    fade_len = int(sr * 0.05)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)

    audio = chord * envelope
    try:
        import soundfile as sf
        sf.write(file_path, audio, sr)
    except (ImportError, OSError):
        try:
            from scipy.io import wavfile
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(file_path, sr, audio_int16)
        except Exception:
            import wave
            audio_int16 = (audio * 32767).astype(np.int16)
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio_int16.tobytes())
    return file_path


def test_polyscribe_pipeline():
    print("[+] Testing Polyscribe audio loading & decoding...")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "test_chord.wav")
        midi_path = os.path.join(tmpdir, "test_chord.mid")

        generate_chord_wav(wav_path, duration=2.0)
        assert os.path.exists(wav_path)
        print(f"    Generated synthetic WAV: {wav_path}")

        # 1. Load Audio
        audio, sr = load_audio(wav_path)
        assert sr == 22050
        assert len(audio) > 0
        print(f"    Audio loaded and resampled to {sr} Hz: {len(audio)} samples")

        # 2. Test Synthetic Frame Matrix Decoding
        # Mock 170 frames (~2 seconds) of C4 (MIDI 60), E4 (MIDI 64), G4 (MIDI 67)
        # Note indices relative to MIDI 21 (A0): 60-21=39, 64-21=43, 67-21=46
        n_frames = 170
        frames_mock = np.zeros((n_frames, 88), dtype=np.float32)
        onsets_mock = np.zeros((n_frames, 88), dtype=np.float32)

        # Set onset at frame 5 for MIDI 60, 64, 67
        onsets_mock[5, [39, 43, 46]] = 0.8
        # Set sustained frame activations from frame 5 to 150
        frames_mock[5:150, [39, 43, 46]] = 0.7

        notes = decode_output_to_notes(
            frames=frames_mock,
            onsets=onsets_mock,
            onset_thresh=0.5,
            frame_thresh=0.3,
            min_note_len=11
        )

        print(f"    Decoded {len(notes)} mock note events.")
        assert len(notes) == 3
        pitches = sorted([n['pitch'] for n in notes])
        assert pitches == [60, 64, 67]
        print(f"    Extracted MIDI Pitches: {pitches} (C4, E4, G4 - C Major Triad!)")

        # 3. Test Export MIDI
        exported_file = export_notes_to_midi(notes, midi_path, bpm=120.0)
        assert os.path.exists(exported_file)
        assert os.path.getsize(exported_file) > 0
        print(f"    Successfully exported MIDI file: {exported_file} ({os.path.getsize(exported_file)} bytes)")

    print("[✔] All Polyscribe component tests passed successfully!")


if __name__ == "__main__":
    test_polyscribe_pipeline()
