"""
Unit test for Classical Non-ML DSP Engine in Polyscribe.
"""

import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from polyscribe.dsp_engine import DSPEngine
from polyscribe.decoder import decode_output_to_notes
from polyscribe.midi_writer import export_notes_to_midi


def test_dsp_engine():
    print("[+] Testing DSPEngine (Classical Non-ML STFT Polyphonic Estimator)...")

    # Generate synthetic 440 Hz (A4) sine wave audio
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), False)
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    engine = DSPEngine()
    output = engine.run(audio)

    assert 'note' in output
    assert 'onset' in output
    assert output['note'].shape[1] == 88

    notes = decode_output_to_notes(
        frames=output['note'],
        onsets=output['onset'],
        onset_thresh=0.2,
        frame_thresh=0.2,
        min_note_len=5
    )

    print(f"    DSP Engine extracted {len(notes)} note events.")
    assert len(notes) > 0
    pitches = [n['pitch'] for n in notes]
    assert 69 in pitches  # A4 = MIDI pitch 69
    print(f"    Detected MIDI pitches: {pitches} (A4 = 69 detected!)")

    with tempfile.TemporaryDirectory() as tmpdir:
        midi_path = os.path.join(tmpdir, "test_dsp.mid")
        export_notes_to_midi(notes, midi_path)
        assert os.path.exists(midi_path)
        print(f"    DSP MIDI exported: {midi_path} ({os.path.getsize(midi_path)} bytes)")

    print("[✔] DSPEngine unit test passed successfully!")


if __name__ == "__main__":
    test_dsp_engine()
