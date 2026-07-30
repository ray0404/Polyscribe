"""
Unit test for Polyscribe pitch bend extraction and MIDI pitchwheel event generation.
"""

import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from polyscribe.decoder import decode_output_to_notes
from polyscribe.midi_writer import export_notes_to_midi


def test_pitch_bend_extraction():
    print("[+] Testing pitch bend extraction and MIDI pitchwheel export...")
    n_frames = 100
    frames = np.zeros((n_frames, 88), dtype=np.float32)
    onsets = np.zeros((n_frames, 88), dtype=np.float32)
    contours = np.zeros((n_frames, 264), dtype=np.float32)

    # Pitch A4 (MIDI 69 -> bin 48)
    pitch_idx = 48
    onsets[5, pitch_idx] = 0.9
    frames[5:80, pitch_idx] = 0.8

    # Add pitch bend contour shift (+1/3 semitone on frames 20 to 60)
    c_start = pitch_idx * 3
    contours[20:60, c_start + 2] = 0.9  # Bin 2 = +1/3 semitone bend

    notes = decode_output_to_notes(
        frames=frames,
        onsets=onsets,
        onset_thresh=0.5,
        frame_thresh=0.3,
        min_note_len=11,
        contours=contours
    )

    assert len(notes) == 1
    note = notes[0]
    assert note['pitch'] == 69
    assert note['pitch_bends'] is not None
    assert len(note['pitch_bends']) > 0

    # Verify pitchwheel events contain positive bend values during frame 20..60
    bends = [val for time_sec, val in note['pitch_bends'] if val != 0]
    assert len(bends) > 0
    assert all(b > 0 for b in bends)
    print(f"    Extracted {len(bends)} active pitch bend values (Max bend: {max(bends)})")

    # Export to MIDI
    with tempfile.TemporaryDirectory() as tmpdir:
        midi_path = os.path.join(tmpdir, "test_pitchbend.mid")
        export_notes_to_midi(notes, midi_path)
        assert os.path.exists(midi_path)
        print(f"    Exported MIDI with pitchwheel events: {midi_path} ({os.path.getsize(midi_path)} bytes)")

    print("[✔] Pitch bend unit test passed successfully!")


if __name__ == "__main__":
    test_pitch_bend_extraction()
