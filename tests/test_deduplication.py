"""
Unit test for note deduplication across overlapping audio chunks.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from polyscribe.decoder import deduplicate_notes


def test_deduplicate_notes():
    print("[+] Testing note deduplication across overlapping chunk boundaries...")

    # Duplicate note pair (same pitch 60, start times 29.5s and 29.52s)
    notes = [
        {'pitch': 60, 'start_time': 29.50, 'end_time': 30.1, 'velocity': 0.8},
        {'pitch': 60, 'start_time': 29.52, 'end_time': 31.2, 'velocity': 0.8}, # Longer duration note from chunk 2
        {'pitch': 64, 'start_time': 10.00, 'end_time': 12.0, 'velocity': 0.7}, # Non-duplicate note
    ]

    deduped = deduplicate_notes(notes, time_tolerance=0.05)

    assert len(deduped) == 2
    pitches = [n['pitch'] for n in deduped]
    assert pitches == [64, 60]

    # Verify that the longer duration note (end_time 31.2) was retained
    note_60 = [n for n in deduped if n['pitch'] == 60][0]
    assert note_60['end_time'] == 31.2
    print("    Successfully merged duplicate boundary notes!")
    print("[✔] Deduplication unit test passed successfully!")


if __name__ == "__main__":
    test_deduplicate_notes()
