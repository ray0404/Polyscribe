"""
Polyphonic activation decoder and note event generator.
Converts neural network outputs (frames, onsets, contours) into MIDI note events.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

MIDI_OFFSET = 21  # Lowest MIDI pitch in 88-key piano (A0 = 21)
FFT_HOP = 256     # Frame hop size in samples
AUDIO_SAMPLE_RATE = 22050
FRAMES_PER_SECOND = AUDIO_SAMPLE_RATE / FFT_HOP  # ~86.13 frames/sec


def frame_to_time(frame_idx: int) -> float:
    """Convert frame index to time in seconds."""
    return frame_idx / FRAMES_PER_SECOND


def extract_pitch_bends(
    contours: np.ndarray, freq_idx: int, start_frame: int, end_frame: int
) -> List[Tuple[float, int]]:
    """
    Extract frame-by-frame pitch bend events for an active note.

    Args:
        contours: Array of shape (n_frames, 264)
        freq_idx: Pitch frequency index (0..87)
        start_frame: Note start frame index
        end_frame: Note end frame index

    Returns:
        List of (time_in_sec, pitchwheel_val) where pitchwheel_val is in [-8192, 8191]
    """
    pitch_bends = []
    c_start = freq_idx * 3
    c_end = c_start + 3

    for f in range(start_frame, end_frame):
        frame_contours = contours[f, c_start:c_end]
        c_sum = np.sum(frame_contours)
        if c_sum > 1e-4:
            # Weighted average offset in semitones (-1/3 to +1/3)
            weights = np.array([-0.3333, 0.0, 0.3333], dtype=np.float32)
            semitone_offset = float(np.sum(frame_contours * weights) / c_sum)
            # Map semitone offset to MIDI pitchwheel range [-8192, 8191] (assuming +/- 2 semitones range)
            pitchwheel_val = int(np.clip(semitone_offset / 2.0 * 8192.0, -8192, 8191))
            pitch_bends.append((frame_to_time(f), pitchwheel_val))
        else:
            pitch_bends.append((frame_to_time(f), 0))

    return pitch_bends


def decode_output_to_notes(
    frames: np.ndarray,
    onsets: np.ndarray,
    onset_thresh: float = 0.5,
    frame_thresh: float = 0.3,
    min_note_len: int = 11,
    infer_onsets: bool = True,
    contours: Optional[np.ndarray] = None
) -> List[Dict]:
    """
    Decode polyphonic frame & onset matrices into discrete MIDI note events.

    Args:
        frames: Array of shape (n_frames, 88) with frame activations [0.0, 1.0]
        onsets: Array of shape (n_frames, 88) with onset activations [0.0, 1.0]
        onset_thresh: Threshold for detecting note onset peak
        frame_thresh: Threshold for sustaining active note frames
        min_note_len: Minimum allowed note duration in frames
        infer_onsets: Infer additional onsets from frame activation deltas
        contours: Optional pitch bend contours array of shape (n_frames, 264)

    Returns:
        List of note dicts:
        [{
            'pitch': int (21..108),
            'start_time': float (seconds),
            'end_time': float (seconds),
            'velocity': float (0.0..1.0),
            'pitch_bends': Optional[List[Tuple[float, int]]]
        }]
    """
    n_frames, n_freqs = frames.shape
    note_events = []

    # Iterate over all 88 MIDI pitch bins
    for freq_idx in range(n_freqs):
        pitch = freq_idx + MIDI_OFFSET
        frame_track = frames[:, freq_idx]
        onset_track = onsets[:, freq_idx]

        in_note = False
        start_frame = 0
        max_velocity = 0.0

        for f in range(n_frames):
            frame_val = frame_track[f]
            onset_val = onset_track[f]

            is_onset = onset_val >= onset_thresh

            # Inferred onset detection: frame jump > 0.25 when sustained
            if not is_onset and infer_onsets and f > 0:
                if frame_val >= frame_thresh and (frame_val - frame_track[f - 1]) >= 0.25:
                    is_onset = True

            if is_onset:
                if in_note:
                    # End current note if new onset triggered
                    end_frame = f
                    if (end_frame - start_frame) >= min_note_len:
                        pb = extract_pitch_bends(contours, freq_idx, start_frame, end_frame) if contours is not None else None
                        note_events.append({
                            'pitch': pitch,
                            'start_time': frame_to_time(start_frame),
                            'end_time': frame_to_time(end_frame),
                            'velocity': min(1.0, max(0.1, max_velocity)),
                            'pitch_bends': pb
                        })
                # Start new note
                in_note = True
                start_frame = f
                max_velocity = max(onset_val, frame_val)
            elif in_note:
                if frame_val >= frame_thresh:
                    max_velocity = max(max_velocity, frame_val)
                else:
                    # End note when frame drops below threshold
                    end_frame = f
                    if (end_frame - start_frame) >= min_note_len:
                        pb = extract_pitch_bends(contours, freq_idx, start_frame, end_frame) if contours is not None else None
                        note_events.append({
                            'pitch': pitch,
                            'start_time': frame_to_time(start_frame),
                            'end_time': frame_to_time(end_frame),
                            'velocity': min(1.0, max(0.1, max_velocity)),
                            'pitch_bends': pb
                        })
                    in_note = False

        # Close any active note reaching end of audio stream
        if in_note:
            end_frame = n_frames
            if (end_frame - start_frame) >= min_note_len:
                pb = extract_pitch_bends(contours, freq_idx, start_frame, end_frame) if contours is not None else None
                note_events.append({
                    'pitch': pitch,
                    'start_time': frame_to_time(start_frame),
                    'end_time': frame_to_time(end_frame),
                    'velocity': min(1.0, max(0.1, max_velocity)),
                    'pitch_bends': pb
                })

    # Sort notes by start_time ascending
    note_events.sort(key=lambda x: (x['start_time'], x['pitch']))
    return note_events
