"""
MIDI file export module using mido.
Converts extracted note events into Standard MIDI Files (.mid).
"""

from typing import List, Dict
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


def export_notes_to_midi(
    notes: List[Dict],
    output_path: str,
    bpm: float = 120.0,
    ticks_per_beat: int = 480
) -> str:
    """
    Convert note event dicts into a Standard MIDI File (SMF Format 0).

    Args:
        notes: List of note dicts with pitch, start_time, end_time, velocity
        output_path: Output file path (.mid)
        bpm: Beats per minute tempo
        ticks_per_beat: Ticks per beat resolution (PPQN)

    Returns:
        Path to written .mid file
    """
    mid = MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    # Set tempo message
    microseconds_per_beat = mido.bpm2tempo(bpm)
    track.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    track.append(MetaMessage('track_name', name='Polyscribe Polyphonic Audio', time=0))

    def sec_to_ticks(seconds: float) -> int:
        return int(seconds * (bpm / 60.0) * ticks_per_beat)

    # Build timed list of all MIDI events (Note On, Note Off, Pitchwheel)
    events = []
    for note in notes:
        start_tick = sec_to_ticks(note['start_time'])
        end_tick = sec_to_ticks(note['end_time'])
        pitch = int(note['pitch'])
        velocity = int(note['velocity'] * 127)

        # Guarantee valid MIDI velocity (1..127 for note_on)
        velocity = max(1, min(127, velocity))

        events.append((start_tick, 'note_on', pitch, velocity))

        # Include pitch bend events if present
        if note.get('pitch_bends'):
            for pb_time, bend_val in note['pitch_bends']:
                pb_tick = sec_to_ticks(pb_time)
                if start_tick <= pb_tick <= end_tick:
                    events.append((pb_tick, 'pitchwheel', pitch, bend_val))

        events.append((end_tick, 'note_off', pitch, 0))

    # Sort events chronologically. For equal ticks: pitchwheel < note_off < note_on
    order = {'pitchwheel': 0, 'note_off': 1, 'note_on': 2}
    events.sort(key=lambda x: (x[0], order.get(x[1], 1)))

    last_tick = 0
    for tick, event_type, pitch, value in events:
        delta_ticks = max(0, tick - last_tick)
        if event_type == 'pitchwheel':
            track.append(Message('pitchwheel', pitch=value, time=delta_ticks))
        else:
            track.append(Message(event_type, note=pitch, velocity=value, time=delta_ticks))
        last_tick = tick

    mid.save(output_path)
    return output_path
