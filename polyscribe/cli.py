"""
Command Line Interface (CLI) entry point for Polyscribe.
"""

import argparse
import sys
import os
import time
from typing import List, Dict

from polyscribe import __version__
from polyscribe.audio import load_audio, chunk_audio
from polyscribe.engine import PolyInferenceEngine
from polyscribe.dsp_engine import DSPEngine
from polyscribe.decoder import decode_output_to_notes
from polyscribe.midi_writer import export_notes_to_midi


def midi_pitch_to_name(pitch: int) -> str:
    """Convert MIDI note number to note name (e.g. 60 -> C4)."""
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (pitch // 12) - 1
    note_name = names[pitch % 12]
    return f"{note_name}{octave}"


def render_rich_summary(
    audio_path: str,
    midi_path: str,
    duration: float,
    notes: List[Dict],
    elapsed: float,
    engine_name: str
):
    """Render formatted summary table using rich if available."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        table = Table(title="[bold green]Polyscribe Audio-to-MIDI Transcription Summary[/bold green]")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Input Audio", audio_path)
        table.add_row("Output MIDI", midi_path)
        table.add_row("Engine", engine_name.upper())
        table.add_row("Audio Duration", f"{duration:.2f}s")
        table.add_row("Total Notes Extracted", str(len(notes)))

        if notes:
            min_pitch = min(n['pitch'] for n in notes)
            max_pitch = max(n['pitch'] for n in notes)
            pitch_range = f"{midi_pitch_to_name(min_pitch)} ({min_pitch}) - {midi_pitch_to_name(max_pitch)} ({max_pitch})"
            table.add_row("Pitch Range", pitch_range)

            pb_count = sum(len(n['pitch_bends']) for n in notes if n.get('pitch_bends'))
            table.add_row("Pitch Bend Events", str(pb_count))

        speed_ratio = duration / max(0.001, elapsed)
        table.add_row("Execution Time", f"{elapsed:.2f}s ({speed_ratio:.1f}x Realtime)")

        console.print()
        console.print(table)
        console.print(f"[bold green]✔ MIDI successfully written to:[/bold green] [yellow]{midi_path}[/yellow]")
    except ImportError:
        print("\n=== Polyscribe Summary ===")
        print(f"  Input Audio    : {audio_path}")
        print(f"  Output MIDI    : {midi_path}")
        print(f"  Engine         : {engine_name.upper()}")
        print(f"  Audio Duration : {duration:.2f}s")
        print(f"  Notes Extracted: {len(notes)}")
        if notes:
            min_pitch = min(n['pitch'] for n in notes)
            max_pitch = max(n['pitch'] for n in notes)
            print(f"  Pitch Range    : {midi_pitch_to_name(min_pitch)} to {midi_pitch_to_name(max_pitch)}")
        print(f"  Execution Time : {elapsed:.2f}s")
        print(f"[✔] Conversion complete. Saved to {midi_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="polyscribe",
        description="Polyscribe: Polyphonic Audio-to-MIDI CLI Converter (Termux / Linux / macOS)"
    )
    parser.add_argument("input", help="Path to input audio file (.wav, .mp3, .flac, .ogg, .m4a)")
    parser.add_argument("output", help="Path to output Standard MIDI File (.mid)")
    parser.add_argument("--engine", choices=["onnx", "dsp"], default="onnx", help="Inference engine: 'onnx' (neural model, default) or 'dsp' (classical non-ML)")
    parser.add_argument("--model", default=None, help="Path to custom ONNX model file (default: auto-download basic_pitch.onnx)")
    parser.add_argument("--pitch-bends", action="store_true", help="Extract and emit MIDI pitch bend events")
    parser.add_argument("--onset-thresh", type=float, default=0.5, help="Note onset detection threshold [0.0 - 1.0] (default: 0.5)")
    parser.add_argument("--frame-thresh", type=float, default=0.3, help="Sustained frame detection threshold [0.0 - 1.0] (default: 0.3)")
    parser.add_argument("--min-note-len", type=int, default=11, help="Minimum note length in frames (~120ms at 86fps, default: 11)")
    parser.add_argument("--bpm", type=float, default=120.0, help="Tempo for MIDI timing in BPM (default: 120.0)")
    parser.add_argument("--chunk-size", type=float, default=30.0, help="Chunk length in seconds for memory safety (default: 30.0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input audio file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    start_time = time.time()
    print(f"=== Polyscribe Polyphonic Audio-to-MIDI CLI v{__version__} ===")
    print(f"[+] Input Audio : {args.input}")
    print(f"[+] Output MIDI : {args.output}")

    # 1. Load Audio
    print("[+] Loading & resampling audio to 22.05 kHz...")
    try:
        audio, sr = load_audio(args.input)
    except Exception as e:
        print(f"Error loading audio: {e}", file=sys.stderr)
        sys.exit(1)

    duration = len(audio) / sr
    print(f"[+] Audio Duration: {duration:.2f}s ({len(audio)} samples)")

    # 2. Load Selected Engine
    print(f"[+] Initializing '{args.engine}' polyphonic inference engine...")
    try:
        if args.engine == "dsp":
            engine = DSPEngine()
        else:
            engine = PolyInferenceEngine(model_path=args.model)
    except Exception as e:
        print(f"Error initializing engine: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Polyphonic Pitch Estimation
    print("[+] Running polyphonic note extraction...")
    all_notes = []

    # Rich progress bar if available
    use_rich_progress = False
    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
        use_rich_progress = not args.verbose
    except ImportError:
        pass

    chunks = list(chunk_audio(audio, chunk_seconds=args.chunk_size, sr=sr))

    if use_rich_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Processing audio chunks...", total=len(chunks))
            for chunk, chunk_start, chunk_end in chunks:
                output = engine.run(chunk)
                contours_in = output.get('contour') if args.pitch_bends else None
                chunk_notes = decode_output_to_notes(
                    frames=output['note'],
                    onsets=output['onset'],
                    onset_thresh=args.onset_thresh,
                    frame_thresh=args.frame_thresh,
                    min_note_len=args.min_note_len,
                    contours=contours_in
                )
                for n in chunk_notes:
                    n['start_time'] += chunk_start
                    n['end_time'] += chunk_start
                    if n.get('pitch_bends'):
                        n['pitch_bends'] = [(t + chunk_start, val) for t, val in n['pitch_bends']]
                    all_notes.append(n)
                progress.advance(task)
    else:
        for i, (chunk, chunk_start, chunk_end) in enumerate(chunks, 1):
            if args.verbose:
                print(f"    Processing chunk {i}/{len(chunks)} [{chunk_start:.1f}s - {chunk_end:.1f}s]...")
            output = engine.run(chunk)
            contours_in = output.get('contour') if args.pitch_bends else None
            chunk_notes = decode_output_to_notes(
                frames=output['note'],
                onsets=output['onset'],
                onset_thresh=args.onset_thresh,
                frame_thresh=args.frame_thresh,
                min_note_len=args.min_note_len,
                contours=contours_in
            )
            for n in chunk_notes:
                n['start_time'] += chunk_start
                n['end_time'] += chunk_start
                if n.get('pitch_bends'):
                    n['pitch_bends'] = [(t + chunk_start, val) for t, val in n['pitch_bends']]
                all_notes.append(n)

    print(f"[+] Extracted {len(all_notes)} polyphonic note events.")

    # 4. Export MIDI
    print(f"[+] Writing MIDI file to {args.output}...")
    export_notes_to_midi(all_notes, args.output, bpm=args.bpm)

    elapsed = time.time() - start_time
    render_rich_summary(
        audio_path=args.input,
        midi_path=args.output,
        duration=duration,
        notes=all_notes,
        elapsed=elapsed,
        engine_name=args.engine
    )


if __name__ == "__main__":
    main()
