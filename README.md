# Polyscribe: Polyphonic Audio-to-MIDI CLI Converter

`polyscribe` is a lightweight, cross-platform CLI tool for converting continuous polyphonic audio signals (.wav, .mp3, .flac, .ogg) into discrete Standard MIDI Files (.mid). Powered by neural pitch estimation (Spotify's Basic Pitch model) via ONNX Runtime / TFLite, `polyscribe` runs efficiently on Linux, macOS, and Android (Termux) without requiring GPU acceleration or heavy deep learning framework dependencies.

---

## 🌟 Features

* **Polyphonic Pitch Estimation**: Transcribes complex multi-note chords and overlapping polyphonic instruments.
* **Low Memory Footprint**: Uses 30-second sliding-window audio chunking to process long files while staying below 100 MB RAM on mobile devices.
* **Cross-Platform**: Supports Linux (x86_64/ARM64), macOS (Intel/Apple Silicon), and Android (Termux).
* **Automatic Weights Management**: Downloads and caches the ~2.5 MB `basic_pitch.onnx` neural model on first run.
* **Tunable Parameters**: Exposes CLI flags for onset threshold, sustained frame threshold, minimum note length, and MIDI tempo.

---

## 🚀 Installation

### 1. Android (Termux)
In Termux, install Python and ARM64-optimized ONNX Runtime wheels from the Termux User Repository (TUR):

```bash
# Update packages and install build tooling
pkg update && pkg upgrade -y
pkg install python python-numpy ffmpeg libsndfile clang cmake -y

# Install Termux User Repository (TUR) & ONNX Runtime
pkg install tur-repo -y
pkg install python-onnxruntime -y

# Install Polyscribe dependencies
pip install soundfile scipy mido

# Clone / Install Polyscribe locally
pip install -e .
```

### 2. Linux & macOS
```bash
pip install soundfile scipy mido onnxruntime
pip install -e .
```

---

## 🎧 Usage Examples

### Basic Conversion
```bash
polyscribe input_guitar.wav output_guitar.mid
```

### Advanced Options
```bash
polyscribe input_piano.mp3 output_piano.mid \
    --onset-thresh 0.6 \
    --frame-thresh 0.35 \
    --min-note-len 11 \
    --bpm 128 \
    --verbose
```

### Command Line Arguments

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | Position | Required | Path to input audio file (`.wav`, `.mp3`, `.flac`, `.ogg`) |
| `output` | Position | Required | Path to output Standard MIDI file (`.mid`) |
| `--model` | String | `None` | Custom path to ONNX model weights (`.onnx`). Auto-downloads `basic_pitch.onnx` if omitted. |
| `--onset-thresh` | Float | `0.5` | Sensitivity threshold for detecting new note starts (`0.0` - `1.0`) |
| `--frame-thresh` | Float | `0.3` | Threshold for sustaining active notes (`0.0` - `1.0`) |
| `--min-note-len` | Int | `11` | Minimum note duration in frames (~120ms at 86 fps) |
| `--bpm` | Float | `120.0` | MIDI project tempo in Beats Per Minute |
| `--chunk-size` | Float | `30.0` | Audio chunk length in seconds for memory safety |
| `--verbose` | Flag | `False` | Enable detailed processing logs |

---

## 🏗 Architecture & Design

```
+-------------------------------------------------------------------+
|                        POLYSCRIBE CLI                             |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 1. AUDIO PREPROCESSOR (polyscribe/audio.py)                       |
|    - Decodes MP3/WAV/FLAC via soundfile                          |
|    - Downmixes audio stream to Mono                               |
|    - Resamples audio stream to 22,050 Hz                         |
|    - Chunking into sliding windows for memory safety              |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 2. INFERENCE ENGINE (polyscribe/engine.py)                        |
|    - Executes Basic Pitch model via ONNX Runtime / TFLite         |
|    - Outputs Note Activations, Onset Activations, Contours        |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 3. ACTIVATION DECODER (polyscribe/decoder.py)                     |
|    - Peak thresholding & onset jump detection                     |
|    - Min note length duration filtering                           |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 4. MIDI GENERATOR (polyscribe/midi_writer.py)                      |
|    - Formats tracks & events using mido                           |
|    - Exports Standard MIDI File (SMF Format 0) to disk            |
+-------------------------------------------------------------------+
```

---

## 📜 License
MIT License. Spotify Basic Pitch model weights licensed under Apache-2.0.
