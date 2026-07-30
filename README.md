# Polyscribe: Polyphonic Audio-to-MIDI CLI Converter

`polyscribe` is a lightweight, cross-platform CLI tool for converting continuous polyphonic audio signals (.wav, .mp3, .flac, .ogg, .m4a) into discrete Standard MIDI Files (.mid). Powered by neural pitch estimation (Spotify's Basic Pitch model) via ONNX Runtime / TFLite or a zero-weights Classical DSP engine, `polyscribe` runs efficiently on Linux, macOS, and Android (Termux) without requiring GPU acceleration or heavy deep learning framework dependencies.

---

## 🌟 Features

* **Polyphonic Pitch Estimation**: Transcribes complex multi-note chords and overlapping polyphonic instruments.
* **Continuous Pitch Bend Tracking**: Extracts sub-semitone contour predictions and emits timestamped MIDI `pitchwheel` events (`--pitch-bends`).
* **Dual Inference Engines**:
  * **`onnx` (Neural Model)**: High-accuracy deep learning transcription using Spotify's Basic Pitch model.
  * **`dsp` (Classical Non-ML Engine)**: Vectorized STFT harmonic peak extractor for ultra-fast offline transcription without model weights.
* **Universal Audio Decoder Chain**: Hierarchical decoding fallback (`soundfile` $\rightarrow$ `scipy.io.wavfile` $\rightarrow$ native Python `wave` $\rightarrow$ `ffmpeg` subprocess pipe for `.mp3`, `.m4a`, `.aac`, `.flac`, `.ogg`).
* **Boundary Note Deduplication**: Merges duplicate boundary notes spanning overlapping 30-second audio chunk windows.
* **Low Memory Footprint**: Uses 30-second sliding-window audio chunking to process long files while staying below 100 MB RAM on mobile devices.
* **Rich Terminal Interface**: Renders animated progress bars, note counts, pitch ranges (e.g. C4 to G6), and real-time execution speed ratios.
* **Cross-Platform**: Full support for Linux (x86_64/ARM64), macOS (Intel/Apple Silicon), and Android (Termux).

---

## 🚀 Installation

### 1. Android (Termux)
In Termux, install Python and ARM64-optimized ONNX Runtime wheels from the Termux User Repository (TUR):

```bash
# Update packages and install build tooling
pkg update && pkg upgrade -y
pkg install python python-numpy ffmpeg clang cmake -y

# Install Termux User Repository (TUR) & ONNX Runtime
pkg install tur-repo -y
pkg install python-onnxruntime -y

# Install Polyscribe dependencies
pip install mido rich

# Clone & Install Polyscribe locally
pip install -e .
```

### 2. Linux & macOS
```bash
pip install numpy mido rich onnxruntime
pip install -e .
```

---

## 🎧 Usage Examples

### Basic Conversion (Neural ONNX Engine)
```bash
polyscribe input_guitar.wav output_guitar.mid
```

### Fast Classical DSP Conversion with Pitch Bends
```bash
polyscribe input_solo.mp3 output_solo.mid --engine dsp --pitch-bends
```

### Advanced Neural Transcription
```bash
polyscribe input_piano.mp3 output_piano.mid \
    --engine onnx \
    --pitch-bends \
    --onset-thresh 0.55 \
    --frame-thresh 0.35 \
    --min-note-len 11 \
    --bpm 128 \
    --verbose
```

### Command Line Arguments

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | Position | Required | Path to input audio file (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`) |
| `output` | Position | Required | Path to output Standard MIDI file (`.mid`) |
| `--engine` | Choice | `onnx` | Inference engine: `onnx` (neural model) or `dsp` (classical non-ML STFT) |
| `--pitch-bends` | Flag | `False` | Extract contour predictions and emit MIDI `pitchwheel` events |
| `--model` | String | `None` | Custom path to ONNX model weights (`.onnx`). Auto-downloads `basic_pitch.onnx` if omitted. |
| `--onset-thresh` | Float | `0.5` | Sensitivity threshold for detecting new note starts (`0.0` - `1.0`) |
| `--frame-thresh` | Float | `0.3` | Threshold for sustaining active notes (`0.0` - `1.0`) |
| `--min-note-len` | Int | `11` | Minimum note duration in frames (~120ms at 86 fps) |
| `--bpm` | Float | `120.0` | MIDI project tempo in Beats Per Minute |
| `--chunk-size` | Float | `30.0` | Audio chunk length in seconds for memory safety |
| `--verbose` | Flag | `False` | Enable detailed processing logs |

---

## 🏗 Architecture & System Design

```
+-------------------------------------------------------------------+
|                        POLYSCRIBE CLI                             |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 1. AUDIO PREPROCESSOR (polyscribe/audio.py)                       |
|    - Hierarchical Decoders: soundfile -> scipy -> wave -> ffmpeg  |
|    - Downmixes audio stream to Mono                               |
|    - Resamples audio stream to 22,050 Hz                         |
|    - 30-second sliding-window chunking for memory safety          |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 2. INFERENCE ENGINE (polyscribe/engine.py & dsp_engine.py)        |
|    - ONNX Engine: Basic Pitch model via ONNX Runtime / TFLite     |
|    - DSP Engine: Vectorized STFT harmonic energy mapping           |
|    - Outputs Note Activations, Onset Activations, Contours        |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 3. ACTIVATION DECODER (polyscribe/decoder.py)                     |
|    - Peak thresholding & onset jump detection                     |
|    - Min note length duration filtering                           |
|    - Sub-semitone contour pitch bend extraction (-8192 to +8191)  |
|    - Overlapping chunk boundary note deduplication                |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 4. MIDI GENERATOR (polyscribe/midi_writer.py)                     |
|    - Formats tracks & events using mido                           |
|    - Interleaves Note On, Note Off, and Pitchwheel messages       |
|    - Exports Standard MIDI File (SMF Format 0) to disk            |
+-------------------------------------------------------------------+
```

---

## 📜 License
MIT License. Spotify Basic Pitch model weights licensed under Apache-2.0.
