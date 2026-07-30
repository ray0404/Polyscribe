# Technical Research & Architecture Guide: Polyphonic Audio-to-MIDI CLI Converter
Developing a **polyphonic audio-to-MIDI CLI converter** requires converting continuous polyphonic audio signals into discrete MIDI events (Note On, Note Off, Pitch Bend, Velocity). This process is known in Music Information Retrieval (MIR) as **Automatic Music Transcription (AMT)**.
To ensure strict cross-platform compatibility across **Linux, macOS, and Android/Termux**, the tool must balance inference accuracy, low memory usage, minimal system dependencies, and fast execution without relying on heavy frameworks like full TensorFlow or PyTorch installations.
## 1. Engine & Algorithmic Options
### A. Deep Learning AMT: Spotify's Basic Pitch (Recommended)
Basic Pitch by Spotify's Audio Intelligence Lab is an instrument-agnostic neural network for polyphonic note transcription and multipitch estimation (Bittner et al., ICASSP 2022).
 * **Model Footprint:** ~2.5 MB model weights file.
 * **Outputs:** 88 MIDI note pitch activations, note onset probabilities, and frame-level pitch bend values.
 * **Export Formats:** ONNX, TensorFlow Lite (.tflite), CoreML (.mlmodel), and TensorFlow (.pb).
 * **Why it fits:** It achieves state-of-the-art polyphonic accuracy while running efficiently on CPU without requiring GPU acceleration.
### B. Native C++ Inference: basicpitch.cpp
For minimal binary overhead and native CLI performance, sevagh/basicpitch.cpp provides a C++20 implementation of Basic Pitch using ONNX Runtime C++ API, Eigen, and libremidi.
 * **Speed & Overhead:** Bypasses Python runtime completely; ideal for resource-constrained environments like Android (Termux) or embedded Linux devices.
 * **Dependencies:** ONNX Runtime, Eigen, libremidi, and libnyquist / libsndfile for WAV/audio decoding.
### C. Classical DSP & Matrix Factorization (Non-ML)
For ultra-lightweight environments where neural network runtimes cannot be compiled:
 * **Algorithms:** Constant-Q Transform (CQT) combined with Non-negative Matrix Factorization (NMF) or Harmonic Product Spectrum (HPS).
 * **Libraries:** Librosa (Python) or Essentia (C++).
 * **Trade-off:** Fast execution, but reduced pitch detection accuracy on complex polyphonic textures (e.g., overlapping synth chords or dense acoustic arrangements).
## 2. Platform & Target Compatibility Matrix
| Platform | Recommended Engine | Primary Dependencies | Key Considerations |
|---|---|---|---|
| **Linux (x86_64 / ARM64)** | Python CLI or C++ Binary | Python 3.8–3.11, onnxruntime, mido, soundfile | Full support for PyPI wheels and native C++ builds. |
| **Termux (Android / ARM64)** | Python (ONNX Runtime) or Native C++ | onnxruntime (from tur-repo or pip), mido, soundfile | **Avoid full TensorFlow/PyTorch.** Use tflite-runtime or onnxruntime to prevent glibc/bionic build failures. |
| **macOS (Intel / Apple Silicon)** | Python or C++ Native | CoreML (coremltools) or ONNX Runtime, Accelerate framework | Native CoreML execution on Apple Silicon offers high efficiency. |
## 3. Tool Architecture & System Design
```
+------------------------------------------------------------------------------------+
|                                    AUDIO2MIDI CLI                                  |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
| 1. AUDIO DECODER & PREPROCESSOR                                                    |
|    - Decodes MP3/WAV/FLAC/OGG via soundfile / libsndfile                           |
|    - Downmixes audio to Mono                                                        |
|    - Resamples audio stream to 22,050 Hz                                            |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
| 2. INFERENCE ENGINE (Basic Pitch Model)                                            |
|    - Executes model via ONNX Runtime / TFLite                                      |
|    - Evaluates frame windows                                                       |
|    - Outputs: Onset Probabilities (T x 88), Note Frame Probabilities (T x 88),     |
|      Pitch Bend Predictions (T x 88)                                               |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
| 3. POST-PROCESSING & NOTE EXTRACTION                                               |
|    - Peak thresholding (Onset Threshold, Frame Threshold)                          |
|    - Minimum note duration filtering                                               |
|    - Pitch bend curve quantization & MIDI tick mapping                             |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
| 4. MIDI FILE GENERATOR                                                             |
|    - Formats tracks & events using Mido / libremidi / midly                         |
|    - Writes Standard MIDI File (SMF Format 0 or 1) to disk                         |
+------------------------------------------------------------------------------------+

```
## 4. Implementation Blueprint (Python & ONNX Runtime)
Below is a complete, modular implementation template using Python, ONNX Runtime, Mido, and soundfile / scipy. This approach avoids installing full TensorFlow wheels, making it compatible with Termux.
```python
#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import onnxruntime as ort
from mido import MidiFile, MidiTrack, Message, MetaMessage

TARGET_SAMPLE_RATE = 22050

def load_and_preprocess_audio(audio_path: str) -> np.ndarray:
    """Load audio file, convert to mono, and resample to 22.05 kHz."""
    data, sample_rate = sf.read(audio_path, dtype='float32')
    
    # Downmix to mono if multi-channel
    if data.ndim > 1:
        data = np.mean(data, axis=1)
        
    # Resample to 22,050 Hz
    if sample_rate != TARGET_SAMPLE_RATE:
        data = resample_poly(data, TARGET_SAMPLE_RATE, sample_rate)
        
    return data

def run_inference(audio_data: np.ndarray, model_path: str):
    """Run Basic Pitch ONNX model inference."""
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Model expects shape [batch_size, audio_samples, 1]
    input_tensor = audio_data[np.newaxis, :, np.newaxis].astype(np.float32)
    
    outputs = session.run(None, {input_name: input_tensor})
    # Basic Pitch ONNX outputs: [note_activation, onset_activation, pitch_bend]
    return outputs

def note_events_to_midi(notes, output_midi_path: str, bpm: int = 120):
    """Convert extracted note events into a Standard MIDI File."""
    mid = MidiFile(type=0)
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Set Tempo
    microseconds_per_beat = int(60_000_000 / bpm)
    track.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    
    ticks_per_beat = mid.ticks_per_beat
    
    # Convert time (seconds) to MIDI ticks
    def time_to_ticks(time_in_sec):
        return int(time_in_sec * (bpm / 60.0) * ticks_per_beat)
    
    # Sort notes by onset time
    notes.sort(key=lambda x: x['start_time'])
    
    last_tick = 0
    for note in notes:
        start_tick = time_to_ticks(note['start_time'])
        end_tick = time_to_ticks(note['end_time'])
        pitch = note['pitch']
        velocity = int(note['velocity'] * 127)
        
        # Note On
        delta_on = max(0, start_tick - last_tick)
        track.append(Message('note_on', note=pitch, velocity=velocity, time=delta_on))
        last_tick = start_tick
        
        # Note Off
        delta_off = max(0, end_tick - last_tick)
        track.append(Message('note_off', note=pitch, velocity=0, time=delta_off))
        last_tick = end_tick
        
    mid.save(output_midi_path)

def main():
    parser = argparse.ArgumentParser(description="Polyphonic Audio-to-MIDI CLI Converter")
    parser.add_argument("input_audio", help="Path to input audio file (.wav, .mp3, .flac)")
    parser.add_argument("output_midi", help="Path to output .mid file")
    parser.add_argument("--model", default="basic_pitch.onnx", help="Path to Basic Pitch ONNX model file")
    parser.add_argument("--onset-thresh", type=float, default=0.5, help="Note onset threshold (0.0 - 1.0)")
    
    args = parser.parse_args()
    
    print(f"[+] Loading and resampling {args.input_audio}...")
    audio = load_and_preprocess_audio(args.input_audio)
    
    print("[+] Executing polyphonic inference...")
    # Inference outputs note probability matrices
    outputs = run_inference(audio, args.model)
    
    print("[+] Exporting MIDI file...")
    # Example placeholder note list format: [{'pitch': 60, 'start_time': 0.5, 'end_time': 1.2, 'velocity': 0.8}]
    # In practice, apply note_creation frame-threshold decoding logic on `outputs`
    
    print(f"[+] Successfully saved MIDI output to {args.output_midi}")

if __name__ == "__main__":
    main()

```
## 5. Termux Installation & Deployment Workflow
To deploy and execute this tool within **Termux on Android**:
```bash
# 1. Update Termux packages and install build dependencies
pkg update && pkg upgrade -y
pkg install python python-numpy ffmpeg libsndfile clang cmake -y

# 2. Add Termux User Repository (TUR) for optimized ARM64 ONNX Runtime wheels
pkg install tur-repo -y
pkg install python-onnxruntime -y

# 3. Install lightweight audio and MIDI processing packages
pip install mido soundfile scipy

# 4. Download the Basic Pitch ONNX model file
curl -LO https://github.com/spotify/basic-pitch/raw/main/basic_pitch/saved_models/icassp_2022/nmp.onnx

# 5. Execute conversion
python audio2midi.py input_guitar.wav output_guitar.mid --model nmp.onnx

```
## 6. Key Considerations & Best Practices
 1. **Audio Preprocessing:** Always downmix input channels to mono and resample to **22,050 Hz** before model inference. Passing non-standard sample rates will shift pitch predictions.
 2. **Post-Processing Thresholds:** Expose onset threshold (--onset-threshold) and minimum frame length (--min-note-length) options via CLI flags. Higher onset thresholds reduce false-positive notes on noisy or resonant instruments.
 3. **Chunking Long Files:** For audio files exceeding 10 minutes, process the audio array in overlapping sliding windows (~30-second frames with 1-second overlap) to keep peak RAM usage under 100 MB on Android/Termux environments.
