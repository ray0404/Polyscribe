"""
Inference engine wrapper supporting ONNX Runtime, TFLite, and auto-downloading model weights.
"""

import os
import urllib.request
import pathlib
import numpy as np
from typing import Dict, Optional, Union

BASIC_PITCH_ONNX_URL = "https://github.com/spotify/basic-pitch/raw/main/basic_pitch/saved_models/icassp_2022/nmp.onnx"
DEFAULT_MODEL_FILENAME = "basic_pitch.onnx"


def get_default_model_path() -> str:
    """Return cache directory path for default ONNX model weights."""
    cache_dir = os.path.expanduser("~/.cache/polyscribe")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, DEFAULT_MODEL_FILENAME)


def ensure_model_exists(model_path: Optional[str] = None) -> str:
    """Ensure ONNX model file exists; downloads default weights if missing."""
    if model_path and os.path.exists(model_path):
        return model_path

    target_path = model_path if model_path else get_default_model_path()
    if os.path.exists(target_path):
        return target_path

    print(f"[+] Downloading Basic Pitch ONNX model weights to {target_path}...")
    try:
        urllib.request.urlretrieve(BASIC_PITCH_ONNX_URL, target_path)
        print("[+] Download complete.")
        return target_path
    except Exception as e:
        raise RuntimeError(f"Failed to download Basic Pitch ONNX model: {e}")


class PolyInferenceEngine:
    """
    Polyphonic audio pitch estimation engine.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = ensure_model_exists(model_path)
        self.session = None
        self.input_name = None
        self.backend = None
        self._init_backend()

    def _init_backend(self):
        """Initialize ONNX Runtime or TFLite backend."""
        # Try ONNX Runtime
        try:
            import onnxruntime as ort

            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')

            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.backend = 'onnx'
            return
        except ImportError:
            pass

        # Try TFLite Runtime fallback
        try:
            import tflite_runtime.interpreter as tflite

            self.interpreter = tflite.Interpreter(self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.backend = 'tflite'
            return
        except ImportError:
            pass

        raise RuntimeError(
            "Neither 'onnxruntime' nor 'tflite_runtime' Python packages were found.\n"
            "Please install onnxruntime via: pip install onnxruntime (or pkg install python-onnxruntime in Termux)."
        )

    def run(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run polyphonic model inference on audio array.

        Args:
            audio: 1D float32 audio array sampled at 22,050 Hz

        Returns:
            Dictionary containing:
                - 'note': Note frame activations shape (T, 88)
                - 'onset': Onset frame activations shape (T, 88)
                - 'contour': Pitch bend contours shape (T, 264)
        """
        # Ensure 3D input tensor shape: [batch_size, audio_samples, 1]
        if audio.ndim == 1:
            input_tensor = audio[np.newaxis, :, np.newaxis].astype(np.float32)
        else:
            input_tensor = audio.astype(np.float32)

        if self.backend == 'onnx':
            outputs = self.session.run(None, {self.input_name: input_tensor})
            # Outputs mapping for Basic Pitch ONNX: [note_activation, onset_activation, contour]
            # Output array shapes match (batch, time, freq)
            note_act = np.squeeze(outputs[0], axis=0) if outputs[0].ndim == 3 else outputs[0]
            onset_act = np.squeeze(outputs[1], axis=0) if outputs[1].ndim == 3 else outputs[1]
            contour_act = np.squeeze(outputs[2], axis=0) if len(outputs) > 2 and outputs[2].ndim == 3 else (outputs[2] if len(outputs) > 2 else None)

            return {
                'note': note_act,
                'onset': onset_act,
                'contour': contour_act
            }
        elif self.backend == 'tflite':
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            out0 = self.interpreter.get_tensor(self.output_details[0]['index'])
            out1 = self.interpreter.get_tensor(self.output_details[1]['index'])
            out2 = self.interpreter.get_tensor(self.output_details[2]['index'])
            return {
                'note': np.squeeze(out0),
                'onset': np.squeeze(out1),
                'contour': np.squeeze(out2)
            }
