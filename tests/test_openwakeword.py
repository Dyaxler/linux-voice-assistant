"""Tests for openWakeWord (CI-safe: mocks native TFLite layer)."""

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from linux_voice_assistant.util import is_arm

_TESTS_DIR = Path(__file__).parent
_REPO_DIR = _TESTS_DIR.parent
_OWW_DIR = _REPO_DIR / "wakewords" / "openWakeWord"

if is_arm():
    _LIB_DIR = _REPO_DIR / "lib" / "linux_arm64"
else:
    _LIB_DIR = _REPO_DIR / "lib" / "linux_amd64"

libtensorflowlite_c_path = _LIB_DIR / "libtensorflowlite_c.so"


def test_features() -> None:
    """Validate our streaming logic and thresholds without entering native code."""

    # --- Fake native lib behavior surface ---
    # We simulate the three interpreters (mel, emb, ww) by stubbing out
    # the TfLite* methods our wrapper calls and returning deterministic outputs.

    # A tiny ring buffer to hold the "last written" input bytes per interpreter
    class _FakeLib:
        def __init__(self):
            self._bytes = {}
            self._tensors = {"mel_in": object(), "mel_out": object(),
                             "emb_in": object(), "emb_out": object(),
                             "ww_in": object(),  "ww_out": object()}
            self._status_ok = 0

        # Model / interpreter
        def TfLiteModelCreateFromFile(self, _path): return object()
        def TfLiteInterpreterCreate(self, _model, _opts): return object()
        def TfLiteInterpreterAllocateTensors(self, _interp): return self._status_ok

        # Tensors
        def TfLiteInterpreterGetInputTensor(self, _interp, _idx):  # select by id
            return {"mel": self._tensors["mel_in"],
                    "emb": self._tensors["emb_in"],
                    "ww":  self._tensors["ww_in"]}.get(getattr(_interp, "_kind", "ww"), self._tensors["ww_in"])

        def TfLiteInterpreterGetOutputTensor(self, _interp, _idx):
            return {"mel": self._tensors["mel_out"],
                    "emb": self._tensors["emb_out"],
                    "ww":  self._tensors["ww_out"]}.get(getattr(_interp, "_kind", "ww"), self._tensors["ww_out"])

        # Shapes
        def TfLiteTensorNumDims(self, _tensor): return 3
        def TfLiteTensorDim(self, _tensor, i):
            # WW input shape [1, 8, 96]
            return (1, 8, 96)[i]

        # IO sizes
        def TfLiteTensorByteSize(self, tensor):
            # mel_out:  (1 * 1 * 1760 * 32) * 4 bytes = 225,280
            # emb_out:  (1 * 1 * 76   * 96) * 4 bytes = 29,184
            # ww_out:   (1) * 4 bytes
            if tensor is self._tensors["mel_out"]:
                return (1760 * 32) * 4
            if tensor is self._tensors["emb_out"]:
                return (76 * 96) * 4
            return 4

        # Copies
        def TfLiteTensorCopyFromBuffer(self, tensor, src_ptr, nbytes):
            self._bytes[id(tensor)] = (C_char_from(src_ptr, nbytes), nbytes)
            return self._status_ok

        def TfLiteInterpreterInvoke(self, _interp):
            return self._status_ok

        def TfLiteTensorCopyToBuffer(self, tensor, dst_ptr, nbytes):
            # Produce deterministic floats into dst based on tensor kind.
            dst = (C.c_char * nbytes).from_address(C.addressof(dst_ptr.contents))
            if tensor is self._tensors["mel_out"]:
                # mel: fill with small ramp
                arr = (np.linspace(0, 1, nbytes // 4, dtype=np.float32)).tobytes()
                dst[:len(arr)] = arr
            elif tensor is self._tensors["emb_out"]:
                # emb: also a ramp
                arr = (np.linspace(0, 0.5, nbytes // 4, dtype=np.float32)).tobytes()
                dst[:len(arr)] = arr
            else:
                # ww_out: probability ~0.8
                arr = (np.array([0.8], dtype=np.float32)).tobytes()
                dst[:len(arr)] = arr
            return self._status_ok

    # helpers for copy-from-buffer capture
    import ctypes as C
    def C_char_from(ptr, nbytes):
        return (C.c_char * nbytes).from_address(C.addressof(ptr.contents))[:]

    fake = _FakeLib()

    # Patch the base class to use our fake C lib
    with patch("linux_voice_assistant.wakeword.TfLiteWakeWord.__init__", lambda self, p: setattr(self, "lib", fake)):
        from linux_voice_assistant.openwakeword import OpenWakeWordFeatures, OpenWakeWord

        features = OpenWakeWordFeatures(
            melspectrogram_model=_OWW_DIR / "melspectrogram.tflite",
            embedding_model=_OWW_DIR / "embedding_model.tflite",
            libtensorflowlite_c_path=libtensorflowlite_c_path,
        )
        ww = OpenWakeWord(
            id="okay_nabu",
            wake_word="okay nabu",
            tflite_model=_OWW_DIR / "okay_nabu.tflite",
            libtensorflowlite_c_path=libtensorflowlite_c_path,
        )

        max_prob = 0.0
        with wave.open(str(_TESTS_DIR / "ok_nabu.wav"), "rb") as wav_file:
            assert wav_file.getframerate() == 16000
            assert wav_file.getsampwidth() == 2
            assert wav_file.getnchannels() == 1

            # Feed all audio at once (your code slides internally)
            for embeddings in features.process_streaming(
                wav_file.readframes(wav_file.getnframes())
            ):
                for prob in ww.process_streaming(embeddings):
                    max_prob = max(max_prob, prob)

        assert max_prob > 0.5
