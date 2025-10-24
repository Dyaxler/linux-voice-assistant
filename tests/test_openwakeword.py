"""Tests for openWakeWord (CI-safe: mocks native TFLite layer end-to-end)."""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import patch
import ctypes as C
import numpy as np
import struct

from linux_voice_assistant.util import is_arm

# Paths (unchanged)
_TESTS_DIR = Path(__file__).parent
_REPO_DIR = _TESTS_DIR.parent
_OWW_DIR = _REPO_DIR / "wakewords" / "openWakeWord"

if is_arm():
    _LIB_DIR = _REPO_DIR / "lib" / "linux_arm64"
else:
    _LIB_DIR = _REPO_DIR / "lib" / "linux_amd64"

libtensorflowlite_c_path = _LIB_DIR / "libtensorflowlite_c.so"

# Shapes/constants used by production code
MEL_SAMPLES = 1760
NUM_MELS = 32
EMB_FEATURES = 76           # typical # of time frames from embedding model per step
WW_FEATURES = 96            # embedding feature dimension
WW_INPUT_WINDOWS = 8        # wake-word model consumes 8 windows of 96-d embeddings
MEL_WINDOWS_PER_CHUNK = 76  # return 76 frames per step


def _addr_of_void_p(v: C.c_void_p | int) -> int:
    if isinstance(v, int):
        return v
    return int(getattr(v, "value", 0) or 0)


def _as_size_t(n) -> int:
    """Normalize int/ctypes/bytes to a Python int."""
    if isinstance(n, int):
        return n
    val = getattr(n, "value", None)
    if isinstance(val, int):
        return int(val)
    if isinstance(n, (bytes, bytearray)):
        if len(n) == 8:
            return struct.unpack("<Q", n)[0]
        if len(n) == 4:
            return struct.unpack("<I", n)[0]
    return int(n)


class _FakeModel:
    __slots__ = ("kind",)

    def __init__(self, kind: str):
        self.kind = kind  # "mel" | "emb" | "ww"


class _FakeInterpreter:
    __slots__ = ("kind",)

    def __init__(self, kind: str):
        self.kind = kind  # "mel" | "emb" | "ww"


class _FakeTensor:
    __slots__ = ("kind",)

    def __init__(self, kind: str):
        self.kind = kind  # "mel_in"|"mel_out"|"emb_in"|"emb_out"|"ww_in"|"ww_out"


class _FakeLib:
    """
    Minimal fake of the TFLite C API surface that openwakeword.py uses.
    Returns Python objects for model/interpreter/tensor "handles" and
    writes deterministic float data into provided buffers.
    """

    def __init__(self):
        # One set of tensors per "kind"
        self._mel_in = _FakeTensor("mel_in")
        self._mel_out = _FakeTensor("mel_out")
        self._emb_in = _FakeTensor("emb_in")
        self._emb_out = _FakeTensor("emb_out")
        self._ww_in = _FakeTensor("ww_in")
        self._ww_out = _FakeTensor("ww_out")
        self._status_ok = 0

    # --- Model & interpreter management ---
    def TfLiteModelCreateFromFile(self, path_bytes):
        try:
            p = path_bytes.decode() if isinstance(path_bytes, (bytes, bytearray)) else str(path_bytes)
        except Exception:
            p = str(path_bytes)
        if p.endswith("melspectrogram.tflite"):
            return _FakeModel("mel")
        if p.endswith("embedding_model.tflite"):
            return _FakeModel("emb")
        return _FakeModel("ww")

    def TfLiteInterpreterCreate(self, model, _opts):
        return _FakeInterpreter(model.kind)

    def TfLiteInterpreterAllocateTensors(self, _interp):
        return self._status_ok  # OK

    # --- Tensors ---
    def TfLiteInterpreterGetInputTensor(self, interp, _index):
        return {"mel": self._mel_in, "emb": self._emb_in, "ww": self._ww_in}[interp.kind]

    def TfLiteInterpreterGetOutputTensor(self, interp, _index):
        return {"mel": self._mel_out, "emb": self._emb_out, "ww": self._ww_out}[interp.kind]

    # --- Shapes/dims for WW input tensor ---
    def TfLiteTensorNumDims(self, _tensor):
        return 3

    def TfLiteTensorDim(self, _tensor, i):
        # WW input shape [1, WW_INPUT_WINDOWS, WW_FEATURES]
        dims = (1, WW_INPUT_WINDOWS, WW_FEATURES)
        return dims[i]

    # --- Resize (used by mel/emb) ---
    def TfLiteInterpreterResizeInputTensor(self, _interp, _tensor_index, _dims, _ndims):
        return self._status_ok  # noop/OK

    # --- Byte sizes for outputs ---
    def TfLiteTensorByteSize(self, tensor):
        if tensor.kind == "mel_out":
            # return 76 frames of 32 mels
            return (MEL_WINDOWS_PER_CHUNK * NUM_MELS) * 4
        if tensor.kind == "emb_out":
            # 76 time frames of 96-d embeddings
            return (EMB_FEATURES * WW_FEATURES) * 4
        # ww_out: single float probability
        return 4

    # --- Misc TFLite bits ---
    def TfLiteVersion(self):
        return b"Fake-TfLite-0.0"

    def TfLiteTensorType(self, _tensor):
        return 1  # kTfLiteFloat32

    def TfLiteTensorData(self, _tensor):
        return C.cast(C.create_string_buffer(4), C.c_void_p)

    # --- Copies ---
    def TfLiteTensorCopyFromBuffer(self, _tensor, _src_ptr, _nbytes):
        return self._status_ok

    def TfLiteInterpreterInvoke(self, _interp):
        return self._status_ok  # OK

    def TfLiteTensorCopyToBuffer(self, tensor, dst_void_p, nbytes):
        addr = _addr_of_void_p(dst_void_p)
        assert addr != 0, "Fake TFLite: destination pointer address was null/unknown"
        n = _as_size_t(nbytes)
        buf = (C.c_char * n).from_address(addr)

        if tensor.kind == "mel_out":
            # ramp over 76*32 floats
            arr = np.linspace(0.0, 1.0, n // 4, dtype=np.float32).tobytes()
            buf[: len(arr)] = arr
            return self._status_ok

        if tensor.kind == "emb_out":
            # ramp over 76*96 floats
            arr = np.linspace(0.0, 0.5, n // 4, dtype=np.float32).tobytes()
            buf[: len(arr)] = arr
            return self._status_ok

        # ww_out: fixed probability 0.8
        arr = np.array([0.8], dtype=np.float32).tobytes()
        buf[: len(arr)] = arr
        return self._status_ok


def test_features() -> None:
    from linux_voice_assistant import wakeword as _wakeword_mod

    # Patch base class so its __init__ uses our fake C API.
    with patch.object(_wakeword_mod.TfLiteWakeWord, "__init__", lambda self, p: setattr(self, "lib", _FakeLib())):
        # Import after patch to ensure the class uses our fake lib
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

            # Feed all frames; production code slides internally
            for embeddings in features.process_streaming(
                wav_file.readframes(wav_file.getnframes())
            ):
                for prob in ww.process_streaming(embeddings):
                    max_prob = max(max_prob, prob)

        assert max_prob > 0.5
