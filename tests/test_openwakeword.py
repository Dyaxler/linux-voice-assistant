"""Tests for openWakeWord."""

import os
import wave
import ctypes
from pathlib import Path

import pytest

from linux_voice_assistant.openwakeword import OpenWakeWordFeatures, OpenWakeWord
from linux_voice_assistant.util import is_arm

_TESTS_DIR = Path(__file__).parent
_REPO_DIR = _TESTS_DIR.parent
_OWW_DIR = _REPO_DIR / "wakewords" / "openWakeWord"

if is_arm():
    _LIB_DIR = _REPO_DIR / "lib" / "linux_arm64"
else:
    _LIB_DIR = _REPO_DIR / "lib" / "linux_amd64"

libtensorflowlite_c_path = _LIB_DIR / "libtensorflowlite_c.so"


def _can_load_tflite(lib_path: Path) -> bool:
    """
    Try to load the bundled libtensorflowlite_c.so via ctypes, and
    add its directory to LD_LIBRARY_PATH just in case.
    """
    try:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path.parent}:{os.environ.get('LD_LIBRARY_PATH','')}"
        ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_LOCAL)
        return True
    except Exception:
        return False


skip_if_no_tflite = pytest.mark.skipif(
    os.environ.get("OPENWAKEWORD_DISABLE_TESTS") == "1" or not _can_load_tflite(libtensorflowlite_c_path),
    reason="TFLite C library not loadable on this runner (missing deps or wrong arch) or tests disabled."
)


@skip_if_no_tflite
def test_features() -> None:
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

        for embeddings in features.process_streaming(
            wav_file.readframes(wav_file.getnframes())
        ):
            for prob in ww.process_streaming(embeddings):
                max_prob = max(max_prob, prob)

    assert max_prob > 0.5
