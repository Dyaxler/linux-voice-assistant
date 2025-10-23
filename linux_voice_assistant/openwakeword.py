"""openWakeWord implementation (hardened against loader/NULL-pointer issues)."""

from __future__ import annotations

import os
import ctypes as C
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Final, Union

import numpy as np

from .wakeword import TfLiteWakeWord

# ---------------------------------------------------------------------
# C aliases
c_void_p = C.c_void_p
c_int32 = C.c_int32
c_size_t = C.c_size_t

# ---------------------------------------------------------------------
# Constants (unchanged)
BATCH_SIZE: Final = 1

AUTOFILL_SECONDS: Final = 8
MAX_SECONDS: Final = 10

SAMPLE_RATE: Final = 16000  # 16Khz
_MAX_SAMPLES: Final = MAX_SECONDS * SAMPLE_RATE

SAMPLES_PER_CHUNK: Final = 1280  # 80 ms @ 16Khz
# NOTE: integer division would yield 0; keeping but not relied upon elsewhere
MS_PER_CHUNK: Final = int(1000 * SAMPLES_PER_CHUNK / SAMPLE_RATE)

# window = 400, hop length = 160
MELS_PER_SECOND: Final = 97
MAX_MELS: Final = MAX_SECONDS * MELS_PER_SECOND
MEL_SAMPLES: Final = 1760
NUM_MELS: Final = 32

EMB_FEATURES: Final = 76  # ~775 ms
EMB_STEP: Final = 8
MAX_EMB: Final = MAX_SECONDS * EMB_STEP
WW_FEATURES: Final = 96

MEL_SHAPE: Final = (BATCH_SIZE, MEL_SAMPLES)
EMB_SHAPE: Final = (BATCH_SIZE, EMB_FEATURES, NUM_MELS, 1)

# ---------------------------------------------------------------------
# Errors & helpers


class TFLiteLoadError(RuntimeError):
    """Raised when we can't load required TFLite shared libs or models."""


def _export_lib_dir_to_ld_path(lib_path: Union[str, Path]) -> None:
    """
    Ensure the directory containing libtensorflowlite_c.so is on LD_LIBRARY_PATH
    BEFORE any ctypes/CDLL/TfLite calls happen. Idempotent.
    """
    lib_dir = str(Path(lib_path).resolve().parent)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    if lib_dir not in parts:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{current}" if current else lib_dir


def _check_not_null(ptr, name: str) -> None:
    """
    Convert a NULL C pointer into a helpful Python exception with guidance.
    """
    if not ptr:
        raise TFLiteLoadError(
            f"{name} returned NULL. This usually means the TFLite C library or its "
            "dependencies failed to load, or the model file is incompatible/missing.\n\n"
            "Checklist:\n"
            "  • Confirm these system libraries are installed: "
            "libstdc++6, libgomp1, libopenblas0, libsndfile1\n"
            "  • Ensure LD_LIBRARY_PATH includes the directory of libtensorflowlite_c.so\n"
            "  • Verify the .tflite file exists and matches the expected architecture\n"
            "  • On CI/Codex, run `ldd libtensorflowlite_c.so` to surface missing deps"
        )


# ---------------------------------------------------------------------
# Main classes (hardened)

class OpenWakeWord(TfLiteWakeWord):
    def __init__(
        self,
        id: str,  # pylint: disable=redefined-builtin
        wake_word: str,
        tflite_model: Union[str, Path],
        libtensorflowlite_c_path: Union[str, Path],
        probability_cutoff: float = 0.5,
    ):
        # Make sure the dynamic loader can find the TFLite C lib BEFORE base init loads it
        _export_lib_dir_to_ld_path(libtensorflowlite_c_path)

        # Base class sets up self.lib (ctypes) with TfLite* symbols
        TfLiteWakeWord.__init__(self, libtensorflowlite_c_path)

        self.id = id
        self.wake_word = wake_word
        self.tflite_model = tflite_model
        self._probability_cutoff = max(0.0, min(1.0, float(probability_cutoff)))

        self.is_active = True

        # ---- Load the model and create interpreter (with NULL checks) ----
        self.model_path = str(Path(tflite_model).resolve()).encode("utf-8")

        self.model = self.lib.TfLiteModelCreateFromFile(self.model_path)
        _check_not_null(self.model, "TfLiteModelCreateFromFile")

        self.interpreter = self.lib.TfLiteInterpreterCreate(self.model, None)
        _check_not_null(self.interpreter, "TfLiteInterpreterCreate")

        # Allocate tensors
        ok = self.lib.TfLiteInterpreterAllocateTensors(self.interpreter)
        if isinstance(ok, int) and ok != 0:
            # Some bindings return status. If non-zero, raise a helpful error.
            raise TFLiteLoadError(
                f"TfLiteInterpreterAllocateTensors returned status {ok}. "
                "The model may be incompatible with this TFLite build."
            )

        # Resolve input/output tensors
        self.input_tensor = self.lib.TfLiteInterpreterGetInputTensor(
            self.interpreter, c_int32(0)
        )
        _check_not_null(self.input_tensor, "TfLiteInterpreterGetInputTensor")

        self.output_tensor = self.lib.TfLiteInterpreterGetOutputTensor(
            self.interpreter, c_int32(0)
        )
        _check_not_null(self.output_tensor, "TfLiteInterpreterGetOutputTensor")

        num_input_dims = self.lib.TfLiteTensorNumDims(self.input_tensor)
        input_shape = [
            self.lib.TfLiteTensorDim(self.input_tensor, i) for i in range(num_input_dims)
        ]
        # Defensive guard for unexpected shapes
        if not input_shape or len(input_shape) < 2:
            raise TFLiteLoadError(
                f"Unexpected input tensor shape {input_shape}. "
                "The model may not match expected [batch, window, features]."
            )
        self.input_windows = int(input_shape[1])

        self.new_embeddings: int = 0
        self.embeddings: np.ndarray = np.zeros(
            shape=(MAX_EMB, WW_FEATURES), dtype=np.float32
        )

    def process_streaming(self, embeddings: np.ndarray) -> Iterable[float]:
        """Generate probabilities from embeddings."""
        num_embedding_windows = embeddings.shape[2]

        # Shift
        self.embeddings[:-num_embedding_windows] = self.embeddings[num_embedding_windows:]

        # Overwrite
        self.embeddings[-num_embedding_windows:] = embeddings[0, 0, :, :]
        self.new_embeddings = min(
            len(self.embeddings),
            self.new_embeddings + num_embedding_windows,
        )

        while self.new_embeddings >= self.input_windows:
            emb_tensor = np.zeros(
                shape=(1, self.input_windows, WW_FEATURES),
                dtype=np.float32,
            )
            emb_tensor[0, :] = self.embeddings[
                -self.new_embeddings : len(self.embeddings) - self.new_embeddings + self.input_windows
            ]
            self.new_embeddings = max(0, self.new_embeddings - 1)

            # Run inference
            emb_ptr = emb_tensor.ctypes.data_as(c_void_p)
            self.lib.TfLiteTensorCopyFromBuffer(
                self.input_tensor, emb_ptr, c_size_t(emb_tensor.nbytes)
            )

            # Invoke
            status = self.lib.TfLiteInterpreterInvoke(self.interpreter)
            if isinstance(status, int) and status != 0:
                raise TFLiteLoadError(
                    f"TfLiteInterpreterInvoke returned status {status}. "
                    "Inference failed; model or runtime may be incompatible."
                )

            output_bytes = self.lib.TfLiteTensorByteSize(self.output_tensor)
            probs = np.empty(
                output_bytes // np.dtype(np.float32).itemsize, dtype=np.float32
            )
            self.lib.TfLiteTensorCopyToBuffer(
                self.output_tensor,
                probs.ctypes.data_as(c_void_p),
                c_size_t(output_bytes),
            )

            yield probs.item()

    def set_probability_cutoff(self, probability_cutoff: float) -> None:
        """Update the probability threshold used for detections."""
        self._probability_cutoff = max(0.0, min(1.0, float(probability_cutoff)))

    def get_probability_cutoff(self) -> float:
        """Return the current detection probability threshold."""
        return self._probability_cutoff

    def should_activate(self, probability: float) -> bool:
        """Return True if the provided probability triggers detection."""
        return probability >= self._probability_cutoff

    @staticmethod
    def from_config(
        config_path: Union[str, Path],
        libtensorflowlite_c_path: Union[str, Path],
    ) -> "OpenWakeWord":
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        probability_cutoff = float(config.get("probability_cutoff", 0.5))

        return OpenWakeWord(
            id=Path(config["model"]).stem,
            wake_word=config["wake_word"],
            tflite_model=config_path.parent / config["model"],
            libtensorflowlite_c_path=libtensorflowlite_c_path,
            probability_cutoff=probability_cutoff,
        )


# -----------------------------------------------------------------------------


class OpenWakeWordFeatures(TfLiteWakeWord):
    def __init__(
        self,
        melspectrogram_model: Union[str, Path],
        embedding_model: Union[str, Path],
        libtensorflowlite_c_path: Union[str, Path],
    ) -> None:
        # Ensure loader sees the directory first
        _export_lib_dir_to_ld_path(libtensorflowlite_c_path)

        TfLiteWakeWord.__init__(self, libtensorflowlite_c_path)

        # ----------------- Melspectrogram -----------------
        mel_path = str(Path(melspectrogram_model).resolve()).encode("utf-8")
        self.mel_model = self.lib.TfLiteModelCreateFromFile(mel_path)
        _check_not_null(self.mel_model, "TfLiteModelCreateFromFile(melspectrogram)")

        self.mel_interpreter = self.lib.TfLiteInterpreterCreate(self.mel_model, None)
        _check_not_null(self.mel_interpreter, "TfLiteInterpreterCreate(melspectrogram)")

        mels_dims = (c_int32 * len(MEL_SHAPE))(*MEL_SHAPE)
        self.lib.TfLiteInterpreterResizeInputTensor(
            self.mel_interpreter,
            c_int32(0),
            mels_dims,
            c_int32(len(MEL_SHAPE)),
        )
        status = self.lib.TfLiteInterpreterAllocateTensors(self.mel_interpreter)
        if isinstance(status, int) and status != 0:
            raise TFLiteLoadError(
                f"AllocateTensors (melspectrogram) returned status {status}"
            )

        self.mel_input_tensor = self.lib.TfLiteInterpreterGetInputTensor(
            self.mel_interpreter, c_int32(0)
        )
        _check_not_null(self.mel_input_tensor, "GetInputTensor(melspectrogram)")

        self.mel_output_tensor = self.lib.TfLiteInterpreterGetOutputTensor(
            self.mel_interpreter, c_int32(0)
        )
        _check_not_null(self.mel_output_tensor, "GetOutputTensor(melspectrogram)")

        # ----------------- Embedding -----------------
        emb_path = str(Path(embedding_model).resolve()).encode("utf-8")
        self.emb_model = self.lib.TfLiteModelCreateFromFile(emb_path)
        _check_not_null(self.emb_model, "TfLiteModelCreateFromFile(embedding)")

        self.emb_interpreter = self.lib.TfLiteInterpreterCreate(self.emb_model, None)
        _check_not_null(self.emb_interpreter, "TfLiteInterpreterCreate(embedding)")

        emb_dims = (c_int32 * len(EMB_SHAPE))(*EMB_SHAPE)
        self.lib.TfLiteInterpreterResizeInputTensor(
            self.emb_interpreter,
            c_int32(0),
            emb_dims,
            c_int32(len(EMB_SHAPE)),
        )
        status = self.lib.TfLiteInterpreterAllocateTensors(self.emb_interpreter)
        if isinstance(status, int) and status != 0:
            raise TFLiteLoadError(
                f"AllocateTensors (embedding) returned status {status}"
            )

        self.emb_input_tensor = self.lib.TfLiteInterpreterGetInputTensor(
            self.emb_interpreter, c_int32(0)
        )
        _check_not_null(self.emb_input_tensor, "GetInputTensor(embedding)")

        self.emb_output_tensor = self.lib.TfLiteInterpreterGetOutputTensor(
            self.emb_interpreter, c_int32(0)
        )
        _check_not_null(self.emb_output_tensor, "GetOutputTensor(embedding)")

        # ----------------- State -----------------
        self.new_audio_samples: int = AUTOFILL_SECONDS * SAMPLE_RATE
        self.audio: np.ndarray = np.zeros(shape=(_MAX_SAMPLES,), dtype=np.float32)
        self.new_mels: int = 0
        self.mels: np.ndarray = np.zeros(shape=(MAX_MELS, NUM_MELS), dtype=np.float32)

    def process_streaming(self, audio_chunk: bytes) -> Iterable[np.ndarray]:
        """Generate embeddings from audio."""
        # Convert little-endian 16-bit PCM -> float32
        chunk_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)

        # Shift samples left, add new to end
        self.audio[: -len(chunk_array)] = self.audio[len(chunk_array) :]
        self.audio[-len(chunk_array) :] = chunk_array
        self.new_audio_samples = min(
            len(self.audio),
            self.new_audio_samples + len(chunk_array),
        )

        while self.new_audio_samples >= MEL_SAMPLES:
            audio_tensor = np.zeros(shape=(BATCH_SIZE, MEL_SAMPLES), dtype=np.float32)
            audio_tensor[0, :] = self.audio[
                -self.new_audio_samples : len(self.audio) - self.new_audio_samples + MEL_SAMPLES
            ]
            audio_tensor = np.ascontiguousarray(audio_tensor)
            self.new_audio_samples = max(0, self.new_audio_samples - SAMPLES_PER_CHUNK)

            # ---- Inference: mels ----
            audio_ptr = audio_tensor.ctypes.data_as(c_void_p)
            self.lib.TfLiteTensorCopyFromBuffer(
                self.mel_input_tensor, audio_ptr, c_size_t(audio_tensor.nbytes)
            )
            status = self.lib.TfLiteInterpreterInvoke(self.mel_interpreter)
            if isinstance(status, int) and status != 0:
                raise TFLiteLoadError(
                    f"TfLiteInterpreterInvoke(mels) returned status {status}"
                )

            mels_output_bytes = self.lib.TfLiteTensorByteSize(self.mel_output_tensor)
            mels = np.empty(
                mels_output_bytes // np.dtype(np.float32).itemsize, dtype=np.float32
            )
            self.lib.TfLiteTensorCopyToBuffer(
                self.mel_output_tensor,
                mels.ctypes.data_as(c_void_p),
                c_size_t(mels_output_bytes),
            )

            # Transform to embedding domain and shape to [1, 1, windows, NUM_MELS]
            mels = (mels / 10) + 2
            mels = mels.reshape((1, 1, -1, NUM_MELS))

            # Slide window buffer
            num_mel_windows = mels.shape[2]
            self.mels[:-num_mel_windows] = self.mels[num_mel_windows:]
            self.mels[-num_mel_windows:] = mels[0, 0, :, :]
            self.new_mels = min(len(self.mels), self.new_mels + num_mel_windows)

            while self.new_mels >= EMB_FEATURES:
                mels_tensor = np.ascontiguousarray(
                    np.zeros(shape=EMB_SHAPE, dtype=np.float32)
                )
                mels_tensor[0, :, :, 0] = self.mels[
                    -self.new_mels : len(self.mels) - self.new_mels + EMB_FEATURES, :
                ]
                self.new_mels = max(0, self.new_mels - EMB_STEP)

                # ---- Inference: embedding ----
                mels_ptr = mels_tensor.ctypes.data_as(c_void_p)
                self.lib.TfLiteTensorCopyFromBuffer(
                    self.emb_input_tensor, mels_ptr, c_size_t(mels_tensor.nbytes)
                )
                status = self.lib.TfLiteInterpreterInvoke(self.emb_interpreter)
                if isinstance(status, int) and status != 0:
                    raise TFLiteLoadError(
                        f"TfLiteInterpreterInvoke(embedding) returned status {status}"
                    )

                emb_output_bytes = self.lib.TfLiteTensorByteSize(self.emb_output_tensor)
                emb = np.empty(
                    emb_output_bytes // np.dtype(np.float32).itemsize, dtype=np.float32
                )
                self.lib.TfLiteTensorCopyToBuffer(
                    self.emb_output_tensor,
                    emb.ctypes.data_as(c_void_p),
                    c_size_t(emb_output_bytes),
                )
                emb = emb.reshape((1, 1, -1, WW_FEATURES))
                yield emb
