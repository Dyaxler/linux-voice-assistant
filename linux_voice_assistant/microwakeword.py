import ctypes
import json
import logging
import statistics
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Deque, List, Union

import numpy as np
from pymicro_features import MicroFrontend

from .wakeword import TfLiteWakeWord


LOGGER = logging.getLogger(__name__)

SAMPLES_PER_SECOND = 16000
SAMPLES_PER_CHUNK = 160  # 10ms
BYTES_PER_SAMPLE = 2  # 16-bit
BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * BYTES_PER_SAMPLE
SECONDS_PER_CHUNK = SAMPLES_PER_CHUNK / SAMPLES_PER_SECOND
STRIDE = 3


class MicroWakeWord(TfLiteWakeWord):
    def __init__(
        self,
        id: str,  # pylint: disable=redefined-builtin
        wake_word: str,
        tflite_model: Union[str, Path],
        probability_cutoff: float,
        sliding_window_size: int,
        trained_languages: List[str],
        libtensorflowlite_c_path: Union[str, Path],
    ) -> None:
        TfLiteWakeWord.__init__(self, libtensorflowlite_c_path)

        self.id = id
        self.wake_word = wake_word
        self.tflite_model = tflite_model
        self._probability_cutoff = float(probability_cutoff)
        self.sliding_window_size = sliding_window_size
        self._probability_enabled = True
        self._last_score_above_cutoff = False
        self.trained_languages = trained_languages

        self.is_active = True

        # Load the model and create interpreter
        self.model_path = str(Path(tflite_model).resolve()).encode("utf-8")
        self._load_model()

        self._features: List[np.ndarray] = []
        self._probabilities: Deque[float] = deque(maxlen=self.sliding_window_size)
        self._audio_buffer = bytes()

    def _load_model(self) -> None:
        self.model = self.lib.TfLiteModelCreateFromFile(self.model_path)
        self.interpreter = self.lib.TfLiteInterpreterCreate(self.model, None)
        self.lib.TfLiteInterpreterAllocateTensors(self.interpreter)

        # Access input and output tensor
        self.input_tensor = self.lib.TfLiteInterpreterGetInputTensor(
            self.interpreter, 0
        )
        self.output_tensor = self.lib.TfLiteInterpreterGetOutputTensor(
            self.interpreter, 0
        )

        # Get quantization parameters
        input_q = self.lib.TfLiteTensorQuantizationParams(self.input_tensor)
        output_q = self.lib.TfLiteTensorQuantizationParams(self.output_tensor)

        self.input_scale, self.input_zero_point = input_q.scale, input_q.zero_point
        self.output_scale, self.output_zero_point = output_q.scale, output_q.zero_point

    @staticmethod
    def from_config(
        config_path: Union[str, Path],
        libtensorflowlite_c_path: Union[str, Path],
    ) -> "MicroWakeWord":
        """Load a microWakeWord model from a JSON config file.

        Parameters
        ----------
        config_path: str or Path
            Path to JSON configuration file
        """
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        micro_config = config["micro"]

        return MicroWakeWord(
            id=Path(config["model"]).stem,
            wake_word=config["wake_word"],
            tflite_model=config_path.parent / config["model"],
            probability_cutoff=micro_config["probability_cutoff"],
            sliding_window_size=micro_config["sliding_window_size"],
            trained_languages=micro_config.get("trained_languages", []),
            libtensorflowlite_c_path=libtensorflowlite_c_path,
        )

    def process_streaming(self, features: np.ndarray) -> bool:
        self._features.append(features)

        if len(self._features) < STRIDE:
            # Not enough windows
            return False

        # Allocate and quantize input data
        with np.errstate(invalid="ignore"):
            quant_features = np.round(
                np.concatenate(self._features, axis=1) / self.input_scale
                + self.input_zero_point
            ).astype(np.uint8)

        # Stride instead of rolling
        self._features.clear()

        # Set tensor
        quant_ptr = quant_features.ctypes.data_as(ctypes.c_void_p)
        self.lib.TfLiteTensorCopyFromBuffer(
            self.input_tensor, quant_ptr, quant_features.nbytes
        )

        # Run inference
        self.lib.TfLiteInterpreterInvoke(self.interpreter)

        # Read output
        output_bytes = self.lib.TfLiteTensorByteSize(self.output_tensor)
        output_data = np.empty(output_bytes, dtype=np.uint8)
        self.lib.TfLiteTensorCopyToBuffer(
            self.output_tensor,
            output_data.ctypes.data_as(ctypes.c_void_p),
            output_bytes,
        )

        # Dequantize output
        result = (
            output_data.astype(np.float32) - self.output_zero_point
        ) * self.output_scale

        self._probabilities.append(result.item())

        if self._probability_enabled:
            if len(self._probabilities) < self.sliding_window_size:
                self._last_score_above_cutoff = False
                return False

            score = statistics.mean(self._probabilities)
            threshold = self._probability_cutoff
        else:
            if not self._probabilities:
                self._last_score_above_cutoff = False
                return False

            score = max(self._probabilities)
            threshold = 0.0

        above_cutoff = score > threshold
        triggered = above_cutoff and not self._last_score_above_cutoff
        self._last_score_above_cutoff = above_cutoff

        return triggered

    def set_probability_cutoff(self, probability_cutoff: float) -> None:
        """Update the probability cutoff used for activation."""

        self._probability_cutoff = max(0.0, min(1.0, float(probability_cutoff)))
        self._last_score_above_cutoff = False

    def use_probability(self, enabled: bool) -> None:
        """Enable or disable the probability threshold gate."""

        self._probability_enabled = bool(enabled)
        self._last_score_above_cutoff = False

    def set_sliding_window_size(self, sliding_window_size: int) -> None:
        """Update the size of the sliding probability window."""

        size = max(1, int(sliding_window_size))
        if size == self.sliding_window_size:
            return

        self.sliding_window_size = size
        self._probabilities = deque(self._probabilities, maxlen=self.sliding_window_size)
        self._last_score_above_cutoff = False

    def reset(self) -> None:
        """Clear any buffered features and probability history."""

        self._features.clear()
        self._probabilities.clear()
        self._last_score_above_cutoff = False

    def get_probability_cutoff(self) -> float:
        """Return the current probability cutoff."""

        return self._probability_cutoff


# -----------------------------------------------------------------------------


class MicroWakeWordFeatures(TfLiteWakeWord):
    def __init__(
        self,
        libtensorflowlite_c_path: Union[str, Path],
    ) -> None:
        TfLiteWakeWord.__init__(self, libtensorflowlite_c_path)

        self._audio_buffer = bytes()
        self._frontend = MicroFrontend()

    def process_streaming(self, audio_bytes: bytes) -> Iterable[np.ndarray]:
        self._audio_buffer += audio_bytes

        if len(self._audio_buffer) < BYTES_PER_CHUNK:
            # Not enough audio to get features
            return

        audio_buffer_idx = 0
        while (audio_buffer_idx + BYTES_PER_CHUNK) <= len(self._audio_buffer):
            # Process chunk
            chunk_bytes = self._audio_buffer[
                audio_buffer_idx : audio_buffer_idx + BYTES_PER_CHUNK
            ]
            frontend_result = self._frontend.ProcessSamples(chunk_bytes)
            audio_buffer_idx += frontend_result.samples_read * BYTES_PER_SAMPLE

            if not frontend_result.features:
                # Not enough audio for a full window
                continue

            yield np.array(frontend_result.features).reshape(
                (1, 1, len(frontend_result.features))
            )

        # Remove processed audio
        self._audio_buffer = self._audio_buffer[audio_buffer_idx:]
