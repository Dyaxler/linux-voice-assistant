"""Shared models."""

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from .entity import (
        ESPHomeEntity,
        MediaPlayerEntity,
        MuteSwitchEntity,
        ConfigButtonEntity,
        ConfigSwitchEntity,
        WakeWordLibrarySelectEntity,
    )
    from .microwakeword import MicroWakeWord
    from .mpv_player import MpvMediaPlayer
    from .openwakeword import OpenWakeWord
    from .satellite import VoiceSatelliteProtocol

_LOGGER = logging.getLogger(__name__)

NO_WAKE_WORD_NAME = "No Wake Word"
NO_WAKE_WORD_SENTINEL = "__no_wake_word__"

WAKE_WORD_SENSITIVITY_LABELS: Tuple[str, ...] = (
    "Slightly sensitive",
    "Moderately sensitive",
    "Very sensitive",
)

WAKE_WORD_SENSITIVITY_TO_CUTOFF: Dict[str, float] = {
    "Slightly sensitive": 0.9,
    "Moderately sensitive": 0.7,
    "Very sensitive": 0.5,
}

DEFAULT_WAKE_WORD_SENSITIVITY = "Moderately sensitive"
DEFAULT_WAKE_WORD_PROBABILITY = WAKE_WORD_SENSITIVITY_TO_CUTOFF[
    DEFAULT_WAKE_WORD_SENSITIVITY
]


def get_wake_word_cutoff_for_label(label: str) -> float:
    return WAKE_WORD_SENSITIVITY_TO_CUTOFF.get(label, DEFAULT_WAKE_WORD_PROBABILITY)


def label_for_wake_word_cutoff(cutoff: float) -> str:
    closest_label = DEFAULT_WAKE_WORD_SENSITIVITY
    smallest_difference = float("inf")
    for label, value in WAKE_WORD_SENSITIVITY_TO_CUTOFF.items():
        difference = abs(value - cutoff)
        if difference < smallest_difference:
            smallest_difference = difference
            closest_label = label
    return closest_label


class WakeWordType(str, Enum):
    MICRO_WAKE_WORD = "micro"
    OPEN_WAKE_WORD = "openWakeWord"


def make_wake_word_unique_id(library: str, model_id: str) -> str:
    return f"{library}:{model_id}"


def make_no_wake_word_id(library: str) -> str:
    return make_wake_word_unique_id(library, NO_WAKE_WORD_SENTINEL)


def parse_wake_word_unique_id(unique_id: str, default_library: str) -> Tuple[str, str]:
    if ":" in unique_id:
        library, model_id = unique_id.split(":", 1)
        return library, model_id

    return default_library, unique_id


@dataclass
class AvailableWakeWord:
    id: str
    library: str
    model_id: str
    type: WakeWordType
    wake_word: str
    trained_languages: List[str]
    config_path: Path

    def load(
        self, libtensorflowlite_c_path: Path
    ) -> "Union[MicroWakeWord, OpenWakeWord]":
        if self.type == WakeWordType.MICRO_WAKE_WORD:
            from .microwakeword import MicroWakeWord

            wake_word = MicroWakeWord.from_config(
                config_path=self.config_path,
                libtensorflowlite_c_path=libtensorflowlite_c_path,
            )
            wake_word.id = self.id
            return wake_word

        if self.type == WakeWordType.OPEN_WAKE_WORD:
            from .openwakeword import OpenWakeWord

            wake_word = OpenWakeWord.from_config(
                config_path=self.config_path,
                libtensorflowlite_c_path=libtensorflowlite_c_path,
            )
            wake_word.id = self.id
            return wake_word

        raise ValueError(f"Unexpected wake word type: {self.type}")


@dataclass
class WakeWordSelection:
    library: str = ""
    model: str = NO_WAKE_WORD_NAME

    @classmethod
    def from_dict(
        cls, data: Optional[Dict[str, Any]], default_library: str, default_model: str
    ) -> "WakeWordSelection":
        if not isinstance(data, dict):
            return cls(library=default_library, model=default_model)

        library_value = data.get("library", default_library)
        library = (
            library_value
            if isinstance(library_value, str) and library_value
            else default_library
        )

        model_value = data.get("model", default_model)
        model = (
            model_value if isinstance(model_value, str) and model_value else default_model
        )
        return cls(library=library, model=model)


@dataclass
class AssistantPreferences:
    wake_word: WakeWordSelection = field(default_factory=WakeWordSelection)

    @classmethod
    def from_dict(
        cls,
        data: Optional[Dict[str, Any]],
        default_library: str,
        default_model: str,
    ) -> "AssistantPreferences":
        wake_word_data: Optional[Dict[str, Any]] = None

        if isinstance(data, dict):
            wake_word_data = data.get("wake_word")

        wake_word = WakeWordSelection.from_dict(
            wake_word_data, default_library=default_library, default_model=default_model
        )
        return cls(wake_word=wake_word)


@dataclass
class Preferences:
    volume: Optional[float] = None
    assistant: AssistantPreferences = field(default_factory=AssistantPreferences)
    assistant_2: AssistantPreferences = field(default_factory=AssistantPreferences)
    microphone_mute: bool = False
    wake_word_sensitivity: str = DEFAULT_WAKE_WORD_SENSITIVITY
    wake_word_probability: float = DEFAULT_WAKE_WORD_PROBABILITY
    wake_sound_enabled: bool = True
    timer_sound_enabled: bool = True

    @classmethod
    def load(
        cls,
        data: Optional[Dict[str, Any]],
        default_library: str,
        default_model: str,
        default_second_model: str,
    ) -> "Preferences":
        preferences = cls()

        if isinstance(data, dict):
            volume = data.get("volume")
            if volume is not None:
                try:
                    preferences.volume = float(volume)
                except (TypeError, ValueError):
                    preferences.volume = None

            preferences.microphone_mute = bool(data.get("microphone_mute", False))
            preferences.wake_sound_enabled = bool(
                data.get("wake_sound_enabled", True)
            )
            preferences.timer_sound_enabled = bool(
                data.get("timer_sound_enabled", True)
            )
        else:
            preferences.microphone_mute = False
            preferences.wake_sound_enabled = True
            preferences.timer_sound_enabled = True

        sensitivity_label = DEFAULT_WAKE_WORD_SENSITIVITY
        probability_value: Optional[float] = None

        if isinstance(data, dict):
            stored_label = data.get("wake_word_sensitivity")
            if (
                isinstance(stored_label, str)
                and stored_label in WAKE_WORD_SENSITIVITY_TO_CUTOFF
            ):
                sensitivity_label = stored_label
            else:
                stored_probability = data.get("wake_word_probability")
                if stored_probability is not None:
                    try:
                        probability_value = float(stored_probability)
                    except (TypeError, ValueError):
                        probability_value = None

        if probability_value is not None:
            sensitivity_label = label_for_wake_word_cutoff(probability_value)

        preferences.wake_word_sensitivity = sensitivity_label
        preferences.wake_word_probability = get_wake_word_cutoff_for_label(
            sensitivity_label
        )

        preferences.assistant = AssistantPreferences.from_dict(
            data.get("assistant") if isinstance(data, dict) else None,
            default_library=default_library,
            default_model=default_model,
        )
        preferences.assistant_2 = AssistantPreferences.from_dict(
            data.get("assistant_2") if isinstance(data, dict) else None,
            default_library=default_library,
            default_model=default_second_model,
        )

        legacy_active = []
        if isinstance(data, dict):
            legacy_active = data.get("active_wake_words", [])

        if isinstance(legacy_active, list) and legacy_active:
            first = str(legacy_active[0])
            preferences.assistant.wake_word = WakeWordSelection(
                library=default_library, model=first
            )
            if len(legacy_active) > 1:
                second = str(legacy_active[1])
            else:
                second = NO_WAKE_WORD_NAME
            preferences.assistant_2.wake_word = WakeWordSelection(
                library=default_library, model=second
            )

        return preferences


@dataclass
class ServerState:
    name: str
    mac_address: str
    audio_queue: "Queue[Optional[bytes]]"
    entities: "List[ESPHomeEntity]"
    wakewords_dir: Path
    available_wake_word_libraries: Dict[str, Dict[str, AvailableWakeWord]]
    active_wake_word_library: str
    wake_words: "Dict[str, Union[MicroWakeWord, OpenWakeWord]]"
    stop_word: "MicroWakeWord"
    music_player: "MpvMediaPlayer"
    tts_player: "MpvMediaPlayer"
    wakeup_sound: str
    timer_finished_sound: str
    preferences: Preferences
    preferences_path: Path
    libtensorflowlite_c_path: Path

    # openWakeWord
    oww_melspectrogram_path: Path
    oww_embedding_path: Path

    media_player_entity: "Optional[MediaPlayerEntity]" = None
    satellite: "Optional[VoiceSatelliteProtocol]" = None
    mute_switch_entity: "Optional[MuteSwitchEntity]" = None
    wake_word_library_entity: "Optional[WakeWordLibrarySelectEntity]" = None
    wake_words_changed: bool = False
    muted: bool = False
    connected: bool = False
    volume: float = 1.0
    preferred_default_model_ids: Tuple[str, ...] = ("okay_nabu",)
    wake_word_sensitivity: str = DEFAULT_WAKE_WORD_SENSITIVITY
    wake_word_probability_cutoff: float = DEFAULT_WAKE_WORD_PROBABILITY
    wake_sound_enabled: bool = True
    timer_sound_enabled: bool = True
    wake_word_sensitivity_entity: "Optional[WakeWordLibrarySelectEntity]" = None
    wake_sound_switch_entity: "Optional[ConfigSwitchEntity]" = None
    timer_sound_switch_entity: "Optional[ConfigSwitchEntity]" = None
    reset_assistant_button_entity: "Optional[ConfigButtonEntity]" = None
    restart_device_button_entity: "Optional[ConfigButtonEntity]" = None
    stop_word_event_active: bool = False
    stop_word_last_detection: float = 0.0
    stop_word_cooldown_until: float = 0.0
    stop_word_reset_pending_log: bool = False

    def save_preferences(self) -> None:
        """Save preferences as JSON."""
        _LOGGER.debug("Saving preferences: %s", self.preferences_path)
        self.preferences_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.preferences_path, "w", encoding="utf-8") as preferences_file:
            json.dump(
                asdict(self.preferences), preferences_file, ensure_ascii=False, indent=4
            )

    def persist_volume(self, volume: float) -> None:
        """Persist the normalized media volume (0.0 - 1.0)."""
        clamped_volume = max(0.0, min(1.0, volume))

        if (
            abs(self.volume - clamped_volume) < 0.0001
            and self.preferences.volume is not None
            and abs(self.preferences.volume - clamped_volume) < 0.0001
        ):
            return

        self.volume = clamped_volume
        self.preferences.volume = clamped_volume
        self.save_preferences()

    def get_active_library_wake_words(self) -> Dict[str, AvailableWakeWord]:
        return self.available_wake_word_libraries.get(
            self.active_wake_word_library, {}
        )

    def get_default_model_id(self) -> Optional[str]:
        library_models = self.get_active_library_wake_words()
        if not library_models:
            return None

        for model_id in self.preferred_default_model_ids:
            if model_id in library_models:
                return model_id

        return next(iter(sorted(library_models.keys())))

    def ensure_preferences_valid(self) -> None:
        library_models = self.get_active_library_wake_words()
        default_model = self.get_default_model_id()

        def _ensure(selection: WakeWordSelection, fallback: Optional[str]) -> None:
            selection.library = self.active_wake_word_library
            if selection.model == NO_WAKE_WORD_NAME:
                return
            if selection.model in library_models:
                return
            if fallback and fallback in library_models:
                selection.model = fallback
            elif default_model and default_model in library_models:
                selection.model = default_model
            else:
                selection.model = NO_WAKE_WORD_NAME

        _ensure(self.preferences.assistant.wake_word, default_model)
        _ensure(self.preferences.assistant_2.wake_word, None)

    def sync_wake_word_models(self) -> None:
        self.ensure_preferences_valid()
        library_models = self.get_active_library_wake_words()
        active_ids: List[str] = []

        selections: Iterable[WakeWordSelection] = (
            self.preferences.assistant.wake_word,
            self.preferences.assistant_2.wake_word,
        )

        previous_ids = set(self.wake_words.keys())
        previous_active = {
            unique_id for unique_id, wake_word in self.wake_words.items() if wake_word.is_active
        }

        for selection in selections:
            selection.library = self.active_wake_word_library
            if selection.model == NO_WAKE_WORD_NAME:
                continue

            model_info = library_models.get(selection.model)
            if not model_info:
                continue

            if model_info.id not in self.wake_words:
                self.wake_words[model_info.id] = model_info.load(
                    self.libtensorflowlite_c_path
                )

            active_ids.append(model_info.id)

        allowed_ids = {model_info.id for model_info in library_models.values()}
        library_prefix = f"{self.active_wake_word_library}:"
        for unique_id in list(self.wake_words.keys()):
            if not unique_id.startswith(library_prefix):
                del self.wake_words[unique_id]
            elif not allowed_ids:
                del self.wake_words[unique_id]
            elif unique_id not in allowed_ids:
                del self.wake_words[unique_id]

        for unique_id, wake_word in self.wake_words.items():
            wake_word.is_active = unique_id in active_ids

        new_ids = set(self.wake_words.keys())
        new_active = {
            unique_id for unique_id, wake_word in self.wake_words.items() if wake_word.is_active
        }

        self.wake_words_changed = (
            previous_ids != new_ids or previous_active != new_active
        )

        self.apply_wake_word_sensitivity()

    def get_active_wake_word_ids(self) -> List[str]:
        library_models = self.get_active_library_wake_words()
        no_id = make_no_wake_word_id(self.active_wake_word_library)

        ids: List[str] = []
        for selection in (
            self.preferences.assistant.wake_word,
            self.preferences.assistant_2.wake_word,
        ):
            if selection.model == NO_WAKE_WORD_NAME:
                ids.append(no_id)
                continue

            model_info = library_models.get(selection.model)
            if model_info:
                ids.append(model_info.id)
            else:
                ids.append(no_id)

        return ids

    def set_active_wake_word_library(self, library: str) -> bool:
        if library not in self.available_wake_word_libraries:
            _LOGGER.warning("Unknown wake word library: %s", library)
            return False

        if library == self.active_wake_word_library:
            return False

        self.active_wake_word_library = library
        self.preferences.assistant.wake_word.library = library
        self.preferences.assistant_2.wake_word.library = library

        self.ensure_preferences_valid()
        self.sync_wake_word_models()
        return True

    def update_wake_word_libraries(
        self, libraries: Dict[str, Dict[str, AvailableWakeWord]]
    ) -> None:
        self.available_wake_word_libraries = libraries

        if self.active_wake_word_library not in libraries:
            self.active_wake_word_library = next(iter(sorted(libraries.keys())), "")

        if not self.active_wake_word_library and libraries:
            self.active_wake_word_library = next(iter(sorted(libraries.keys())))

        self.ensure_preferences_valid()
        self.sync_wake_word_models()

    def apply_wake_word_sensitivity(self) -> None:
        cutoff = get_wake_word_cutoff_for_label(self.wake_word_sensitivity)
        self.wake_word_probability_cutoff = cutoff
        self.preferences.wake_word_probability = cutoff
        for wake_word in self.wake_words.values():
            if hasattr(wake_word, "set_probability_cutoff"):
                wake_word.set_probability_cutoff(cutoff)

    def set_wake_word_sensitivity(self, label: str) -> bool:
        if label not in WAKE_WORD_SENSITIVITY_TO_CUTOFF:
            _LOGGER.warning("Unknown wake word sensitivity: %s", label)
            return False

        if self.wake_word_sensitivity == label:
            return False

        self.wake_word_sensitivity = label
        self.preferences.wake_word_sensitivity = label
        self.apply_wake_word_sensitivity()
        self.save_preferences()
        return True

    def set_wake_sound_enabled(self, enabled: bool) -> bool:
        enabled = bool(enabled)
        if (
            self.wake_sound_enabled == enabled
            and self.preferences.wake_sound_enabled == enabled
        ):
            return False

        self.wake_sound_enabled = enabled
        self.preferences.wake_sound_enabled = enabled
        self.save_preferences()
        return True

    def set_timer_sound_enabled(self, enabled: bool) -> bool:
        enabled = bool(enabled)
        if (
            self.timer_sound_enabled == enabled
            and self.preferences.timer_sound_enabled == enabled
        ):
            return False

        self.timer_sound_enabled = enabled
        self.preferences.timer_sound_enabled = enabled
        self.save_preferences()
        return True
