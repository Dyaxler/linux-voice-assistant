"""Shared models."""

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

from time import monotonic

if TYPE_CHECKING:
    from pymicro_wakeword import MicroWakeWord
    from pyopen_wakeword import OpenWakeWord

    from .entity import ESPHomeEntity, MediaPlayerEntity, MuteSwitchEntity
    from .mpv_player import MpvMediaPlayer
    from .satellite import VoiceSatelliteProtocol

_LOGGER = logging.getLogger(__name__)


class StopWordFilter:
    """Soft confidence filter to limit spurious stop-word activations."""

    def __init__(
        self,
        *,
        threshold: float = 1.0,
        increment: float = 0.65,
        decay_seconds: float = 0.45,
        cooldown_seconds: float = 0.75,
    ) -> None:
        self.threshold = threshold
        self.increment = increment
        self.decay_seconds = decay_seconds
        self.cooldown_seconds = cooldown_seconds
        self._score = 0.0
        self._last_update: Optional[float] = None
        self._active_samples = 0
        self._cooldown_until: Optional[float] = None
        self._cooldown_logged = False

    @property
    def score(self) -> float:
        return self._score

    def reset(self, *, keep_cooldown: bool = False) -> None:
        self._score = 0.0
        self._last_update = None
        self._active_samples = 0
        if not keep_cooldown:
            self._cooldown_until = None
        self._cooldown_logged = False

    def observe(
        self,
        activated: bool,
        *,
        now: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ) -> bool:
        """Update the confidence score and return True when the stop word should fire."""

        timestamp = now if now is not None else monotonic()

        if self._cooldown_until and timestamp < self._cooldown_until:
            if activated and logger and not self._cooldown_logged:
                remaining = self._cooldown_until - timestamp
                logger.debug(
                    "Stop filter suppressing activation during %.2fs cooldown", remaining
                )
                self._cooldown_logged = True

            self._last_update = timestamp
            return False

        self._cooldown_logged = False

        if self._last_update is not None:
            elapsed = max(0.0, timestamp - self._last_update)
            if elapsed > 0.0:
                decay = math.exp(-elapsed / self.decay_seconds)
                self._score *= decay
        else:
            # First observation after reset starts from a neutral score.
            self._score = max(0.0, self._score)

        if activated:
            self._active_samples += 1
            self._score += self.increment
        else:
            self._active_samples = 0

        self._last_update = timestamp

        if self._score >= self.threshold:
            samples = max(1, self._active_samples)
            if logger:
                logger.debug(
                    "Stop filter threshold reached after %d samples (confidence=%.2f >= %.2f)",
                    samples,
                    self._score,
                    self.threshold,
                )

            if self.cooldown_seconds > 0.0:
                self._cooldown_until = timestamp + self.cooldown_seconds
            else:
                self._cooldown_until = None

            self._score = 0.0
            self._last_update = None
            self._active_samples = 0
            self._cooldown_logged = False
            return True

        return False


class WakeWordType(str, Enum):
    MICRO_WAKE_WORD = "micro"
    OPEN_WAKE_WORD = "openWakeWord"


@dataclass
class AvailableWakeWord:
    id: str
    type: WakeWordType
    wake_word: str
    trained_languages: List[str]
    wake_word_path: Path

    def load(self) -> "Union[MicroWakeWord, OpenWakeWord]":
        if self.type == WakeWordType.MICRO_WAKE_WORD:
            from pymicro_wakeword import MicroWakeWord

            return MicroWakeWord.from_config(config_path=self.wake_word_path)

        if self.type == WakeWordType.OPEN_WAKE_WORD:
            from pyopen_wakeword import OpenWakeWord

            oww_model = OpenWakeWord.from_model(model_path=self.wake_word_path)
            setattr(oww_model, "wake_word", self.wake_word)

            return oww_model

        raise ValueError(f"Unexpected wake word type: {self.type}")


@dataclass
class Preferences:
    active_wake_words: List[str] = field(default_factory=list)
    media_volume: float = 1.0


@dataclass
class ServerState:
    name: str
    mac_address: str
    audio_queue: "Queue[Optional[bytes]]"
    entities: "List[ESPHomeEntity]"
    available_wake_words: "Dict[str, AvailableWakeWord]"
    wake_words: "Dict[str, Union[MicroWakeWord, OpenWakeWord]]"
    active_wake_words: Set[str]
    stop_word: "MicroWakeWord"
    stop_filter: "StopWordFilter"
    music_player: "MpvMediaPlayer"
    tts_player: "MpvMediaPlayer"
    wakeup_sound: str
    timer_finished_sound: str
    preferences: Preferences
    preferences_path: Path
    media_volume: float

    media_player_entity: "Optional[MediaPlayerEntity]" = None
    satellite: "Optional[VoiceSatelliteProtocol]" = None
    mute_switch_entity: "Optional[MuteSwitchEntity]" = None
    wake_words_changed: bool = False
    muted: bool = False
    connected: bool = False

    def save_preferences(self) -> None:
        """Save preferences as JSON."""
        _LOGGER.debug("Saving preferences: %s", self.preferences_path)
        self.preferences_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.preferences_path, "w", encoding="utf-8") as preferences_file:
            json.dump(
                asdict(self.preferences), preferences_file, ensure_ascii=False, indent=4
            )

    def set_media_volume(self, volume: float, *, persist: bool = True) -> None:
        """Update and optionally persist the normalized media volume."""

        volume = max(0.0, min(1.0, float(volume)))

        if math.isclose(self.media_volume, volume, rel_tol=0.0, abs_tol=1e-4):
            return

        _LOGGER.debug("Updating stored media volume to %.3f", volume)

        self.media_volume = volume
        self.preferences.media_volume = volume

        if persist:
            self.save_preferences()
