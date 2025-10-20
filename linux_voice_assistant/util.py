"""Utility methods."""

from __future__ import annotations

import json
import logging
import platform
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Dict, Optional

from .models import AvailableWakeWord, WakeWordType, make_wake_word_unique_id

_LOGGER = logging.getLogger(__name__)


def get_mac() -> str:
    mac = uuid.getnode()
    mac_str = ":".join(f"{(mac >> i) & 0xff:02x}" for i in range(40, -1, -8))
    return mac_str


def call_all(*callables: Optional[Callable[[], None]]) -> None:
    for item in filter(None, callables):
        item()


def is_arm() -> bool:
    machine = platform.machine()
    return ("arm" in machine) or ("aarch" in machine)


def discover_wake_word_libraries(
    base_dir: Path,
) -> Dict[str, Dict[str, AvailableWakeWord]]:
    """Scan ``base_dir`` for wake word libraries and available models."""

    libraries: Dict[str, Dict[str, AvailableWakeWord]] = {}

    if not base_dir.exists():
        _LOGGER.warning("Wake word directory does not exist: %s", base_dir)
        return libraries

    for library_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        library_name = library_dir.name
        models: Dict[str, AvailableWakeWord] = {}

        for config_path in sorted(library_dir.glob("*.json")):
            if config_path.stem == "stop":
                continue

            try:
                with open(config_path, "r", encoding="utf-8") as config_file:
                    config = json.load(config_file)
            except (OSError, json.JSONDecodeError):
                _LOGGER.warning("Failed to load wake word config: %s", config_path, exc_info=True)
                continue

            model_type = config.get("type")
            wake_word = config.get("wake_word")
            if not model_type or not wake_word:
                _LOGGER.warning("Invalid wake word config (missing type/wake_word): %s", config_path)
                continue

            try:
                wake_word_type = WakeWordType(model_type)
            except ValueError:
                _LOGGER.warning("Unknown wake word type '%s' in %s", model_type, config_path)
                continue

            model_id = config_path.stem
            available_wake_word = AvailableWakeWord(
                id=make_wake_word_unique_id(library_name, model_id),
                library=library_name,
                model_id=model_id,
                type=wake_word_type,
                wake_word=wake_word,
                trained_languages=config.get("trained_languages", []),
                config_path=config_path,
            )
            models[model_id] = available_wake_word

        libraries[library_name] = models

    return libraries
