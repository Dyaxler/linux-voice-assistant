#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import signal
import threading
import time
from contextlib import suppress
from pathlib import Path
from queue import Queue
from typing import List, Optional, Union

import numpy as np
import sounddevice as sd

from . import api_server
from .microwakeword import MicroWakeWord, MicroWakeWordFeatures
from .models import NO_WAKE_WORD_NAME, Preferences, ServerState, WakeWordType
from .mpv_player import MpvMediaPlayer
from .openwakeword import OpenWakeWord, OpenWakeWordFeatures
from .satellite import VoiceSatelliteProtocol
from .util import discover_wake_word_libraries, get_mac, is_arm
from .zeroconf import HomeAssistantZeroconf

_LOGGER = logging.getLogger(__name__)
_MODULE_DIR = Path(__file__).parent
_REPO_DIR = _MODULE_DIR.parent
_WAKEWORDS_DIR = _REPO_DIR / "wakewords"
_OWW_DIR = _WAKEWORDS_DIR / "openWakeWord"
_SOUNDS_DIR = _REPO_DIR / "sounds"

if is_arm():
    _LIB_DIR = _REPO_DIR / "lib" / "linux_arm64"
else:
    _LIB_DIR = _REPO_DIR / "lib" / "linux_amd64"


# -----------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--audio-input-device",
        default="default",
        help="sounddevice name for input device",
    )
    parser.add_argument("--audio-input-block-size", type=int, default=1024)
    parser.add_argument("--audio-output-device", help="mpv name for output device")
    parser.add_argument(
        "--stop-model",
        default="stop",
        help=(
            "Stop wake word model. Accepts a model id, an optional"
            " 'library:model' pair, or a direct path to a JSON config."
        ),
    )
    #
    parser.add_argument(
        "--oww-melspectrogram-model",
        default=_OWW_DIR / "melspectrogram.tflite",
        help="Path to openWakeWord melspectrogram model",
    )
    parser.add_argument(
        "--oww-embedding-model",
        default=_OWW_DIR / "embedding_model.tflite",
        help="Path to openWakeWord embedding model",
    )
    #
    parser.add_argument(
        "--wakeup-sound", default=str(_SOUNDS_DIR / "wake_word_triggered.flac")
    )
    parser.add_argument(
        "--timer-finished-sound", default=str(_SOUNDS_DIR / "timer_finished.flac")
    )
    #
    parser.add_argument("--preferences-file", default=_REPO_DIR / "preferences.json")
    #
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Address for ESPHome server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=6053, help="Port for ESPHome server (default: 6053)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="Include verbose network keepalive logging",
    )
    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        _LOGGER.debug("Ignoring unknown arguments: %s", unknown_args)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    api_server.set_network_logging_enabled(args.network)
    _LOGGER.debug(args)

    wake_word_libraries = discover_wake_word_libraries(_WAKEWORDS_DIR)
    if not wake_word_libraries:
        raise RuntimeError(f"No wake word libraries found in {_WAKEWORDS_DIR}")

    _LOGGER.debug(
        "Discovered wake word libraries: %s", sorted(wake_word_libraries.keys())
    )

    library_names = sorted(wake_word_libraries.keys())
    default_library = (
        "microWakeWord" if "microWakeWord" in wake_word_libraries else library_names[0]
    )

    preferred_default_model_ids = ("okay_nabu",)

    def _default_model_for_library(library_name: str) -> str:
        library_models = wake_word_libraries.get(library_name, {})
        if not library_models:
            return NO_WAKE_WORD_NAME

        for model_id in preferred_default_model_ids:
            if model_id in library_models:
                return model_id

        return next(iter(sorted(library_models.keys())))

    default_model = _default_model_for_library(default_library)

    # Load preferences
    preferences_path = Path(args.preferences_file)
    preferences_path_exists = preferences_path.exists()
    preferences_data = None
    if preferences_path_exists:
        _LOGGER.debug("Loading preferences: %s", preferences_path)
        with open(preferences_path, "r", encoding="utf-8") as preferences_file:
            preferences_data = json.load(preferences_file)

    preferences = Preferences.load(
        preferences_data,
        default_library=default_library,
        default_model=default_model,
        default_second_model=NO_WAKE_WORD_NAME,
    )

    initial_volume = preferences.volume if preferences.volume is not None else 1.0
    initial_volume = max(0.0, min(1.0, float(initial_volume)))
    preferences.volume = initial_volume

    active_library = preferences.assistant.wake_word.library
    if active_library not in wake_word_libraries:
        active_library = default_library
    if not active_library:
        active_library = default_library

    preferences.assistant.wake_word.library = active_library
    preferences.assistant_2.wake_word.library = active_library

    libtensorflowlite_c_path = _LIB_DIR / "libtensorflowlite_c.so"
    _LOGGER.debug("libtensorflowlite_c path: %s", libtensorflowlite_c_path)

    stop_argument = str(args.stop_model)
    stop_config_path: Optional[Path] = None
    stop_library_name: Optional[str] = None
    stop_model_id: Optional[str] = None

    stop_path_candidate = Path(stop_argument).expanduser()
    if stop_path_candidate.suffix:
        candidate_paths = []
        if stop_path_candidate.is_absolute():
            candidate_paths.append(stop_path_candidate)
        else:
            candidate_paths.extend(
                [
                    _REPO_DIR / stop_path_candidate,
                    _WAKEWORDS_DIR / stop_path_candidate,
                    stop_path_candidate,
                ]
            )

        for candidate_path in candidate_paths:
            if candidate_path.exists():
                stop_config_path = candidate_path
                stop_library_name = candidate_path.parent.name
                stop_model_id = candidate_path.stem
                break
    else:
        library_hint: Optional[str] = None
        model_hint = stop_argument.strip()

        if ":" in model_hint:
            library_part, model_part = model_hint.split(":", 1)
            library_hint = library_part.strip() or None
            model_hint = model_part.strip()

        if not model_hint:
            raise RuntimeError("Stop model id cannot be empty")

        search_directories = []
        if library_hint:
            candidate_dir = _WAKEWORDS_DIR / library_hint
            if not candidate_dir.is_dir():
                raise RuntimeError(
                    f"Stop wake word library '{library_hint}' not found in {_WAKEWORDS_DIR}"
                )
            search_directories.append(candidate_dir)
        else:
            search_directories = [
                library_dir
                for library_dir in sorted(_WAKEWORDS_DIR.iterdir())
                if library_dir.is_dir()
            ]

        for library_dir in search_directories:
            candidate_path = library_dir / f"{model_hint}.json"
            if candidate_path.exists():
                stop_config_path = candidate_path
                stop_library_name = library_dir.name
                stop_model_id = model_hint
                break

    if stop_config_path is None:
        raise RuntimeError(
            f"Unable to locate stop wake word config for argument '{stop_argument}'"
        )

    try:
        with open(stop_config_path, "r", encoding="utf-8") as stop_config_file:
            stop_config = json.load(stop_config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to load stop wake word config: {stop_config_path}") from exc

    stop_type = stop_config.get("type")
    if stop_type != WakeWordType.MICRO_WAKE_WORD.value:
        raise RuntimeError(
            "Stop wake word must be a microWakeWord model. "
            f"Config {stop_config_path} reports type '{stop_type}'."
        )

    _LOGGER.debug(
        "Loading stop model: %s (library=%s, model_id=%s)",
        stop_config_path,
        stop_library_name,
        stop_model_id,
    )
    stop_model = MicroWakeWord.from_config(
        stop_config_path, libtensorflowlite_c_path
    )
    stop_model.use_probability(True)
    stop_model.set_probability_cutoff(
        max(stop_model.get_probability_cutoff(), _STOP_WORD_PROBABILITY_CUTOFF)
    )
    stop_model.set_sliding_window_size(_STOP_WORD_SLIDING_WINDOW_SIZE)
    stop_model.use_probability(True)
    stop_model.reset()
    _LOGGER.debug(
        "Configured stop model sensitivity (cutoff=%.2f, window=%d)",
        stop_model.get_probability_cutoff(),
        stop_model.sliding_window_size,
    )

    state = ServerState(
        name=args.name,
        mac_address=get_mac(),
        audio_queue=Queue(),
        entities=[],
        wakewords_dir=_WAKEWORDS_DIR,
        available_wake_word_libraries=wake_word_libraries,
        active_wake_word_library=active_library,
        wake_words={},
        stop_word=stop_model,
        music_player=MpvMediaPlayer(device=args.audio_output_device),
        tts_player=MpvMediaPlayer(device=args.audio_output_device),
        wakeup_sound=args.wakeup_sound,
        timer_finished_sound=args.timer_finished_sound,
        preferences=preferences,
        preferences_path=preferences_path,
        libtensorflowlite_c_path=libtensorflowlite_c_path,
        oww_melspectrogram_path=Path(args.oww_melspectrogram_model),
        oww_embedding_path=Path(args.oww_embedding_model),
        volume=initial_volume,
        preferred_default_model_ids=preferred_default_model_ids,
        wake_word_sensitivity=preferences.wake_word_sensitivity,
        wake_word_probability_cutoff=preferences.wake_word_probability,
        wake_sound_enabled=preferences.wake_sound_enabled,
        timer_sound_enabled=preferences.timer_sound_enabled,
    )

    if not preferences_path_exists:
        state.save_preferences()

    state.sync_wake_word_models()

    initial_volume_percent = int(round(initial_volume * 100))
    state.music_player.set_volume(initial_volume_percent)
    state.tts_player.set_volume(initial_volume_percent)

    process_audio_thread = threading.Thread(
        target=process_audio, args=(state,), daemon=True
    )
    process_audio_thread.start()

    def sd_callback(indata, _frames, _time, _status):
        state.audio_queue.put_nowait(bytes(indata))

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    registered_signals: List[int] = []

    def _request_shutdown(signame: str) -> None:
        _LOGGER.info("Received %s, shutting down", signame)
        stop_event.set()

    for signame in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signame):
            sig = getattr(signal, signame)
            try:
                loop.add_signal_handler(
                    sig,
                    lambda signame=signame: _request_shutdown(signame),
                )
            except NotImplementedError:
                _LOGGER.debug("Signal handlers not supported for %s", signame)
            else:
                registered_signals.append(sig)

    server = await loop.create_server(
        lambda: VoiceSatelliteProtocol(state), host=args.host, port=args.port
    )

    # Auto discovery (zeroconf, mDNS)
    discovery = HomeAssistantZeroconf(port=args.port, name=args.name)

    try:
        await discovery.register_server()

        try:
            _LOGGER.debug("Opening audio input device: %s", args.audio_input_device)
            with sd.RawInputStream(
                samplerate=16000,
                blocksize=args.audio_input_block_size,
                device=args.audio_input_device,
                dtype="int16",
                channels=1,
                callback=sd_callback,
            ):
                async with server:
                    _LOGGER.info(
                        "Server started (host=%s, port=%s)", args.host, args.port
                    )
                    server_task = asyncio.create_task(server.serve_forever())
                    server_task.add_done_callback(lambda _fut: stop_event.set())
                    try:
                        await stop_event.wait()
                    finally:
                        server_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await server_task
        except KeyboardInterrupt:
            _request_shutdown("KeyboardInterrupt")
    finally:
        stop_event.set()
        if state.satellite is not None:
            try:
                state.satellite.reset_pipeline(
                    "shutdown",
                    notify_finished=False,
                    reset_assistant_index=True,
                )
            except Exception:  # pragma: no cover - defensive safety net
                _LOGGER.exception("Failed to reset pipeline during shutdown")
        state.audio_queue.put_nowait(None)
        process_audio_thread.join()
        for sig in registered_signals:
            loop.remove_signal_handler(sig)
        await discovery.shutdown()

    _LOGGER.debug("Server stopped")


# -----------------------------------------------------------------------------


_STOP_WORD_EVENT_HOLD_SECONDS = 0.35
_STOP_WORD_COOLDOWN_SECONDS = 0.75
_STOP_WORD_PROBABILITY_CUTOFF = 0.8
_STOP_WORD_SLIDING_WINDOW_SIZE = 6


def process_audio(state: ServerState):
    """Process audio chunks from the microphone."""

    wake_words: List[Union[MicroWakeWord, OpenWakeWord]] = []
    micro_features: Optional[MicroWakeWordFeatures] = None
    micro_inputs: List[np.ndarray] = []

    oww_features: Optional[OpenWakeWordFeatures] = None
    oww_inputs: List[np.ndarray] = []
    has_oww = False

    try:
        while True:
            audio_chunk = state.audio_queue.get()
            if audio_chunk is None:
                break

            if state.satellite is None:
                continue

            if (not wake_words) or (state.wake_words_changed and state.wake_words):
                # Update list of wake word models to process
                state.wake_words_changed = False
                wake_words = [ww for ww in state.wake_words.values() if ww.is_active]

                has_oww = False
                for wake_word in wake_words:
                    if isinstance(wake_word, OpenWakeWord):
                        has_oww = True

                if micro_features is None:
                    micro_features = MicroWakeWordFeatures(
                        libtensorflowlite_c_path=state.libtensorflowlite_c_path,
                    )

                if has_oww and (oww_features is None):
                    oww_features = OpenWakeWordFeatures(
                        melspectrogram_model=state.oww_melspectrogram_path,
                        embedding_model=state.oww_embedding_path,
                        libtensorflowlite_c_path=state.libtensorflowlite_c_path,
                    )

            try:
                state.satellite.handle_audio(audio_chunk)

                assert micro_features is not None
                micro_inputs.clear()
                micro_inputs.extend(micro_features.process_streaming(audio_chunk))

                if has_oww:
                    assert oww_features is not None
                    oww_inputs.clear()
                    oww_inputs.extend(oww_features.process_streaming(audio_chunk))

                for wake_word in wake_words:
                    activated = False
                    if isinstance(wake_word, MicroWakeWord):
                        for micro_input in micro_inputs:
                            if wake_word.process_streaming(micro_input):
                                activated = True
                                break
                    elif isinstance(wake_word, OpenWakeWord):
                        for oww_input in oww_inputs:
                            for prob in wake_word.process_streaming(oww_input):
                                if wake_word.should_activate(prob):
                                    activated = True
                                    break
                            if activated:
                                break

                    if activated and not state.muted:
                        state.satellite.wakeup(wake_word)
                        break

                # Always process to keep state correct
                stopped = False
                for micro_input in micro_inputs:
                    if state.stop_word.process_streaming(micro_input):
                        stopped = True

                now = time.monotonic()

                if stopped:
                    state.stop_word_last_detection = now
                    state.stop_word_reset_pending_log = True
                    if state.muted:
                        _LOGGER.info(
                            "Stop word detected while muted; ignoring trigger"
                        )
                        state.stop_word.reset()
                        state.stop_word_event_active = True
                        continue

                    if not state.stop_word.is_active:
                        _LOGGER.debug(
                            "Stop word detected while handler inactive; forwarding"
                        )

                    if now < state.stop_word_cooldown_until:
                        state.stop_word.reset()
                        state.stop_word_event_active = True
                        continue

                    if not state.stop_word_event_active:
                        _LOGGER.info("Stop word detected; requesting cancellation")
                        state.satellite.stop()
                        state.stop_word_cooldown_until = (
                            now + _STOP_WORD_COOLDOWN_SECONDS
                        )

                    state.stop_word.reset()
                    state.stop_word_event_active = True
                else:
                    if state.stop_word_event_active:
                        time_since_detection = now - state.stop_word_last_detection
                        if time_since_detection >= _STOP_WORD_EVENT_HOLD_SECONDS:
                            state.stop_word.reset()
                            state.stop_word_event_active = False
                            if state.stop_word_reset_pending_log:
                                _LOGGER.info("Stop word detector reset after cooldown")
                                state.stop_word_reset_pending_log = False
            except Exception:
                _LOGGER.exception("Unexpected error handling audio")

    except Exception:
        _LOGGER.exception("Unexpected error processing audio")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
