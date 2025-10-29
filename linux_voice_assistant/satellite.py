"""Voice satellite protocol."""

import asyncio
import logging
import signal
import re
import subprocess
import threading
import time
from collections.abc import Iterable
from typing import Dict, List, Optional, Union

from dataclasses import dataclass

# pylint: disable=no-name-in-module
from aioesphomeapi.api_pb2 import (  # type: ignore[attr-defined]
    ButtonCommandRequest,
    DeviceInfoRequest,
    DeviceInfoResponse,
    HomeassistantServiceMap,
    HomeassistantServiceResponse,
    ListEntitiesDoneResponse,
    ListEntitiesRequest,
    MediaPlayerCommandRequest,
    SelectCommandRequest,
    SubscribeHomeAssistantStatesRequest,
    SwitchCommandRequest,
    VoiceAssistantAnnounceFinished,
    VoiceAssistantAnnounceRequest,
    VoiceAssistantAudio,
    VoiceAssistantConfigurationRequest,
    VoiceAssistantConfigurationResponse,
    VoiceAssistantEventResponse,
    VoiceAssistantRequest,
    VoiceAssistantSetConfiguration,
    VoiceAssistantTimerEventResponse,
    VoiceAssistantWakeWord,
    ConnectRequest,
)
from aioesphomeapi.core import MESSAGE_TYPE_TO_PROTO

from aioesphomeapi.model import (
    VoiceAssistantEventType,
    VoiceAssistantFeature,
    VoiceAssistantTimerEventType,
)
from google.protobuf import message

from .api_server import APIServer
from .entity import (
    ConfigButtonEntity,
    ConfigSwitchEntity,
    MediaPlayerEntity,
    MuteSwitchEntity,
    WakeWordLibrarySelectEntity,
)
from .microwakeword import MicroWakeWord
from .models import (
    NO_WAKE_WORD_NAME,
    NO_WAKE_WORD_SENTINEL,
    ServerState,
    make_no_wake_word_id,
    make_wake_word_unique_id,
    parse_wake_word_unique_id,
    WAKE_WORD_SENSITIVITY_LABELS,
)
from .openwakeword import OpenWakeWord
from .util import call_all, discover_wake_word_libraries

_LOGGER = logging.getLogger(__name__)

PROTO_TO_MESSAGE_TYPE = {v: k for k, v in MESSAGE_TYPE_TO_PROTO.items()}

_WAKE_RESTART_COOLDOWN = 0.5
_STT_NO_SPEECH_TIMEOUT = 5.0


@dataclass
class QueuedWakeWord:
    phrase: str
    assistant_index: int
    sound_played: bool = False
    timestamp: float = 0.0


class VoiceSatelliteProtocol(APIServer):

    def __init__(self, state: ServerState) -> None:
        super().__init__(state.name)

        self._is_streaming_audio = False
        self._audio_stream_open = False
        self._tts_url: Optional[str] = None
        self._tts_played = False
        self._continue_conversation = False
        self._timer_finished = False
        self._pipeline_active = False
        self._current_assistant_index = 0
        self._pending_assistant_index: Optional[int] = None
        self._disconnect_event = asyncio.Event()
        self._pipeline_restart_available = False
        self._last_wake_event_time: Optional[float] = None
        self._heard_speech_during_stt = False
        self._pending_purge_events = 0
        self._stt_timeout_handle: Optional[asyncio.TimerHandle] = None
        self._queued_wake_word: Optional[QueuedWakeWord] = None
        self._restart_waiting_for_ack = False
        self._queued_wake_word_pending_start = False
        self._suppress_stale_events = False
        self._last_wake_ignore_reason: Optional[str] = None
        self._last_wake_ignore_time = 0.0
        self._last_ignored_event_type: Optional[VoiceAssistantEventType] = None
        self._last_ignored_event_time = 0.0
        self._run_end_received = True

        self.state = state
        self.state.connected = False

        self._refresh_wake_word_libraries()

        existing_media_players = [
            entity
            for entity in self.state.entities
            if isinstance(entity, MediaPlayerEntity)
        ]
        if existing_media_players:
            # Keep the first instance and remove any extras.
            self.state.media_player_entity = existing_media_players[0]
            for extra in existing_media_players[1:]:
                self.state.entities.remove(extra)

        existing_mute_switches = [
            entity
            for entity in self.state.entities
            if isinstance(entity, MuteSwitchEntity)
        ]
        if existing_mute_switches:
            self.state.mute_switch_entity = existing_mute_switches[0]
            for extra in existing_mute_switches[1:]:
                self.state.entities.remove(extra)

        if self.state.media_player_entity is None:
            self.state.media_player_entity = MediaPlayerEntity(
                server=self,
                key=len(state.entities),
                name="Media Player",
                object_id="linux_voice_assistant_media_player",
                music_player=state.music_player,
                announce_player=state.tts_player,
                initial_volume=state.volume,
            )
            self.state.entities.append(self.state.media_player_entity)
        elif self.state.media_player_entity not in self.state.entities:
            self.state.entities.append(self.state.media_player_entity)

        self.state.media_player_entity.server = self
        self.state.media_player_entity.volume = state.volume
        self.state.media_player_entity.previous_volume = state.volume

        # Add/update mute switch entity (like ESPHome Voice PE)
        mute_switch = self.state.mute_switch_entity
        if mute_switch is None:
            mute_switch = MuteSwitchEntity(
                server=self,
                key=len(self.state.entities),
                name="Mute",
                object_id="mute",
                get_muted=lambda: self.state.muted,
                set_muted=self._set_muted,
            )
            self.state.entities.append(mute_switch)
            self.state.mute_switch_entity = mute_switch
        elif mute_switch not in self.state.entities:
            self.state.entities.append(mute_switch)

        mute_switch.server = self
        mute_switch.update_get_muted(lambda: self.state.muted)
        mute_switch.update_set_muted(self._set_muted)
        mute_switch.sync_with_state()

        existing_library_selects = [
            entity
            for entity in self.state.entities
            if isinstance(entity, WakeWordLibrarySelectEntity)
        ]
        if existing_library_selects:
            self.state.wake_word_library_entity = existing_library_selects[0]
            for extra in existing_library_selects[1:]:
                self.state.entities.remove(extra)

        library_select = self.state.wake_word_library_entity
        if library_select is None:
            library_select = WakeWordLibrarySelectEntity(
                server=self,
                key=len(self.state.entities),
                name="Wake word lib",
                object_id="wake_word_library",
                get_options=self._get_wake_word_library_options,
                get_state=self._get_wake_word_library_state,
                set_state=self._set_wake_word_library,
            )
            self.state.entities.append(library_select)
            self.state.wake_word_library_entity = library_select
        else:
            library_select.server = self
            library_select.update_callbacks(
                self._get_wake_word_library_options,
                self._get_wake_word_library_state,
                self._set_wake_word_library,
            )
            if library_select not in self.state.entities:
                self.state.entities.append(library_select)

        sensitivity_select = self.state.wake_word_sensitivity_entity
        if sensitivity_select is None:
            sensitivity_select = WakeWordLibrarySelectEntity(
                server=self,
                key=len(self.state.entities),
                name="Wake word sensitivity",
                object_id="wake_word_sensitivity",
                get_options=self._get_wake_word_sensitivity_options,
                get_state=self._get_wake_word_sensitivity_state,
                set_state=self._set_wake_word_sensitivity,
            )
            self.state.entities.append(sensitivity_select)
            self.state.wake_word_sensitivity_entity = sensitivity_select
        else:
            sensitivity_select.server = self
            sensitivity_select.update_callbacks(
                self._get_wake_word_sensitivity_options,
                self._get_wake_word_sensitivity_state,
                self._set_wake_word_sensitivity,
            )
            if sensitivity_select not in self.state.entities:
                self.state.entities.append(sensitivity_select)

        wake_sound_switch = self.state.wake_sound_switch_entity
        if wake_sound_switch is None:
            wake_sound_switch = ConfigSwitchEntity(
                server=self,
                key=len(self.state.entities),
                name="Wake sound",
                object_id="wake_sound",
                icon="mdi:bullhorn",
                get_state=self._get_wake_sound_enabled,
                set_state=self._set_wake_sound_enabled,
            )
            self.state.entities.append(wake_sound_switch)
            self.state.wake_sound_switch_entity = wake_sound_switch
        else:
            wake_sound_switch.server = self
            wake_sound_switch.update_callbacks(
                self._get_wake_sound_enabled,
                self._set_wake_sound_enabled,
            )
            if wake_sound_switch not in self.state.entities:
                self.state.entities.append(wake_sound_switch)
        wake_sound_switch.sync_with_state()

        timer_sound_switch = self.state.timer_sound_switch_entity
        if timer_sound_switch is None:
            timer_sound_switch = ConfigSwitchEntity(
                server=self,
                key=len(self.state.entities),
                name="Timer finished",
                object_id="timer_finished_sound",
                icon="mdi:timer",
                get_state=self._get_timer_sound_enabled,
                set_state=self._set_timer_sound_enabled,
            )
            self.state.entities.append(timer_sound_switch)
            self.state.timer_sound_switch_entity = timer_sound_switch
        else:
            timer_sound_switch.server = self
            timer_sound_switch.update_callbacks(
                self._get_timer_sound_enabled,
                self._set_timer_sound_enabled,
            )
            if timer_sound_switch not in self.state.entities:
                self.state.entities.append(timer_sound_switch)
        timer_sound_switch.sync_with_state()

        reset_button = self.state.reset_assistant_button_entity
        if reset_button is None:
            reset_button = ConfigButtonEntity(
                server=self,
                key=len(self.state.entities),
                name="Reset assistant",
                object_id="reset_assistant",
                icon="mdi:restart",
                press_callback=self._reset_assistant,
            )
            self.state.entities.append(reset_button)
            self.state.reset_assistant_button_entity = reset_button
        else:
            reset_button.server = self
            reset_button.update_callback(self._reset_assistant)
            if reset_button not in self.state.entities:
                self.state.entities.append(reset_button)

        restart_button = self.state.restart_device_button_entity
        if restart_button is None:
            restart_button = ConfigButtonEntity(
                server=self,
                key=len(self.state.entities),
                name="Restart device",
                object_id="restart_device",
                icon="mdi:power",
                press_callback=self._restart_device,
            )
            self.state.entities.append(restart_button)
            self.state.restart_device_button_entity = restart_button
        else:
            restart_button.server = self
            restart_button.update_callback(self._restart_device)
            if restart_button not in self.state.entities:
                self.state.entities.append(restart_button)

        self.state.satellite = self
        self.reset_pipeline(
            "startup",
            notify_finished=False,
            reset_assistant_index=True,
        )

    def _assistant_index_for_wake_word(self, wake_word_id: str) -> int:
        """Return the assistant index associated with a wake word id."""

        selections = (
            self.state.preferences.assistant.wake_word,
            self.state.preferences.assistant_2.wake_word,
        )

        for index, selection in enumerate(selections):
            if selection.model == NO_WAKE_WORD_NAME:
                continue

            expected_id = make_wake_word_unique_id(selection.library, selection.model)
            if expected_id == wake_word_id:
                return index

        return 0

    def _set_muted(self, new_state: bool) -> None:
        self.state.muted = bool(new_state)

        if self.state.muted:
            # voice_assistant.stop behavior
            _LOGGER.debug("Muting voice assistant (voice_assistant.stop)")
            self._stop_audio_stream()
            self.state.tts_player.stop()
            # Stop any ongoing voice processing
            self.state.stop_word.is_active = False
        else:
            # voice_assistant.start_continuous behavior
            _LOGGER.debug("Unmuting voice assistant (voice_assistant.start_continuous)")
            # Resume normal operation - stop word detection will be active again
            self.state.stop_word.is_active = True

    def _refresh_wake_word_libraries(self) -> None:
        libraries = discover_wake_word_libraries(self.state.wakewords_dir)
        if not libraries:
            return

        self.state.update_wake_word_libraries(libraries)

    def _get_wake_word_library_options(self) -> List[str]:
        self._refresh_wake_word_libraries()
        return sorted(self.state.available_wake_word_libraries.keys())

    def _get_wake_word_library_state(self) -> str:
        return self.state.active_wake_word_library

    def _set_wake_word_library(self, new_library: str) -> bool:
        if not new_library:
            return False

        self._refresh_wake_word_libraries()
        changed = self.state.set_active_wake_word_library(new_library)
        if changed:
            _LOGGER.info("Wake word library set: %s", new_library)
            self.state.save_preferences()
            self._reload_esphome_config_entry()
        return changed

    @staticmethod
    def _slugify_name(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return slug or "linux_voice_assistant"

    def _get_wake_word_sensitivity_options(self) -> List[str]:
        return list(WAKE_WORD_SENSITIVITY_LABELS)

    def _get_wake_word_sensitivity_state(self) -> str:
        return self.state.wake_word_sensitivity

    def _set_wake_word_sensitivity(self, new_label: str) -> bool:
        changed = self.state.set_wake_word_sensitivity(new_label)
        if changed:
            _LOGGER.info("Wake word sensitivity set: %s", new_label)
        return changed

    def _get_wake_sound_enabled(self) -> bool:
        return self.state.wake_sound_enabled

    def _set_wake_sound_enabled(self, enabled: bool) -> None:
        changed = self.state.set_wake_sound_enabled(enabled)
        if changed:
            _LOGGER.info(
                "Wake sound %s",
                "enabled" if enabled else "disabled",
            )

    def _get_timer_sound_enabled(self) -> bool:
        return self.state.timer_sound_enabled

    def _set_timer_sound_enabled(self, enabled: bool) -> None:
        changed = self.state.set_timer_sound_enabled(enabled)
        if changed:
            _LOGGER.info(
                "Timer finished sound %s",
                "enabled" if enabled else "disabled",
            )
            if not enabled and self._timer_finished:
                self._timer_finished = False
                self.state.tts_player.stop()
                self.unduck()

    def _reset_assistant(self) -> None:
        self._run_background_command(
            [
                "/usr/bin/systemctl",
                "--user",
                "restart",
                "linux-voice-assistant.service",
            ],
            "Restarting linux-voice-assistant service",
        )

    def _restart_device(self) -> None:
        self._run_background_command(
            ["sudo", "/usr/bin/systemctl", "reboot"],
            "Rebooting device",
        )

    def _run_background_command(self, command: List[str], log_message: str) -> None:
        def _execute() -> None:
            _LOGGER.info(log_message)
            try:
                completed = subprocess.run(command, check=False)
            except FileNotFoundError:
                _LOGGER.exception("Command not found: %s", command[0])
            except Exception:  # pragma: no cover - defensive safety net
                _LOGGER.exception("Unexpected error running command: %s", command)
            else:
                returncode = completed.returncode
                if returncode < 0:
                    signal_number = -returncode
                    try:
                        signal_name = signal.Signals(signal_number).name
                    except ValueError:
                        signal_name = str(signal_number)
                    _LOGGER.debug(
                        "Command terminated by signal %s (%s): %s",
                        signal_number,
                        signal_name,
                        command,
                    )
                elif returncode > 0:
                    _LOGGER.error(
                        "Command exited with code %s: %s",
                        returncode,
                        command,
                    )

        threading.Thread(target=_execute, daemon=True).start()

    def _wake_word_library_entity_id(self) -> str:
        slug = self._slugify_name(self.state.name)
        return f"select.{slug}_wake_word_lib"

    def _reload_esphome_config_entry(self) -> None:
        entity_id = self._wake_word_library_entity_id()
        self.send_messages(
            [
                HomeassistantServiceResponse(
                    service="homeassistant.reload_config_entry",
                    data=[
                        HomeassistantServiceMap(
                            key="entity_id",
                            value=entity_id,
                        )
                    ],
                )
            ]
        )

    def handle_voice_event(
        self, event_type: VoiceAssistantEventType, data: Dict[str, str]
    ) -> None:
        _LOGGER.debug("Voice event: type=%s, data=%s", event_type.name, data)

        if (
            self._suppress_stale_events
            and event_type
            not in (
                VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END,
                VoiceAssistantEventType.VOICE_ASSISTANT_ERROR,
            )
        ):
            now = time.monotonic()
            if (
                self._last_ignored_event_type != event_type
                or (now - self._last_ignored_event_time) >= 0.5
            ):
                _LOGGER.debug(
                    "Ignoring voice event %s while waiting for pipeline shutdown",
                    event_type.name,
                )
                self._last_ignored_event_type = event_type
                self._last_ignored_event_time = now
            return

        if event_type == VoiceAssistantEventType.VOICE_ASSISTANT_ERROR:
            _LOGGER.warning("Voice assistant error: %s", data)
            self._purge_active_pipeline(
                "voice assistant error",
                notify_finished=False,
                record_pending_finish=False,
                send_stop_request=True,
            )
            return
        if event_type == VoiceAssistantEventType.VOICE_ASSISTANT_RUN_START:
            # Enable the stop word for the duration of the pipeline so the
            # conversation can be cancelled even if Home Assistant delivers
            # audio via streaming events instead of an announce request.  The
            # flag will be cleared as part of the general pipeline reset
            # logic inside ``_purge_active_pipeline`` and ``_tts_finished``.
            self.state.stop_word.is_active = True
            self._run_end_received = False
            self._tts_url = data.get("url")
            self._tts_played = False
            self._continue_conversation = False
            self._pipeline_active = True
            self._start_audio_stream()

            assistant_index = self._pending_assistant_index
            if assistant_index is None:
                assistant_index = self._current_assistant_index

            self._current_assistant_index = assistant_index
            self._pending_assistant_index = assistant_index
            self._restart_waiting_for_ack = False
            self._queued_wake_word_pending_start = False
            self._queued_wake_word = None
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_STT_START:
            self._heard_speech_during_stt = False
            self._schedule_stt_timeout()
            self._start_audio_stream()
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_START:
            self._heard_speech_during_stt = True
            self._cancel_stt_timeout()
        elif event_type in (
            VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_END,
            VoiceAssistantEventType.VOICE_ASSISTANT_STT_END,
        ):
            self._cancel_stt_timeout()
            self._stop_audio_stream()
            if event_type == VoiceAssistantEventType.VOICE_ASSISTANT_STT_END:
                if self._has_nonempty_speech_text(data):
                    self._heard_speech_during_stt = True

                if not self._heard_speech_during_stt and self._pipeline_active:
                    _LOGGER.debug(
                        "No speech detected during STT; resetting pipeline"
                    )
                    self._purge_active_pipeline(
                        "no speech detected",
                        notify_finished=True,
                        record_pending_finish=True,
                        send_stop_request=True,
                    )
                    return

                self._heard_speech_during_stt = False
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_PROGRESS:
            if data.get("tts_start_streaming") == "1":
                # Start streaming early
                self.play_tts()
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_END:
            if self._pipeline_active and data.get("continue_conversation") == "1":
                self._continue_conversation = True
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_TTS_END:
            self._tts_url = data.get("url")
            self.play_tts()
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END:
            self._run_end_received = True
            self._stop_audio_stream()
            if self._pending_purge_events > 0:
                self._pending_purge_events -= 1
                _LOGGER.debug(
                    "Skipping pipeline finish due to pending purge (remaining=%s)",
                    self._pending_purge_events,
                )
                if self._pending_purge_events == 0:
                    self._suppress_stale_events = False
                    self._last_ignored_event_type = None
                if self._queued_wake_word is not None:
                    if self._restart_waiting_for_ack:
                        _LOGGER.debug(
                            "Restart queued wake word after pipeline end"
                        )
                        self._restart_waiting_for_ack = False
                        self._start_queued_wake_word()
                    elif self._queued_wake_word_pending_start:
                        _LOGGER.debug(
                            "Starting queued wake word after pipeline end"
                        )
                        self._start_queued_wake_word()
            elif not self._tts_played:
                self._tts_finished()

            if not (self._restart_waiting_for_ack or self._queued_wake_word_pending_start):
                self._pending_assistant_index = None
            if not self._queued_wake_word_pending_start:
                self._pipeline_restart_available = False
            if not self._restart_waiting_for_ack:
                self._last_wake_event_time = None

        # TODO: handle error

    def handle_timer_event(
        self,
        event_type: VoiceAssistantTimerEventType,
        msg: VoiceAssistantTimerEventResponse,
    ) -> None:
        _LOGGER.debug("Timer event: type=%s", event_type.name)
        if event_type == VoiceAssistantTimerEventType.VOICE_ASSISTANT_TIMER_FINISHED:
            if not self._timer_finished:
                self.state.stop_word.is_active = True
                self._timer_finished = True
                self.duck()
                self._play_timer_finished()

    def handle_message(self, msg: message.Message) -> Iterable[message.Message]:
        if isinstance(msg, VoiceAssistantEventResponse):
            # Pipeline event
            data: Dict[str, str] = {}
            for arg in msg.data:
                data[arg.name] = arg.value

            self.handle_voice_event(VoiceAssistantEventType(msg.event_type), data)
        elif isinstance(msg, VoiceAssistantAnnounceRequest):
            _LOGGER.debug("Announcing: %s", msg.text)

            assert self.state.media_player_entity is not None

            urls = []
            if msg.preannounce_media_id:
                urls.append(msg.preannounce_media_id)

            urls.append(msg.media_id)

            # Treat announcements like active TTS playback so the stop word can
            # cancel them and the normal finish callback logic runs.
            self._tts_played = True
            self.state.stop_word.is_active = True
            self._continue_conversation = msg.start_conversation

            self.duck()
            yield from self.state.media_player_entity.play(
                urls, announcement=True, done_callback=self._tts_finished
            )
        elif isinstance(msg, VoiceAssistantTimerEventResponse):
            self.handle_timer_event(VoiceAssistantTimerEventType(msg.event_type), msg)
        elif isinstance(msg, DeviceInfoRequest):
            # Compute dynamic device name
            base_name = re.sub(r"[\s-]+", "-", self.state.name.lower()).strip("-")
            mac_no_colon = self.state.mac_address.replace(":", "").lower()
            mac_last6 = mac_no_colon[-6:]
            device_name = f"{base_name}-{mac_last6}"

            yield DeviceInfoResponse(
                uses_password=False,
                name=device_name,
                mac_address=self.state.mac_address,
                manufacturer="Open Home Foundation",
                model="Linux Voice Assistant",
                voice_assistant_feature_flags=(
                    VoiceAssistantFeature.VOICE_ASSISTANT
                    | VoiceAssistantFeature.API_AUDIO
                    | VoiceAssistantFeature.ANNOUNCE
                    | VoiceAssistantFeature.START_CONVERSATION
                    | VoiceAssistantFeature.TIMERS
                ),
            )
        elif isinstance(
            msg,
            (
                ListEntitiesRequest,
                SubscribeHomeAssistantStatesRequest,
                MediaPlayerCommandRequest,
                SelectCommandRequest,
                SwitchCommandRequest,
                ButtonCommandRequest,
            ),
        ):
            for entity in self.state.entities:
                yield from entity.handle_message(msg)

            if isinstance(msg, ListEntitiesRequest):
                yield ListEntitiesDoneResponse()
        elif isinstance(msg, VoiceAssistantConfigurationRequest):
            self._refresh_wake_word_libraries()
            self.state.sync_wake_word_models()

            available_models = self.state.get_active_library_wake_words()
            library = self.state.active_wake_word_library

            available_wake_words = [
                VoiceAssistantWakeWord(
                    id=ww.id,
                    wake_word=ww.wake_word,
                    trained_languages=ww.trained_languages,
                )
                for ww in available_models.values()
            ]

            available_wake_words.append(
                VoiceAssistantWakeWord(
                    id=make_no_wake_word_id(library),
                    wake_word=NO_WAKE_WORD_NAME,
                    trained_languages=[],
                )
            )

            yield VoiceAssistantConfigurationResponse(
                available_wake_words=available_wake_words,
                active_wake_words=self.state.get_active_wake_word_ids(),
                max_active_wake_words=2,
            )
            _LOGGER.info("Connected to Home Assistant")
        elif isinstance(msg, VoiceAssistantSetConfiguration):
            self._refresh_wake_word_libraries()

            requested_ids = list(msg.active_wake_words)
            library = self.state.active_wake_word_library
            no_id = make_no_wake_word_id(library)

            while len(requested_ids) < 2:
                requested_ids.append(no_id)

            selections = [
                self.state.preferences.assistant.wake_word,
                self.state.preferences.assistant_2.wake_word,
            ]

            changed = False
            library_models = self.state.get_active_library_wake_words()

            for index, wake_word_id in enumerate(requested_ids[:2]):
                selection = selections[index]
                lib_name, model_id = parse_wake_word_unique_id(
                    wake_word_id, library
                )

                if lib_name != library:
                    continue

                if model_id == NO_WAKE_WORD_SENTINEL or wake_word_id == no_id:
                    if selection.model != NO_WAKE_WORD_NAME:
                        selection.model = NO_WAKE_WORD_NAME
                        changed = True
                else:
                    if model_id not in library_models:
                        _LOGGER.warning("Unrecognized wake word id: %s", wake_word_id)
                        continue

                    if selection.model != model_id:
                        selection.model = model_id
                        changed = True

                selection.library = library

            self.state.sync_wake_word_models()
            _LOGGER.debug(
                "Active wake word ids: %s", self.state.get_active_wake_word_ids()
            )

            if changed:
                self.state.save_preferences()

    def handle_audio(self, audio_chunk: bytes) -> None:

        if not self._is_streaming_audio or self.state.muted:
            return

        self.send_messages([VoiceAssistantAudio(data=audio_chunk)])

    def _start_audio_stream(self) -> None:
        self._is_streaming_audio = True
        if not self._audio_stream_open:
            self._audio_stream_open = True

    def _stop_audio_stream(self) -> None:
        if self._audio_stream_open:
            self.send_messages([VoiceAssistantAudio(end=True)])
            self._audio_stream_open = False
        self._is_streaming_audio = False

    def _purge_active_pipeline(
        self,
        reason: str,
        *,
        notify_finished: bool,
        reset_continue: bool = True,
        record_pending_finish: bool = False,
        reset_assistant_index: bool = False,
        clear_tts_flag: bool = True,
        preserve_restart_queue: bool = False,
        send_stop_request: bool = False,
        reactivate_stop_word: bool = True,
    ) -> None:
        _LOGGER.debug("Purging pipeline (%s)", reason)

        self._cancel_stt_timeout()

        self._stop_audio_stream()

        try:
            self.state.tts_player.stop()
        except Exception:  # pragma: no cover - defensive safety net
            _LOGGER.exception("Failed to stop TTS playback during pipeline purge")

        self.state.stop_word.is_active = False

        if reset_continue:
            self._continue_conversation = False

        if clear_tts_flag:
            self._tts_played = False

        self._pipeline_active = False
        self._pipeline_restart_available = False
        self._pending_assistant_index = None
        self._tts_url = None
        self._timer_finished = False
        self._last_wake_event_time = None
        self._heard_speech_during_stt = False
        self._restart_waiting_for_ack = False

        if reset_assistant_index:
            self._current_assistant_index = 0

        if record_pending_finish and not self._run_end_received:
            self._pending_purge_events += 1
            self._suppress_stale_events = True
        else:
            self._suppress_stale_events = False
            if not record_pending_finish:
                self._run_end_received = True

        should_notify = notify_finished and self.state.connected
        should_send_stop = send_stop_request and self.state.connected

        if should_send_stop:
            self.send_messages([VoiceAssistantRequest(start=False)])

        if should_notify:
            self.send_messages([VoiceAssistantAnnounceFinished()])

        try:
            self.unduck()
        except Exception:  # pragma: no cover - defensive safety net
            _LOGGER.exception("Failed to unduck audio during pipeline purge")

        if not record_pending_finish and self._pending_purge_events:
            # Ensure stale counters do not linger when not expecting follow-up events.
            self._pending_purge_events = 0
            self._suppress_stale_events = False
            self._last_ignored_event_type = None

        if not preserve_restart_queue:
            self._queued_wake_word = None
            self._queued_wake_word_pending_start = False

        if reactivate_stop_word and not self.state.muted:
            self.state.stop_word.is_active = True

    def reset_pipeline(
        self,
        reason: str,
        *,
        notify_finished: bool = False,
        reset_assistant_index: bool = False,
    ) -> None:
        """Reset the active pipeline state without expecting follow-up events."""

        self._purge_active_pipeline(
            reason,
            notify_finished=notify_finished,
            reset_assistant_index=reset_assistant_index,
            record_pending_finish=False,
        )

    @staticmethod
    def _has_nonempty_speech_text(data: Dict[str, str]) -> bool:
        for key in (
            "stt_output_text",
            "stt_output",
            "stt_text",
            "text",
            "transcript",
        ):
            value = data.get(key)
            if value and value.strip():
                return True

        for key in ("speech_detected", "has_speech", "stt_speech_detected"):
            value = data.get(key)
            if value and value.strip().lower() in {"1", "true", "yes"}:
                return True

        return False

    def _log_wake_ignore(self, message: str) -> None:
        now = time.monotonic()
        if (
            self._last_wake_ignore_reason == message
            and (now - self._last_wake_ignore_time) < 0.5
        ):
            return

        self._last_wake_ignore_reason = message
        self._last_wake_ignore_time = now
        _LOGGER.debug(message)

    def wakeup(self, wake_word: Union[MicroWakeWord, OpenWakeWord]) -> None:
        if self._timer_finished:
            # Stop timer instead
            self._timer_finished = False
            self.state.tts_player.stop()
            _LOGGER.debug("Stopping timer finished sound")
            return

        if self.state.muted:
            # Don't respond to wake words when muted (voice_assistant.stop behavior)
            return

        now = time.monotonic()

        wake_word_phrase = wake_word.wake_word
        wake_word_id = getattr(wake_word, "id", None)
        if isinstance(wake_word_id, str):
            assistant_index = self._assistant_index_for_wake_word(wake_word_id)
        else:
            assistant_index = 0

        if self._pipeline_active:
            if not self._pipeline_restart_available:
                self._log_wake_ignore("Ignoring wake word while pipeline is active")
                return

            if (
                self._last_wake_event_time is not None
                and (now - self._last_wake_event_time) < _WAKE_RESTART_COOLDOWN
            ):
                self._log_wake_ignore(
                    "Ignoring wake word restart request due to cooldown"
                )
                return

            _LOGGER.debug(
                "Wake word detected while pipeline active; restarting conversation"
            )
            self._queued_wake_word = QueuedWakeWord(
                phrase=wake_word_phrase,
                assistant_index=assistant_index,
                timestamp=now,
            )
            self._queued_wake_word_pending_start = True
            self._purge_active_pipeline(
                "wake word restart",
                notify_finished=True,
                record_pending_finish=True,
                preserve_restart_queue=True,
                send_stop_request=True,
            )
            return

        if self._pending_purge_events > 0:
            self._log_wake_ignore(
                "Ignoring wake word while previous pipeline is finishing"
            )
            return

        if self._restart_waiting_for_ack or self._queued_wake_word_pending_start:
            self._log_wake_ignore("Ignoring wake word while restart is pending")
            return

        self._pipeline_restart_available = True
        self._queued_wake_word = QueuedWakeWord(
            phrase=wake_word_phrase,
            assistant_index=assistant_index,
            timestamp=now,
        )
        self._queued_wake_word_pending_start = True
        self._start_queued_wake_word(now=now)

    def _start_queued_wake_word(self, *, now: Optional[float] = None) -> None:
        if self._queued_wake_word is None:
            return

        queued = self._queued_wake_word
        if now is None:
            now = time.monotonic()

        queued.timestamp = now
        self._last_wake_event_time = now
        self._pending_assistant_index = queued.assistant_index
        self._queued_wake_word_pending_start = False
        self._last_wake_ignore_reason = None

        # Activate the stop word immediately so users can cancel during the
        # brief window before Home Assistant confirms the pipeline start.
        self.state.stop_word.is_active = True
        self.state.stop_word_cooldown_until = 0.0
        self.state.stop_word_last_detection = 0.0
        _LOGGER.debug("Detected wake word: %s", queued.phrase)
        self.send_messages(
            [VoiceAssistantRequest(start=True, wake_word_phrase=queued.phrase)]
        )
        self.duck()

        if (
            self.state.wake_sound_enabled
            and self.state.wakeup_sound
            and not queued.sound_played
        ):
            self.state.tts_player.play(self.state.wakeup_sound)
            queued.sound_played = True

        self._restart_waiting_for_ack = True

    def _schedule_stt_timeout(self) -> None:
        loop = getattr(self, "_loop", None)
        if loop is None:
            return

        self._cancel_stt_timeout()
        self._stt_timeout_handle = loop.call_later(
            _STT_NO_SPEECH_TIMEOUT, self._handle_stt_timeout
        )

    def _cancel_stt_timeout(self) -> None:
        if self._stt_timeout_handle is not None:
            self._stt_timeout_handle.cancel()
            self._stt_timeout_handle = None

    def _handle_stt_timeout(self) -> None:
        self._stt_timeout_handle = None
        if not self._pipeline_active or self._heard_speech_during_stt:
            return

        _LOGGER.debug("STT timeout without detected speech; purging pipeline")
        self._purge_active_pipeline(
            "stt timeout without speech",
            notify_finished=True,
            record_pending_finish=True,
            send_stop_request=True,
        )

    def stop(self) -> None:
        pipeline_running = bool(
            (self._pipeline_active and not self._run_end_received)
            or self._audio_stream_open
            or self._is_streaming_audio
        )

        awaiting_pipeline_start = self._restart_waiting_for_ack
        tts_active = bool(self._tts_played or self._tts_url)

        music_playing = self.state.music_player.is_playing
        finishing_pipeline = self._pending_purge_events > 0
        stop_music_only = (
            music_playing
            and not pipeline_running
            and not awaiting_pipeline_start
            and not tts_active
            and not self._timer_finished
            and not finishing_pipeline
        )

        if self._timer_finished:
            _LOGGER.info("Stop word cancelled timer chime")
        elif pipeline_running:
            _LOGGER.info("Stop word cancelling active pipeline")
        elif awaiting_pipeline_start:
            _LOGGER.info("Stop word cancelled pending pipeline start")
        elif tts_active:
            _LOGGER.info("Stop word stopped pending TTS playback")
        elif stop_music_only:
            _LOGGER.info("Stop word stopped media playback")
        else:
            _LOGGER.info("Stop word detected with nothing active to cancel")

        should_notify = bool(pipeline_running or tts_active)

        self._purge_active_pipeline(
            "stop word",
            notify_finished=should_notify,
            record_pending_finish=pipeline_running,
            send_stop_request=pipeline_running or awaiting_pipeline_start,
        )

        if stop_music_only:
            try:
                self.state.music_player.stop()
                media_entity = self.state.media_player_entity
                if media_entity is not None:
                    new_state = media_entity._update_state(
                        media_entity._determine_state()
                    )
                    if self.state.connected:
                        self.send_messages([new_state])
            except Exception:  # pragma: no cover - defensive safety net
                _LOGGER.exception(
                    "Failed to stop media playback after stop word trigger"
                )

    def play_tts(self) -> None:
        if (not self._tts_url) or self._tts_played:
            return

        self._tts_played = True
        _LOGGER.debug("Playing TTS response: %s", self._tts_url)

        self.state.stop_word.is_active = True
        self.state.stop_word_cooldown_until = 0.0
        self.state.stop_word_last_detection = 0.0
        self.state.tts_player.play(self._tts_url, done_callback=self._tts_finished)

    def duck(self) -> None:
        _LOGGER.debug("Ducking audio output")
        self.state.music_player.duck()
        if self.state.tts_player is not self.state.music_player and self.state.tts_player.is_playing:
            self.state.tts_player.duck()

    def unduck(self) -> None:
        _LOGGER.debug("Unducking audio output")
        self.state.music_player.unduck()
        if self.state.tts_player is not self.state.music_player:
            self.state.tts_player.unduck()

    def _tts_finished(self) -> None:
        if (
            not self._tts_played
            and not self._continue_conversation
            and not self._pipeline_active
        ):
            _LOGGER.debug(
                "Ignoring TTS finished callback with no active TTS playback",
            )
            return

        continue_conversation = self._continue_conversation
        self._continue_conversation = False

        if continue_conversation:
            # Keep the stop word armed for the follow-up turn so users can
            # immediately cancel if they change their mind.
            self.state.stop_word.is_active = True
            self.state.stop_word_cooldown_until = 0.0
            self.state.stop_word_last_detection = 0.0
        else:
            self.state.stop_word.is_active = False

        self.send_messages([VoiceAssistantAnnounceFinished()])

        if continue_conversation:
            self._pending_assistant_index = self._current_assistant_index
            # Ensure the stop word remains active between turns, even before the
            # pipeline start event reaffirms it.
            self.state.stop_word.is_active = True
            self.state.stop_word_cooldown_until = 0.0
            self.state.stop_word_last_detection = 0.0
            self.send_messages([VoiceAssistantRequest(start=True)])
            self._pipeline_restart_available = True
            self._last_wake_event_time = None
            self._restart_waiting_for_ack = True
            _LOGGER.debug("Continuing conversation")
        else:
            self._purge_active_pipeline(
                "tts finished",
                notify_finished=False,
                reset_continue=False,
                record_pending_finish=False,
                clear_tts_flag=False,
            )
        self._tts_played = False
        _LOGGER.debug("TTS response finished")

    def _play_timer_finished(self) -> None:
        if not self._timer_finished:
            self.unduck()
            return

        if (not self.state.timer_sound_enabled) or (not self.state.timer_finished_sound):
            self._timer_finished = False
            self.unduck()
            return

        self.state.tts_player.play(
            self.state.timer_finished_sound,
            done_callback=lambda: call_all(
                lambda: time.sleep(1.0), self._play_timer_finished
            ),
        )

    def connection_lost(self, exc):
        super().connection_lost(exc)

        self._disconnect_event.set()
        self._purge_active_pipeline(
            "connection lost",
            notify_finished=False,
            record_pending_finish=False,
            reset_assistant_index=True,
        )
        self._pending_purge_events = 0

        # Stop any ongoing audio playback and wake/stop word processing.
        try:
            self.state.music_player.stop()
        except Exception:  # pragma: no cover - defensive safety net
            _LOGGER.exception("Failed to stop music player during disconnect")

        self.state.connected = False
        if self.state.satellite is self:
            self.state.satellite = None

        if self.state.mute_switch_entity is not None:
            self.state.mute_switch_entity.sync_with_state()
        if self.state.wake_sound_switch_entity is not None:
            self.state.wake_sound_switch_entity.sync_with_state()
        if self.state.timer_sound_switch_entity is not None:
            self.state.timer_sound_switch_entity.sync_with_state()

        _LOGGER.info("Disconnected from Home Assistant; waiting for reconnection")

    def process_packet(self, msg_type: int, packet_data: bytes) -> None:
        super().process_packet(msg_type, packet_data)

        if msg_type == PROTO_TO_MESSAGE_TYPE[ConnectRequest]:
            self.state.connected = True
            # Send states after connect
            states = []
            for entity in self.state.entities:
                states.extend(entity.handle_message(SubscribeHomeAssistantStatesRequest()))
            self.send_messages(states)
            _LOGGER.debug("Sent entity states after connect")
