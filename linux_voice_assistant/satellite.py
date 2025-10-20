"""Voice satellite protocol."""

import asyncio
import logging
import re
import time
from collections.abc import Iterable
from typing import Dict, List, Optional, Union

# pylint: disable=no-name-in-module
from aioesphomeapi.api_pb2 import (  # type: ignore[attr-defined]
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
from .entity import MediaPlayerEntity, MuteSwitchEntity, WakeWordLibrarySelectEntity
from .microwakeword import MicroWakeWord
from .models import (
    NO_WAKE_WORD_NAME,
    NO_WAKE_WORD_SENTINEL,
    ServerState,
    make_no_wake_word_id,
    make_wake_word_unique_id,
    parse_wake_word_unique_id,
)
from .openwakeword import OpenWakeWord
from .util import call_all, discover_wake_word_libraries

_LOGGER = logging.getLogger(__name__)

PROTO_TO_MESSAGE_TYPE = {v: k for k, v in MESSAGE_TYPE_TO_PROTO.items()}


class VoiceSatelliteProtocol(APIServer):

    def __init__(self, state: ServerState) -> None:
        super().__init__(state.name)

        self.state = state
        self.state.satellite = self
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

        self._is_streaming_audio = False
        self._tts_url: Optional[str] = None
        self._tts_played = False
        self._continue_conversation = False
        self._timer_finished = False
        self._pipeline_active = False
        self._current_assistant_index = 0
        self._pending_assistant_index: Optional[int] = None

        self._disconnect_event = asyncio.Event()

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
            self._is_streaming_audio = False
            self.state.tts_player.stop()
            # Stop any ongoing voice processing
            self.state.stop_word.is_active = False
        else:
            # voice_assistant.start_continuous behavior
            _LOGGER.debug("Unmuting voice assistant (voice_assistant.start_continuous)")
            # Resume normal operation - wake word detection will be active again
            pass

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

        if event_type == VoiceAssistantEventType.VOICE_ASSISTANT_RUN_START:
            self._tts_url = data.get("url")
            self._tts_played = False
            self._continue_conversation = False
            self._pipeline_active = True

            assistant_index = self._pending_assistant_index
            if assistant_index is None:
                assistant_index = self._current_assistant_index

            self._current_assistant_index = assistant_index
            self._pending_assistant_index = assistant_index
        elif event_type in (
            VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_END,
            VoiceAssistantEventType.VOICE_ASSISTANT_STT_END,
        ):
            self._is_streaming_audio = False
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_PROGRESS:
            if data.get("tts_start_streaming") == "1":
                # Start streaming early
                self.play_tts()
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_END:
            if data.get("continue_conversation") == "1":
                self._continue_conversation = True
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_TTS_END:
            self._tts_url = data.get("url")
            self.play_tts()
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END:
            self._is_streaming_audio = False
            if not self._tts_played:
                self._tts_finished()

            self._tts_played = False
            self._pending_assistant_index = None

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

        if self._pipeline_active:
            _LOGGER.debug("Ignoring wake word while pipeline is active")
            return

        wake_word_phrase = wake_word.wake_word
        wake_word_id = getattr(wake_word, "id", None)
        if isinstance(wake_word_id, str):
            assistant_index = self._assistant_index_for_wake_word(wake_word_id)
        else:
            assistant_index = 0

        self._pending_assistant_index = assistant_index
        _LOGGER.debug("Detected wake word: %s", wake_word_phrase)
        self.send_messages(
            [VoiceAssistantRequest(start=True, wake_word_phrase=wake_word_phrase)]
        )
        self.duck()
        self._is_streaming_audio = True
        self._pipeline_active = True
        self.state.tts_player.play(self.state.wakeup_sound)

    def stop(self) -> None:
        self.state.stop_word.is_active = False
        self.state.tts_player.stop()
        self._continue_conversation = False
        self._pipeline_active = False

        if self._timer_finished:
            self._timer_finished = False
            _LOGGER.debug("Stopping timer finished sound")
        else:
            _LOGGER.debug("TTS response stopped manually")
            self._tts_finished()

    def play_tts(self) -> None:
        if (not self._tts_url) or self._tts_played:
            return

        self._tts_played = True
        _LOGGER.debug("Playing TTS response: %s", self._tts_url)

        self.state.stop_word.is_active = True
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
        self.state.stop_word.is_active = False
        self.send_messages([VoiceAssistantAnnounceFinished()])

        continue_conversation = self._continue_conversation
        self._continue_conversation = False

        if continue_conversation:
            self._pending_assistant_index = self._current_assistant_index
            self.send_messages([VoiceAssistantRequest(start=True)])
            self._is_streaming_audio = True
            self._pipeline_active = True
            _LOGGER.debug("Continuing conversation")
        else:
            self._pipeline_active = False
            self.unduck()

        _LOGGER.debug("TTS response finished")

    def _play_timer_finished(self) -> None:
        if not self._timer_finished:
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
        self._is_streaming_audio = False
        self._tts_url = None
        self._tts_played = False
        self._continue_conversation = False
        self._timer_finished = False
        self._pipeline_active = False

        # Stop any ongoing audio playback and wake/stop word processing.
        try:
            self.state.music_player.stop()
        except Exception:  # pragma: no cover - defensive safety net
            _LOGGER.exception("Failed to stop music player during disconnect")

        try:
            self.state.tts_player.stop()
        except Exception:  # pragma: no cover - defensive safety net
            _LOGGER.exception("Failed to stop TTS player during disconnect")

        self.state.stop_word.is_active = False
        self.state.connected = False
        if self.state.satellite is self:
            self.state.satellite = None

        if self.state.mute_switch_entity is not None:
            self.state.mute_switch_entity.sync_with_state()

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
