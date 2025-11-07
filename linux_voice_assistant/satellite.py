"""Voice satellite protocol."""

import asyncio
import logging
import re
import time
from contextlib import suppress
from collections.abc import Iterable
from typing import Dict, Optional, Set, Union

# pylint: disable=no-name-in-module
from aioesphomeapi.api_pb2 import (  # type: ignore[attr-defined]
    DeviceInfoRequest,
    DeviceInfoResponse,
    ListEntitiesDoneResponse,
    ListEntitiesRequest,
    MediaPlayerCommandRequest,
    SubscribeHomeAssistantStatesRequest,
    DisconnectRequest,
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
from pymicro_wakeword import MicroWakeWord
from pyopen_wakeword import OpenWakeWord

from .api_server import APIServer
from .entity import MediaPlayerEntity, MuteSwitchEntity
from .models import ServerState
from .util import call_all

_LOGGER = logging.getLogger(__name__)

PROTO_TO_MESSAGE_TYPE = {v: k for k, v in MESSAGE_TYPE_TO_PROTO.items()}

_RESTART_DELAY_BASE = 0.1
_RESTART_DELAY_MAX = 3.0
_MAX_RESTART_ATTEMPTS = 6
_WAKE_WORD_INTERRUPT_GRACE = 1.0

class VoiceSatelliteProtocol(APIServer):

    def __init__(self, state: ServerState) -> None:
        super().__init__(state.name)

        self.state = state
        self.state.satellite = self
        self.state.connected = False

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
                initial_volume=state.media_volume,
                on_volume_changed=state.set_media_volume,
            )
            self.state.entities.append(self.state.media_player_entity)
        elif self.state.media_player_entity not in self.state.entities:
            self.state.entities.append(self.state.media_player_entity)

        self.state.media_player_entity.server = self
        self.state.media_player_entity.set_volume_callback(self.state.set_media_volume)
        self.state.media_player_entity.apply_volume_from_state(self.state.media_volume)

        # Add/update mute switch entity (like ESPHome Voice PE)
        mute_switch = self.state.mute_switch_entity
        if mute_switch is None:
            mute_switch = MuteSwitchEntity(
                server=self,
                key=len(state.entities),
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

        self._is_streaming_audio = False
        self._tts_url: Optional[str] = None
        self._tts_played = False
        self._continue_conversation = False
        self._timer_finished = False
        self._wake_word_interrupt_used = False
        self._pipeline_active = False
        self._audio_stream_open = False
        self._pending_pipeline_stop = False
        self._pending_purge_events = 0
        self._queued_wake_word: Optional[str] = None
        self._restart_task: Optional[asyncio.Task[None]] = None
        self._active_wake_word_phrase: Optional[str] = None
        self._queued_restart_attempts = 0
        self._pipeline_started_at = 0.0
        self._grace_ignore_logged = False
        self._reset_pending_logged = False
        self._stop_logged = False
        self._is_ducked = False

        self._disconnect_event = asyncio.Event()

        # Ensure any stale state is cleared on startup
        self._purge_pipeline("startup", notify=False)

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
            # Resume normal operation - wake word detection will be active again
            pass

    def handle_voice_event(
        self, event_type: VoiceAssistantEventType, data: Dict[str, str]
    ) -> None:
        _LOGGER.debug("Voice event: type=%s, data=%s", event_type.name, data)

        if event_type == VoiceAssistantEventType.VOICE_ASSISTANT_ERROR:
            self._handle_pipeline_error(data)
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_RUN_START:
            self._tts_url = data.get("url")
            self._tts_played = False
            self._continue_conversation = False
            self._pipeline_active = True
            self._audio_stream_open = False
            self._pending_pipeline_stop = False
            self._pending_purge_events = 0
            self._queued_restart_attempts = 0
        elif event_type in (
            VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_END,
            VoiceAssistantEventType.VOICE_ASSISTANT_STT_END,
        ):
            self._stop_audio_stream()

            if self._is_no_speech_event(data):
                self._purge_pipeline(
                    "no speech detected",
                    send_stop_request=True,
                    expect_followup=True,
                )
                self._tts_played = True
                self._continue_conversation = False
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
            self._stop_audio_stream()
            self._pipeline_active = False

            if self._pending_purge_events > 0:
                self._pending_purge_events -= 1

                if self._pending_purge_events == 0:
                    self._pending_pipeline_stop = False
                    queued_phrase = self._queued_wake_word
                    if queued_phrase:
                        self._schedule_pipeline_restart(queued_phrase)
                    else:
                        self._wake_word_interrupt_used = False

                self._tts_played = False
                return

            if not self._tts_played:
                self._tts_finished()

            self._tts_played = False

        # TODO: handle error

    def handle_timer_event(
        self,
        event_type: VoiceAssistantTimerEventType,
        msg: VoiceAssistantTimerEventResponse,
    ) -> None:
        _LOGGER.debug("Timer event: type=%s", event_type.name)
        if event_type == VoiceAssistantTimerEventType.VOICE_ASSISTANT_TIMER_FINISHED:
            if not self._timer_finished:
                self.state.active_wake_words.add(self.state.stop_word.id)
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

            self.state.active_wake_words.add(self.state.stop_word.id)
            self._continue_conversation = msg.start_conversation

            self.duck()
            yield from self.state.media_player_entity.play(
                urls, announcement=True, done_callback=self._tts_finished
            )
        elif isinstance(msg, VoiceAssistantTimerEventResponse):
            self.handle_timer_event(VoiceAssistantTimerEventType(msg.event_type), msg)
        elif isinstance(msg, DeviceInfoRequest):
            # Compute dynamic device name
            base_name = re.sub(r'[\s-]+', '-', self.state.name.lower()).strip('-')
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
                SwitchCommandRequest,
            ),
        ):
            for entity in self.state.entities:
                yield from entity.handle_message(msg)

            if isinstance(msg, ListEntitiesRequest):
                yield ListEntitiesDoneResponse()
        elif isinstance(msg, VoiceAssistantConfigurationRequest):
            yield VoiceAssistantConfigurationResponse(
                available_wake_words=[
                    VoiceAssistantWakeWord(
                        id=ww.id,
                        wake_word=ww.wake_word,
                        trained_languages=ww.trained_languages,
                    )
                    for ww in self.state.available_wake_words.values()
                ],
                active_wake_words=[
                    ww.id
                    for ww in self.state.wake_words.values()
                    if ww.id in self.state.active_wake_words
                ],
                max_active_wake_words=2,
            )
            _LOGGER.info("Connected to Home Assistant")
        elif isinstance(msg, VoiceAssistantSetConfiguration):
            # Change active wake words
            active_wake_words: Set[str] = set()

            for wake_word_id in msg.active_wake_words:
                if wake_word_id in self.state.wake_words:
                    # Already active
                    active_wake_words.add(wake_word_id)
                    continue

                model_info = self.state.available_wake_words.get(wake_word_id)
                if not model_info:
                    continue

                _LOGGER.debug("Loading wake word: %s", model_info.wake_word_path)
                self.state.wake_words[wake_word_id] = model_info.load()

                _LOGGER.info("Wake word set: %s", wake_word_id)
                active_wake_words.add(wake_word_id)
                break

            self.state.active_wake_words = active_wake_words
            _LOGGER.debug("Active wake words: %s", active_wake_words)

            self.state.preferences.active_wake_words = list(active_wake_words)
            self.state.save_preferences()
            self.state.wake_words_changed = True

    def handle_audio(self, audio_chunk: bytes) -> None:

        if not self._is_streaming_audio or self.state.muted:
            return

        if not self._audio_stream_open:
            self._audio_stream_open = True
        self.send_messages([VoiceAssistantAudio(data=audio_chunk)])

    def _purge_pipeline(
        self,
        reason: str,
        *,
        notify: bool = True,
        reset_interrupt: bool = True,
        send_stop_request: bool = False,
        expect_followup: bool = False,
        preserve_queue: bool = False,
        log: bool = True,
        preserve_stop_filter: bool = False,
    ) -> None:
        """Reset local and remote pipeline state."""

        if log:
            _LOGGER.debug("Purging voice assistant pipeline (%s)", reason)

        self._stop_audio_stream()
        self._is_streaming_audio = False
        self._tts_url = None
        self._tts_played = False
        self._continue_conversation = False
        self._timer_finished = False
        self._pipeline_active = False
        self._active_wake_word_phrase = None
        self._pipeline_started_at = 0.0
        self._grace_ignore_logged = False
        self._reset_pending_logged = False
        self._stop_logged = False

        if self._restart_task is not None:
            self._restart_task.cancel()
            self._restart_task = None

        if reset_interrupt:
            self._wake_word_interrupt_used = False

        if not preserve_queue:
            self._queued_wake_word = None
            self._queued_restart_attempts = 0

        if expect_followup:
            self._pending_purge_events += 1
            self._pending_pipeline_stop = True
        else:
            self._pending_purge_events = 0
            self._pending_pipeline_stop = False

        self.state.active_wake_words.discard(self.state.stop_word.id)
        self.state.stop_word.is_active = False
        self.state.stop_filter.reset(keep_cooldown=preserve_stop_filter)

        with suppress(Exception):
            self.state.tts_player.stop()

        # Ensure any ducked audio is restored.
        with suppress(Exception):
            self.unduck()

        if send_stop_request and self.state.connected:
            self.send_messages([VoiceAssistantRequest(start=False)])

        if notify and self.state.connected:
            self.send_messages([VoiceAssistantAnnounceFinished()])

    def reset_pipeline(self, reason: str, *, notify: bool = True) -> None:
        """Public helper to reset the pipeline state."""

        self._purge_pipeline(reason, notify=notify)

    @staticmethod
    def _is_no_speech_event(data: Dict[str, str]) -> bool:
        """Return True if event data indicates no speech was detected."""

        speech_detected = data.get("speech_detected")
        if speech_detected and speech_detected.lower() in {"0", "false", "no"}:
            return True

        for key in ("error_code", "code", "error", "message"):
            value = data.get(key)
            if value and ("no_speech" in value.lower() or "no speech" in value.lower()):
                return True

        transcript = data.get("text") or data.get("stt_text") or data.get("result")
        if transcript is not None and not transcript.strip():
            if speech_detected is not None or "stt_stream" in data or "stt_language" in data:
                return True

        return False

    def _start_pipeline_for_wake_word(self, phrase: str) -> None:
        """Start a new pipeline for the provided wake word."""

        self._queued_wake_word = None
        self._wake_word_interrupt_used = False
        self._pending_pipeline_stop = False
        self._pipeline_active = True
        self._is_streaming_audio = True
        self._audio_stream_open = False
        self._active_wake_word_phrase = phrase
        self._pipeline_started_at = time.monotonic()
        self._grace_ignore_logged = False
        self._reset_pending_logged = False
        self._stop_logged = False

        _LOGGER.debug("Detected wake word: %s", phrase)
        with suppress(Exception):
            self.state.tts_player.stop()
        self.send_messages([VoiceAssistantRequest(start=True, wake_word_phrase=phrase)])
        self.duck()

        if self.state.wakeup_sound:
            self.state.tts_player.play(self.state.wakeup_sound)

    def _schedule_pipeline_restart(self, phrase: str) -> None:
        """Restart the pipeline after Home Assistant acknowledges the stop."""

        if self._restart_task is not None:
            self._restart_task.cancel()
            self._restart_task = None

        if self._queued_restart_attempts >= _MAX_RESTART_ATTEMPTS:
            _LOGGER.warning(
                "Aborting wake word restart for '%s' after %s attempts",
                phrase,
                self._queued_restart_attempts,
            )
            self._queued_wake_word = None
            self._wake_word_interrupt_used = False
            self._pending_pipeline_stop = False
            self._pending_purge_events = 0
            self._queued_restart_attempts = 0
            self._active_wake_word_phrase = None
            return

        delay = min(
            _RESTART_DELAY_BASE * (2 ** self._queued_restart_attempts),
            _RESTART_DELAY_MAX,
        )

        _LOGGER.debug(
            "Scheduling pipeline restart for '%s' in %.3f seconds",
            phrase,
            delay,
        )

        async def _restart_after_delay() -> None:
            try:
                await asyncio.sleep(delay)
                if self._queued_wake_word == phrase:
                    self._queued_restart_attempts += 1
                    self._start_pipeline_for_wake_word(phrase)
            except asyncio.CancelledError:
                pass
            finally:
                self._restart_task = None

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        self._restart_task = loop.create_task(_restart_after_delay())

    def _handle_pipeline_error(self, data: Dict[str, str]) -> None:
        """Handle pipeline errors by purging the active run."""

        error_code = data.get("error_code") or data.get("code") or data.get("error")
        message = data.get("message") or data.get("error_message") or ""

        if error_code or message:
            _LOGGER.debug(
                "Pipeline error received: code=%s message=%s", error_code, message
            )
        else:
            _LOGGER.debug("Pipeline error received: %s", data)

        normalized_code = (error_code or "").lower()
        normalized_message = message.lower()

        if (
            "duplicate_wake_up_detected" in normalized_code
            or "duplicate wake-up" in normalized_message
        ):
            phrase = self._active_wake_word_phrase or self._queued_wake_word

            if phrase:
                self._queued_wake_word = phrase
                self._wake_word_interrupt_used = True
                _LOGGER.debug(
                    "Duplicate wake detected while restarting; will retry '%s'",
                    phrase,
                )

                self._purge_pipeline(
                    "error: duplicate_wake_up_detected",
                    notify=True,
                    reset_interrupt=False,
                    send_stop_request=True,
                    expect_followup=True,
                    preserve_queue=True,
                )
                return

        self._purge_pipeline(
            f"error: {error_code or 'unknown'}",
            notify=True,
            send_stop_request=True,
            expect_followup=True,
        )

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

        wake_word_phrase = wake_word.wake_word

        if self._pipeline_active:
            if (
                not self._wake_word_interrupt_used
                and (time.monotonic() - self._pipeline_started_at)
                < _WAKE_WORD_INTERRUPT_GRACE
            ):
                if not self._grace_ignore_logged:
                    _LOGGER.debug("Wake word ignored because pipeline just started")
                    self._grace_ignore_logged = True
                return

            if self._wake_word_interrupt_used or self._queued_wake_word is not None:
                if not self._reset_pending_logged:
                    _LOGGER.debug("Wake word ignored because pipeline reset is pending")
                    self._reset_pending_logged = True
                return

            self._queued_wake_word = wake_word_phrase
            self._wake_word_interrupt_used = True
            self._queued_restart_attempts = 0

            self._purge_pipeline(
                "wake word interrupt",
                reset_interrupt=False,
                send_stop_request=True,
                expect_followup=True,
                preserve_queue=True,
            )
            return

        if self._queued_wake_word is not None:
            if not self._reset_pending_logged:
                _LOGGER.debug("Wake word ignored because pipeline reset is pending")
                self._reset_pending_logged = True
            return

        if self._pending_purge_events > 0 or self._pending_pipeline_stop:
            if not self._reset_pending_logged:
                _LOGGER.debug("Wake word ignored because pipeline reset is pending")
                self._reset_pending_logged = True
            return

        self._wake_word_interrupt_used = False
        self._queued_restart_attempts = 0
        self._start_pipeline_for_wake_word(wake_word_phrase)

    def stop(self) -> None:
        if not self._stop_logged:
            _LOGGER.debug("Stop word detected")
            self._stop_logged = True

        if self._timer_finished:
            self._timer_finished = False
            _LOGGER.debug("Stopping timer finished sound")

        _LOGGER.debug(
            "Stop handler invoked (pipeline_active=%s, pending_stop=%s, pending_events=%d, "
            "queued_wake=%s, restart_task=%s, streaming_audio=%s, tts_url=%s, tts_played=%s)",
            self._pipeline_active,
            self._pending_pipeline_stop,
            self._pending_purge_events,
            self._queued_wake_word,
            self._restart_task is not None,
            self._is_streaming_audio,
            bool(self._tts_url),
            self._tts_played,
        )

        if self._pending_pipeline_stop or self._pending_purge_events > 0:
            _LOGGER.debug(
                "Stop word requested while purge in-flight; reinforcing remote stop"
            )
            # A stop request is already in flight; ensure we cancel any pending
            # restarts and reinforce the remote stop without spamming additional
            # purge cycles.
            self._stop_audio_stream()
            with suppress(Exception):
                self.state.tts_player.stop()

            if self._restart_task is not None:
                self._restart_task.cancel()
                self._restart_task = None

            self._queued_wake_word = None
            self._queued_restart_attempts = 0
            self._wake_word_interrupt_used = False
            self._active_wake_word_phrase = None

            if self.state.connected:
                self.send_messages([VoiceAssistantRequest(start=False)])

            self.state.stop_word.is_active = False
            return

        should_notify = bool(
            self._pipeline_active
            or self._pending_pipeline_stop
            or self._pending_purge_events
            or self._is_streaming_audio
            or self._tts_played
            or self._tts_url
        )

        _LOGGER.debug(
            "Stop handler notify=%s (queued_wake=%s, restart_task=%s)",
            should_notify,
            self._queued_wake_word,
            self._restart_task is not None,
        )

        if not should_notify and not self._queued_wake_word and self._restart_task is None:
            _LOGGER.debug("Stop word halting idle playback and clearing local state")
            self._stop_audio_stream()
            with suppress(Exception):
                self.state.tts_player.stop()
            with suppress(Exception):
                self.state.music_player.stop()

            self._wake_word_interrupt_used = False
            self.state.stop_word.is_active = False
            self._queued_restart_attempts = 0

            # Maintain the filter cooldown so a single utterance does not retrigger.
            self.state.stop_filter.reset(keep_cooldown=True)

            with suppress(Exception):
                self.unduck()

            self._sync_media_player_state()
            return

        self._purge_pipeline(
            "stop word",
            notify=should_notify,
            send_stop_request=self.state.connected,
            expect_followup=should_notify,
            log=should_notify,
            preserve_stop_filter=True,
        )

    def play_tts(self) -> None:
        if (not self._tts_url) or self._tts_played:
            return

        self._tts_played = True
        _LOGGER.debug("Playing TTS response: %s", self._tts_url)

        self.state.active_wake_words.add(self.state.stop_word.id)
        self.state.tts_player.play(self._tts_url, done_callback=self._tts_finished)

    def duck(self) -> None:
        if not self._is_ducked:
            _LOGGER.debug("Ducking music")
            self._is_ducked = True
        self.state.music_player.duck()

    def unduck(self) -> None:
        if self._is_ducked:
            _LOGGER.debug("Unducking music")
            self._is_ducked = False
        self.state.music_player.unduck()

    def _stop_audio_stream(self) -> None:
        if self._audio_stream_open:
            with suppress(Exception):
                self.send_messages([VoiceAssistantAudio(end=True)])
            self._audio_stream_open = False
        self._is_streaming_audio = False

    def _sync_media_player_state(self) -> None:
        entity = self.state.media_player_entity
        if entity is None:
            return

        try:
            state_message = entity.sync_state()
        except Exception:  # pragma: no cover - defensive logging
            _LOGGER.exception("Failed to sync media player state")
            return

        if self.state.connected:
            self.send_messages([state_message])

    def _tts_finished(self) -> None:
        if self._continue_conversation:
            self.state.active_wake_words.discard(self.state.stop_word.id)
            self.state.stop_word.is_active = False

            if self.state.connected:
                self.send_messages([VoiceAssistantAnnounceFinished()])

            self._tts_url = None
            self._tts_played = False
            self._continue_conversation = False
            self._wake_word_interrupt_used = False

            self.send_messages([VoiceAssistantRequest(start=True)])
            self._is_streaming_audio = True
            self._audio_stream_open = False
            _LOGGER.debug("Continuing conversation")
            return

        self._purge_pipeline("tts finished")
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
        self._purge_pipeline("connection lost", notify=False)

        # Stop any ongoing audio playback beyond ducked volume.
        try:
            self.state.music_player.stop()
        except Exception:  # pragma: no cover - defensive safety net
            _LOGGER.exception("Failed to stop music player during disconnect")
        else:
            self._sync_media_player_state()
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
