"""
Standalone translation support for OpenAI gpt-realtime-translate model.

This module subclasses livekit-plugins-openai's RealtimeModel and RealtimeSession
to add translation support without modifying the original plugin.

Usage:
    from translation import TranslationModel

    model = TranslationModel(
        api_key="your-api-key",
        target_language="hi",
        input_audio_transcription=AudioTranscription(
            model="gpt-realtime-whisper",
            language="en",
        ),
    )
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import aiohttp
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from openai.types.beta.realtime.session import InputAudioNoiseReduction
from openai.types.realtime import (
    AudioTranscription,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    NoiseReductionType,
    RealtimeAudioInputTurnDetection,
)
from openai.types.realtime.realtime_audio_config_input import NoiseReduction

from livekit.plugins.openai.realtime.realtime_model import (
    RealtimeModel,
    RealtimeSession,
    _MessageGeneration,
    _ResponseGeneration,
    OPENAI_BASE_URL,
    SAMPLE_RATE,
    NUM_CHANNELS,
)

# Delays for translation flush logic
DEFAULT_TRANSLATION_INPUT_FLUSH_DELAY = 0.2   # seconds of silence → emit user item
DEFAULT_TRANSLATION_OUTPUT_FALLBACK_DELAY = 1  # seconds of silence → emit assistant item (fallback)

TRANSLATION_ENDPOINT_PATH = "/realtime/translations"
TRANSLATION_MODEL = "gpt-realtime-translate"


class TranslationModel(RealtimeModel):
    """
    RealtimeModel subclass for gpt-realtime-translate.

    Connects to the dedicated /v1/realtime/translations endpoint and
    handles the translation-specific event protocol.

    Args:
        target_language (str): BCP-47 language code for output, e.g. "hi", "es", "fr".
        safety_identifier (str | None): Hashed user ID sent as OpenAI-Safety-Identifier header.
        api_key (str | None): OpenAI API key.
        input_audio_transcription: AudioTranscription config. Use gpt-realtime-whisper for
            streaming source-language transcripts.
        input_audio_noise_reduction: Noise reduction config.
        http_session: Optional shared aiohttp session.

    Example:
        from translation import TranslationModel
        from openai.types.realtime import AudioTranscription

        model = TranslationModel(
            target_language="hi",
            safety_identifier="hashed-user-id",
            api_key="sk-...",
            input_audio_transcription=AudioTranscription(
                model="gpt-realtime-whisper",
                language="en",
            ),
        )
    """

    def __init__(
        self,
        *,
        target_language: str,
        translation_input_flush_delay: float | None = None,
        translation_output_fallback_delay: float | None = None,
        safety_identifier: str | None = None,
        api_key: str | None = None,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[
            NoiseReductionType | NoiseReduction | InputAudioNoiseReduction | None
        ] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        **kwargs: Any,
    ) -> None:
        # Strip unsupported params for translation API
        kwargs.pop("turn_detection", None)
        kwargs.pop("tool_choice", None)
        kwargs.pop("speed", None)
        kwargs.pop("tracing", None)
        kwargs.pop("truncation", None)
        kwargs.pop("voice", None)
        kwargs.pop("modalities", None)

        super().__init__(
            model=TRANSLATION_MODEL,
            api_key=api_key,
            input_audio_transcription=input_audio_transcription,
            input_audio_noise_reduction=input_audio_noise_reduction,
            http_session=http_session,
            base_url=base_url if is_given(base_url) else OPENAI_BASE_URL,
            modalities=["audio"],
            turn_detection=None,  # translation API handles VAD internally
            **kwargs,
        )

        self._target_language = target_language
        self._safety_identifier = safety_identifier

        self._translation_input_flush_delay = (
            translation_input_flush_delay
            if translation_input_flush_delay is not None
            else DEFAULT_TRANSLATION_INPUT_FLUSH_DELAY
        )

        self._translation_output_fallback_delay = (
            translation_output_fallback_delay
            if translation_output_fallback_delay is not None
            else DEFAULT_TRANSLATION_OUTPUT_FALLBACK_DELAY
        )

    @property
    def target_language(self) -> str:
        return self._target_language

    @property
    def safety_identifier(self) -> str | None:
        return self._safety_identifier

    @property
    def translation_input_flush_delay(self) -> float:
        return self._translation_input_flush_delay

    @property
    def translation_output_fallback_delay(self) -> float:
        return self._translation_output_fallback_delay

    def session(self) -> TranslationSession:  # type: ignore[override]
        sess = TranslationSession(self)
        self._sessions.add(sess)
        return sess


class TranslationSession(RealtimeSession):
    """
    RealtimeSession subclass that handles gpt-realtime-translate protocol.

    Key differences from standard RealtimeSession:
    - Connects to /v1/realtime/translations endpoint
    - Uses session.input_audio_buffer.append instead of input_audio_buffer.append
    - Handles session.output_audio.delta / session.output_transcript.delta /
      session.input_transcript.delta events
    - Emits conversation items using VAD + silence timers as turn boundaries
    - Blocks unsupported operations (tools, instructions, chat ctx sync, generate_reply)
    """

    def __init__(self, realtime_model: TranslationModel) -> None:
        self._translation_model = realtime_model

        self._input_flush_delay = realtime_model.translation_input_flush_delay
        self._output_flush_delay = realtime_model.translation_output_fallback_delay

        # Translation transcript state
        self._translation_input_transcript: str = ""
        self._translation_output_transcript: str = ""
        self._translation_input_item_id: str | None = None
        self._translation_output_item_id: str | None = None
        self._translation_input_flush_handle: asyncio.TimerHandle | None = None
        self._translation_output_flush_handle: asyncio.TimerHandle | None = None
        self._translation_user_item_emitted: bool = False

        super().__init__(realtime_model)

    # ------------------------------------------------------------------
    # Override: WebSocket connection — translation endpoint + safety header
    # ------------------------------------------------------------------

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        from livekit.agents import APIConnectionError

        headers = {"User-Agent": "LiveKit Agents"}
        headers["Authorization"] = f"Bearer {self._translation_model._opts.api_key}"

        if self._translation_model.safety_identifier:
            headers["OpenAI-Safety-Identifier"] = self._translation_model.safety_identifier

        # Build translation URL: /v1/realtime/translations
        base = self._translation_model._opts.base_url
        if base.startswith("http"):
            base = base.replace("http", "ws", 1)

        parsed = urlparse(base)
        query_params = parse_qs(parsed.query)
        path = parsed.path.rstrip("/")

        if path in ["", "/v1", "/openai", "/openai/v1"]:
            path = path + TRANSLATION_ENDPOINT_PATH
        else:
            path = path

        query_params["model"] = [TRANSLATION_MODEL]
        new_query = urlencode(query_params, doseq=True)
        url = urlunparse((parsed.scheme, parsed.netloc, path, "", new_query, ""))

        try:
            return await asyncio.wait_for(
                self._translation_model._ensure_http_session().ws_connect(
                    url=url, headers=headers
                ),
                self._translation_model._opts.conn_options.timeout,
            )
        except aiohttp.ClientError as e:
            raise APIConnectionError(
                "Translation API client connection error"
            ) from e
        except asyncio.TimeoutError as e:
            raise APIConnectionError(
                message="Translation API connection timed out"
            ) from e

    # ------------------------------------------------------------------
    # Override: Session update — translation API only accepts language + transcription
    # ------------------------------------------------------------------

    def _create_session_update_event(self) -> dict[str, Any]:
        opts = self._translation_model._opts
        target_language = self._translation_model.target_language

        session_dict: dict[str, Any] = {
            "audio": {
                "output": {
                    "language": target_language,
                },
            },
        }

        audio_input: dict[str, Any] = {}

        if opts.input_audio_noise_reduction is not None:
            audio_input["noise_reduction"] = opts.input_audio_noise_reduction.model_dump(
                by_alias=True, exclude_unset=True
            )

        if opts.input_audio_transcription is not None:
            transcription_dict = opts.input_audio_transcription.model_dump(
                by_alias=True, exclude_unset=True
            )
            model_name = transcription_dict.get("model", "")

            if model_name != "gpt-realtime-whisper":
                # gpt-realtime-whisper supports language — others don't
                transcription_dict.pop("language", None)

            # Remove fields not supported by translation transcription config
            transcription_dict.pop("prompt", None)
            transcription_dict.pop("delay", None)

            if transcription_dict:
                audio_input["transcription"] = transcription_dict

        if audio_input:
            session_dict["audio"]["input"] = audio_input

        return {
            "type": "session.update",
            "event_id": utils.shortuuid("session_update_"),
            "session": session_dict,
        }

    # ------------------------------------------------------------------
    # Override: Audio push — use session.input_audio_buffer.append
    # ------------------------------------------------------------------

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        for f in self._resample_audio(frame):
            data = f.data.tobytes()
            for nf in self._bstream.write(data):
                self.send_event({
                    "type": "session.input_audio_buffer.append",
                    "audio": base64.b64encode(nf.data).decode("utf-8"),
                })
                self._pushed_duration_s += nf.duration

    # ------------------------------------------------------------------
    # Override: Block unsupported operations
    # ------------------------------------------------------------------

    def commit_audio(self) -> None:
        pass  # not supported by translation API

    def clear_audio(self) -> None:
        pass  # not supported by translation API

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        pass  # translation API does not support tools

    async def update_instructions(self, instructions: str) -> None:
        self._instructions = instructions  # store but don't send — not supported

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        pass  # conversation.item.create/delete not supported

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
        fut.set_exception(
            llm.RealtimeError("generate_reply is not supported in translation mode")
        )
        return fut

    def interrupt(self) -> None:
        pass  # response.cancel not supported by translation API

    # ------------------------------------------------------------------
    # Override: Event routing — handle translation-specific events
    # ------------------------------------------------------------------

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        """Override _run_ws to intercept translation events before normal handling."""
        closing = False

        @utils.log_exceptions(logger=self._realtime_model._opts.__class__.__module__)  # type: ignore
        async def _send_task() -> None:
            nonlocal closing
            async for msg in self._msg_ch:
                try:
                    if hasattr(msg, "model_dump"):
                        msg = msg.model_dump(
                            by_alias=True, exclude_unset=True, exclude_defaults=False
                        )
                    self.emit("openai_client_event_queued", msg)
                    await ws_conn.send_str(json.dumps(msg))
                except Exception:
                    pass
            closing = True
            await ws_conn.close()

        async def _recv_task() -> None:
            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:
                        return
                    from livekit.agents import APIConnectionError
                    raise APIConnectionError(
                        message="Translation API connection closed unexpectedly"
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                event = json.loads(msg.data)
                self.emit("openai_server_event_received", event)

                try:
                    event_type = event.get("type", "")

                    if event_type == "session.output_audio.delta":
                        self._handle_translation_audio_delta(event)

                    elif event_type == "session.output_audio.done":
                        self._handle_translation_audio_done(event)

                    elif event_type == "session.output_transcript.delta":
                        self._handle_translation_output_transcript_delta(event)

                    elif event_type == "session.input_transcript.delta":
                        self._handle_translation_input_transcript_delta(event)

                    elif event_type == "error":
                        from openai.types.realtime import RealtimeErrorEvent
                        self._handle_error(RealtimeErrorEvent.construct(**event))

                    # session.created / session.updated — no action needed

                except Exception:
                    pass

        tasks = [
            asyncio.create_task(_recv_task(), name="translation_recv_task"),
            asyncio.create_task(_send_task(), name="translation_send_task"),
        ]

        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws_conn.close()

    # ------------------------------------------------------------------
    # Translation event handlers
    # ------------------------------------------------------------------

    def _handle_translation_audio_delta(self, event: dict[str, Any]) -> None:
        """Decode and deliver translated audio frames."""
        if self._current_generation is None:
            self._current_generation = _ResponseGeneration(
                message_ch=utils.aio.Chan(),
                function_ch=utils.aio.Chan(),
                messages={},
                _created_timestamp=time.time(),
                _done_fut=asyncio.Future(),
            )
            item_id = utils.shortuuid("translation_")
            item_gen = _MessageGeneration(
                message_id=item_id,
                text_ch=utils.aio.Chan(),
                audio_ch=utils.aio.Chan(),
                modalities=asyncio.Future(),
            )
            item_gen.modalities.set_result(["audio", "text"])
            self._current_generation.messages[item_id] = item_gen
            self._current_generation.message_ch.send_nowait(
                llm.MessageGeneration(
                    message_id=item_id,
                    text_stream=item_gen.text_ch,
                    audio_stream=item_gen.audio_ch,
                    modalities=item_gen.modalities,
                )
            )
            self.emit(
                "generation_created",
                llm.GenerationCreatedEvent(
                    message_stream=self._current_generation.message_ch,
                    function_stream=self._current_generation.function_ch,
                    user_initiated=False,
                    response_id=item_id,
                ),
            )

        item_gen = next(iter(self._current_generation.messages.values()))
        if self._current_generation._first_token_timestamp is None:
            self._current_generation._first_token_timestamp = time.time()

        data = base64.b64decode(event["delta"])
        item_gen.audio_ch.send_nowait(
            rtc.AudioFrame(
                data=data,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(data) // 2,
            )
        )

    def _handle_translation_audio_done(self, event: dict[str, Any]) -> None:
        """Audio stream done — close channels only. Item emitted by VAD or fallback timer."""
        if self._current_generation:
            for ig in self._current_generation.messages.values():
                if not ig.audio_ch.closed:
                    ig.audio_ch.close()
                if not ig.text_ch.closed:
                    ig.text_ch.close()
            self._current_generation.function_ch.close()
            self._current_generation.message_ch.close()
            with contextlib.suppress(asyncio.InvalidStateError):
                self._current_generation._done_fut.set_result(None)
            self._current_generation = None

    def _handle_translation_output_transcript_delta(self, event: dict[str, Any]) -> None:
        """Accumulate translated transcript. Emit item via VAD or fallback timer."""
        delta = event.get("delta", "")
        if not delta:
            return

        self._translation_output_transcript += delta

        # Feed into text channel for live display
        if self._current_generation:
            item_gen = next(iter(self._current_generation.messages.values()), None)
            if item_gen and not item_gen.text_ch.closed:
                item_gen.text_ch.send_nowait(delta)

        # Reset fallback timer — fires if user never speaks again
        if self._translation_output_flush_handle:
            self._translation_output_flush_handle.cancel()
        self._translation_output_flush_handle = asyncio.get_event_loop().call_later(
            self._output_flush_delay, self._flush_translation_output_now
        )

    def _handle_translation_input_transcript_delta(self, event: dict[str, Any]) -> None:
        """Accumulate source language transcript. Emit item after silence."""
        delta = event.get("delta", "")
        if not delta:
            return

        self._translation_input_transcript += delta

        # Reset input flush timer
        if self._translation_input_flush_handle:
            self._translation_input_flush_handle.cancel()
        self._translation_input_flush_handle = asyncio.get_event_loop().call_later(
            self._input_flush_delay, self._flush_translation_input
        )

    # ------------------------------------------------------------------
    # Override: VAD speech started — flush assistant item first
    # ------------------------------------------------------------------

    def _handle_input_audio_buffer_speech_started(
        self, _: InputAudioBufferSpeechStartedEvent
    ) -> None:
        # Cancel pending input timer
        if self._translation_input_flush_handle:
            self._translation_input_flush_handle.cancel()
            self._translation_input_flush_handle = None

        # Flush previous assistant turn FIRST
        self._flush_translation_output_now()

        # Clear stale input buffer
        self._translation_input_transcript = ""

        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _handle_input_audio_buffer_speech_stopped(
        self, _: InputAudioBufferSpeechStoppedEvent
    ) -> None:
        user_transcription_enabled = (
            self._translation_model._opts.input_audio_transcription is not None
        )
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(
                user_transcription_enabled=user_transcription_enabled
            ),
        )

    # ------------------------------------------------------------------
    # Flush helpers
    # ------------------------------------------------------------------

    def _flush_translation_input(self) -> None:
        """Silence after last input delta — emit ONE complete user conversation item."""
        self._translation_input_flush_handle = None
        transcript = self._translation_input_transcript.strip()
        self._translation_input_transcript = ""

        if not transcript:
            return

        item_id = utils.shortuuid("translation_user_")
        self._translation_input_item_id = item_id
        self._translation_user_item_emitted = True

        lk_item = llm.ChatMessage(
            id=item_id,
            role="user",
            content=[transcript],
        )
        self._remote_chat_ctx.insert(None, lk_item)
        self.emit(
            "remote_item_added",
            llm.RemoteItemAddedEvent(previous_item_id=None, item=lk_item),
        )
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=item_id,
                transcript=transcript,
                is_final=True,
                confidence=None,
            ),
        )

    def _flush_translation_output_now(self) -> None:
        """Emit ONE complete assistant conversation item.

        Called by:
        - VAD speech_started (user starts speaking → previous assistant turn is done)
        - Fallback timer (user stays silent after assistant finishes)
        """
        # Cancel fallback timer
        if self._translation_output_flush_handle:
            self._translation_output_flush_handle.cancel()
            self._translation_output_flush_handle = None

        transcript = self._translation_output_transcript.strip()
        self._translation_output_transcript = ""

        # Close generation channels
        if self._current_generation:
            for ig in self._current_generation.messages.values():
                if not ig.audio_ch.closed:
                    ig.audio_ch.close()
                if not ig.text_ch.closed:
                    ig.text_ch.close()
            self._current_generation.function_ch.close()
            self._current_generation.message_ch.close()
            with contextlib.suppress(asyncio.InvalidStateError):
                self._current_generation._done_fut.set_result(None)
            self._current_generation = None

        if not transcript:
            return

        item_id = utils.shortuuid("translation_asst_")
        previous_id = self._translation_input_item_id

        lk_item = llm.ChatMessage(
            id=item_id,
            role="assistant",
            content=[transcript],
        )
        self._remote_chat_ctx.insert(previous_id, lk_item)
        self._translation_output_item_id = item_id
        self._translation_user_item_emitted = False  # reset for next turn

        self.emit(
            "remote_item_added",
            llm.RemoteItemAddedEvent(
                previous_item_id=previous_id,
                item=lk_item,
            ),
        )