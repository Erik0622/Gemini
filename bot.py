import asyncio
import os
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.gemini_multimodal_live import GeminiMultimodalLiveLLMService
from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalModalities,
    InputParams,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

load_dotenv(override=True)


DEFAULT_SYSTEM_PROMPT = (
    "Du bist der deutschsprachige Voice-Agent unserer Agentur."
    " Du hilfst aktiv bei Terminbuchungen und nutzt dafür die bereitgestellten"
    " Tools. Prüfe Verfügbarkeiten mit `get_bookings` und lege Termine mit"
    " `create_booking` an. Arbeite ausschließlich mit Zeiten zur vollen Stunde"
    " (Format HH:MM) und in der Zeitzone Europe/Berlin. Halte dich an die"
    " Öffnungszeiten: Montag–Freitag Startzeiten 07:00–14:00, Samstag"
    " 07:00–12:00, Sonntag geschlossen. Erfrage stets Name, E-Mail-Adresse,"
    " gewünschte Kontaktart (phone/zoom/teams) sowie – bei Telefonterminen –"
    " eine Telefonnummer, bevor du buchst. Bestätige Details, prüfe Slots und"
    " mache Alternativvorschläge, falls gewünschte Termine nicht verfügbar"
    " sind."
)

DEFAULT_BOOKING_BASE_URL = "https://agentur.fly.dev"
BOOKING_TIMEZONE = "Europe/Berlin"
ALLOWED_MEETING_TYPES = {"phone", "zoom", "teams"}
MEETING_TYPE_ALIASES = {
    "telefon": "phone",
    "telefonat": "phone",
    "anruf": "phone",
    "call": "phone",
    "microsoft teams": "teams",
    "teams-call": "teams",
    "microsoft-teams": "teams",
}


def build_booking_tools_schema() -> ToolsSchema:
    """Create the tool definitions that Gemini should be aware of."""

    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="get_bookings",
                description=(
                    "Liest bestehende Buchungen in einem Zeitraum aus, um"
                    " Verfügbarkeiten zu prüfen. Wenn kein to_date angegeben"
                    " ist, verwende denselben Tag wie from_date."
                ),
                properties={
                    "from_date": {
                        "type": "string",
                        "description": (
                            "Startdatum (inklusive) im Format YYYY-MM-DD in"
                            f" {BOOKING_TIMEZONE}."
                        ),
                        "pattern": r"^\\d{4}-\\d{2}-\\d{2}$",
                    },
                    "to_date": {
                        "type": "string",
                        "description": (
                            "Optionales Enddatum (inklusive) im Format"
                            " YYYY-MM-DD."
                        ),
                        "pattern": r"^\\d{4}-\\d{2}-\\d{2}$",
                    },
                },
                required=["from_date"],
            ),
            FunctionSchema(
                name="create_booking",
                description=(
                    "Legt eine neue Buchung an. Nutze dies erst, wenn alle"
                    " relevanten Daten bestätigt wurden."
                ),
                properties={
                    "name": {
                        "type": "string",
                        "description": "Vollständiger Name des Kunden.",
                    },
                    "email": {
                        "type": "string",
                        "description": "Gültige E-Mail-Adresse des Kunden.",
                        "format": "email",
                    },
                    "meetingType": {
                        "type": "string",
                        "description": (
                            "Kontaktart: phone für Telefon, zoom oder teams"
                            " für Online-Meetings."
                        ),
                        "enum": sorted(ALLOWED_MEETING_TYPES),
                    },
                    "phone": {
                        "type": "string",
                        "description": (
                            "Pflicht bei meetingType=phone. Internationale"
                            " Schreibweise bevorzugt."
                        ),
                    },
                    "date": {
                        "type": "string",
                        "description": (
                            "Datum im Format YYYY-MM-DD (Europe/Berlin)."
                        ),
                        "pattern": r"^\\d{4}-\\d{2}-\\d{2}$",
                    },
                    "time": {
                        "type": "string",
                        "description": (
                            "Startzeit zur vollen Stunde im Format HH:MM,"
                            " z. B. 09:00."
                        ),
                        "pattern": r"^\\d{2}:\\d{2}$",
                    },
                },
                required=["name", "email", "meetingType", "date", "time"],
            ),
        ]
    )


class BookingAPI:
    """Client for the Fly.io booking API used by the voice agent."""

    def __init__(
        self,
        base_url: str,
        agent_secret: Optional[str] = None,
        timezone: str = BOOKING_TIMEZONE,
    ) -> None:
        self.base_url = base_url.rstrip("/") or DEFAULT_BOOKING_BASE_URL
        self.agent_secret = agent_secret
        self.timezone = timezone
        self._timeout = httpx.Timeout(10.0)

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self.agent_secret:
            headers["x-agent-secret"] = self.agent_secret
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        headers = self._build_headers()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    headers=headers,
                )
        except httpx.RequestError as exc:
            logger.exception("HTTP request to booking API failed: {}", exc)
            raise
        return response

    @staticmethod
    def _extract_error(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            text = response.text.strip()
            return text or f"HTTP {response.status_code}"

        if isinstance(data, Mapping):
            error = data.get("error") or data.get("message")
            if isinstance(error, str):
                translations = {
                    "slot_taken": "Der Termin ist bereits belegt.",
                    "invalid_payload": "Die übermittelten Daten wurden vom Server abgelehnt.",
                    "unauthorized": "Authentifizierung fehlgeschlagen (x-agent-secret prüfen).",
                }
                return translations.get(error, error)
            return str(data)
        return str(data)

    @staticmethod
    def _canonicalize_time_string(value: str) -> str:
        cleaned = value.strip()
        for fmt in ("%H:%M", "%H:%M:%S"):
            try:
                parsed = datetime.strptime(cleaned, fmt).time()
                return f"{parsed.hour:02d}:{parsed.minute:02d}"
            except ValueError:
                continue
        return cleaned

    @staticmethod
    def _allowed_start_hours(weekday: int) -> List[int]:
        if weekday < 5:
            return list(range(7, 15))
        if weekday == 5:
            return list(range(7, 13))
        return []

    async def _get_bookings(
        self, from_date: str, to_date: str
    ) -> tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        try:
            response = await self._request(
                "GET",
                "/api/bookings",
                params={"from": from_date, "to": to_date},
            )
        except httpx.RequestError as exc:
            return None, {
                "success": False,
                "error": "network_error",
                "message": f"Verbindung zur Booking-API fehlgeschlagen: {exc}",
            }

        if response.status_code != 200:
            return None, {
                "success": False,
                "error": "http_error",
                "status_code": response.status_code,
                "message": self._extract_error(response),
            }

        try:
            data = response.json()
        except ValueError:
            return None, {
                "success": False,
                "error": "invalid_response",
                "message": "Die Booking-API lieferte keine gültige JSON-Antwort.",
            }

        if data is None:
            bookings: List[Dict[str, Any]] = []
        elif isinstance(data, list):
            bookings = data
        else:
            bookings = [data]

        return bookings, None

    def _available_slots(
        self,
        start_date: date,
        end_date: date,
        bookings: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        occupied: Dict[str, set[str]] = {}
        for entry in bookings:
            if not isinstance(entry, Mapping):
                continue
            date_value = entry.get("date")
            time_value = entry.get("time")
            if not isinstance(date_value, str) or not isinstance(time_value, str):
                continue
            canonical_time = self._canonicalize_time_string(time_value)
            occupied.setdefault(date_value, set()).add(canonical_time)

        result: List[Dict[str, Any]] = []
        current = start_date
        while current <= end_date:
            hours = self._allowed_start_hours(current.weekday())
            if hours:
                date_key = current.isoformat()
                busy = occupied.get(date_key, set())
                available_times = [
                    f"{hour:02d}:00" for hour in hours if f"{hour:02d}:00" not in busy
                ]
                if available_times:
                    result.append(
                        {
                            "date": date_key,
                            "times": available_times,
                            "timezone": self.timezone,
                        }
                    )
            current += timedelta(days=1)
        return result

    def _validate_slot(
        self, date_str: str, time_str: str
    ) -> tuple[bool, Optional[str], Optional[date], Optional[str]]:
        try:
            date_obj = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
        except ValueError:
            return False, "Datum muss im Format YYYY-MM-DD vorliegen.", None, None

        normalized_time = None
        minute = 0
        for fmt in ("%H:%M", "%H:%M:%S"):
            try:
                parsed_time = datetime.strptime(time_str.strip(), fmt).time()
                normalized_time = f"{parsed_time.hour:02d}:00"
                minute = parsed_time.minute
                break
            except ValueError:
                continue
        if normalized_time is None:
            return False, "Zeit muss im Format HH:MM (z. B. 09:00) angegeben werden.", date_obj, None
        if minute != 0:
            return False, "Termine sind nur zur vollen Stunde möglich.", date_obj, None

        allowed_hours = self._allowed_start_hours(date_obj.weekday())
        if not allowed_hours:
            return False, "An diesem Tag werden keine Termine angeboten.", date_obj, None
        hour = int(normalized_time.split(":", 1)[0])
        if hour not in allowed_hours:
            first = f"{allowed_hours[0]:02d}:00" if allowed_hours else ""
            last = f"{allowed_hours[-1]:02d}:00" if allowed_hours else ""
            return (
                False,
                f"Startzeiten müssen zwischen {first} und {last} liegen.",
                date_obj,
                None,
            )

        return True, None, date_obj, normalized_time

    @staticmethod
    def _normalize_meeting_type(meeting_type: str) -> str:
        normalized = meeting_type.strip().lower()
        return MEETING_TYPE_ALIASES.get(normalized, normalized)

    async def handle_get_bookings(self, params: FunctionCallParams) -> None:
        args = dict(params.arguments or {})
        from_date_raw = str(args.get("from_date") or args.get("from") or "").strip()
        to_date_raw = str(args.get("to_date") or args.get("to") or "").strip()

        if not from_date_raw:
            await params.result_callback(
                {
                    "success": False,
                    "error": "validation_error",
                    "messages": ["from_date ist erforderlich."],
                }
            )
            return

        if not to_date_raw:
            to_date_raw = from_date_raw

        errors: List[str] = []
        try:
            start_date = datetime.strptime(from_date_raw, "%Y-%m-%d").date()
        except ValueError:
            errors.append("from_date muss im Format YYYY-MM-DD vorliegen.")
            start_date = None  # type: ignore

        try:
            end_date = datetime.strptime(to_date_raw, "%Y-%m-%d").date()
        except ValueError:
            errors.append("to_date muss im Format YYYY-MM-DD vorliegen.")
            end_date = None  # type: ignore

        if not errors and start_date and end_date and start_date > end_date:
            errors.append("from_date darf nicht nach to_date liegen.")

        if errors:
            await params.result_callback(
                {
                    "success": False,
                    "error": "validation_error",
                    "messages": errors,
                    "requested": {"from": from_date_raw, "to": to_date_raw},
                }
            )
            return

        logger.info(
            "Calling get_bookings for range {} – {}", from_date_raw, to_date_raw
        )
        bookings, error = await self._get_bookings(from_date_raw, to_date_raw)
        if error:
            error_payload = dict(error)
            error_payload["requested"] = {"from": from_date_raw, "to": to_date_raw}
            await params.result_callback(error_payload)
            return

        assert start_date is not None and end_date is not None
        available_slots = self._available_slots(start_date, end_date, bookings or [])
        await params.result_callback(
            {
                "success": True,
                "from": from_date_raw,
                "to": to_date_raw,
                "bookings": bookings,
                "availableSlots": available_slots,
                "meetingTypes": sorted(ALLOWED_MEETING_TYPES),
                "timezone": self.timezone,
            }
        )

    async def handle_create_booking(self, params: FunctionCallParams) -> None:
        args = dict(params.arguments or {})
        name = str(args.get("name") or "").strip()
        email = str(args.get("email") or "").strip()
        meeting_type_raw = str(args.get("meetingType") or "").strip()
        meeting_type = self._normalize_meeting_type(meeting_type_raw)
        phone = str(args.get("phone") or "").strip()
        date_raw = str(args.get("date") or "").strip()
        time_raw = str(args.get("time") or "").strip()

        errors: List[str] = []
        if not name:
            errors.append("name: Bitte den vollständigen Namen erfassen.")
        if not email or "@" not in email:
            errors.append("email: Bitte eine gültige E-Mail-Adresse angeben.")
        if not meeting_type:
            errors.append(
                "meetingType: Bitte phone, zoom oder teams als Kontaktart wählen."
            )
        elif meeting_type not in ALLOWED_MEETING_TYPES:
            errors.append(
                "meetingType: Ungültig. Erlaubt sind 'phone', 'zoom' oder 'teams'."
            )
        if meeting_type == "phone" and not phone:
            errors.append("phone: Für Telefontermine ist eine Telefonnummer nötig.")
        if not date_raw:
            errors.append("date: Bitte ein Datum im Format YYYY-MM-DD angeben.")
        if not time_raw:
            errors.append("time: Bitte eine Startzeit im Format HH:MM angeben.")

        slot_date = None
        slot_time = None
        if not errors and date_raw and time_raw:
            valid, slot_error, slot_date, slot_time = self._validate_slot(
                date_raw, time_raw
            )
            if not valid:
                errors.append(f"slot: {slot_error}" if slot_error else "slot: Ungültig.")

        if errors:
            await params.result_callback(
                {
                    "success": False,
                    "error": "validation_error",
                    "messages": errors,
                }
            )
            return

        assert slot_date is not None and slot_time is not None
        booking_payload: Dict[str, Any] = {
            "name": name,
            "email": email,
            "meetingType": meeting_type,
            "date": slot_date.isoformat(),
            "time": slot_time,
        }
        if phone:
            booking_payload["phone"] = phone

        logger.info(
            "Calling create_booking for {} on {} {}",
            meeting_type,
            booking_payload["date"],
            booking_payload["time"],
        )

        try:
            response = await self._request(
                "POST",
                "/api/bookings",
                json=booking_payload,
            )
        except httpx.RequestError as exc:
            await params.result_callback(
                {
                    "success": False,
                    "error": "network_error",
                    "message": f"Verbindung zur Booking-API fehlgeschlagen: {exc}",
                    "requested": booking_payload,
                }
            )
            return

        if response.status_code == 200:
            try:
                response_data = response.json()
            except ValueError:
                response_data = {"success": True}

            await params.result_callback(
                {
                    "success": True,
                    "bookingId": response_data.get("id"),
                    "message": response_data.get("message")
                    or "Termin wurde erfolgreich gebucht.",
                    "request": booking_payload,
                    "response": response_data,
                    "timezone": self.timezone,
                }
            )
            return

        if response.status_code == 409:
            message = self._extract_error(response)
            same_day = slot_date.isoformat()
            bookings, error = await self._get_bookings(same_day, same_day)
            alternatives: List[Dict[str, Any]] = []
            if bookings is not None:
                alternatives = self._available_slots(slot_date, slot_date, bookings)

            payload: Dict[str, Any] = {
                "success": False,
                "error": "slot_taken",
                "message": message,
                "requested": booking_payload,
                "timezone": self.timezone,
            }
            if alternatives:
                payload["availableSlots"] = alternatives
            if error:
                payload["availabilityError"] = error
            await params.result_callback(payload)
            return

        # Other HTTP errors
        await params.result_callback(
            {
                "success": False,
                "error": "http_error",
                "status_code": response.status_code,
                "message": self._extract_error(response),
                "requested": booking_payload,
            }
        )

async def run_bot(transport: BaseTransport, handle_sigint: bool):
    system_prompt = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    tools_schema = build_booking_tools_schema()

    llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="models/gemini-2.5-flash-preview-native-audio-dialog",
        system_instruction=system_prompt,
        tools=tools_schema,
        params=InputParams(
            modalities=GeminiMultimodalModalities.AUDIO,
            language=Language.DE_DE,
        ),
        transcribe_user_audio=True,
        transcribe_model_audio=True,
    )

    booking_base_url = os.getenv("BOOKINGS_BASE_URL", DEFAULT_BOOKING_BASE_URL)
    agent_secret = os.getenv("AGENT_SECRET")
    booking_client = BookingAPI(base_url=booking_base_url, agent_secret=agent_secret)

    llm.register_function("get_bookings", booking_client.handle_get_bookings)
    llm.register_function("create_booking", booking_client.handle_create_booking)

    logger.info(
        "Booking API configured at {} (secret configured: {})",
        booking_client.base_url,
        bool(agent_secret),
    )

    context = OpenAILLMContext(
        [
            {
                "role": "user",
                "content": (
                    "Begrüße den Anrufer freundlich, erkläre kurz, dass du bei"
                    " Terminbuchungen helfen kannst, und frage nach dem Anliegen."
                ),
            }
        ]
    )
    context.set_tools(tools_schema)
    context.set_tool_choice("auto")
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,  # LLM (Audio)
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint, force_gc=True)
    await runner.run(task)


async def bot(runner_args: RunnerArguments, testing: bool | None = False):
    # Robust: Twilio-Initialframes kommen gelegentlich fragmentiert; kurzer Retry hilft beim allerersten Call
    for _ in range(3):
        try:
            transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
            break
        except Exception:
            await asyncio.sleep(0.2)
    else:
        # Wenn Parsing wiederholt scheitert, breche sauber ab
        return
    logger.info(f"Detected transport: {transport_type}")

    serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    await run_bot(transport, runner_args.handle_sigint)


