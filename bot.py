import asyncio
import os
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx
import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberFormat, region_code_for_number
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

# Hinweis: Für lokale Checks auf Merge-Konfliktmarker kann
# scripts/check_no_merge_conflicts.py ausgeführt werden.


DEFAULT_SYSTEM_PROMPT = """
Du bist der deutschsprachige Voice-Agent unserer Agentur. Dein Ziel ist es, den
Terminwunsch des Anrufers aufzunehmen und die Details per SMS an unser Team zu
schicken, damit wir den Rückruf manuell organisieren können.

Vorgehen:
1. Begrüße freundlich und erkläre, dass du Termine vormerken und weiterleiten kannst.
2. Frage ausschließlich nach dem Namen sowie nach gewünschtem Datum und Uhrzeit
   (volle Stunde, Format HH:MM). Die Telefonnummer stammt automatisch aus dem System.
3. Wiederhole die Daten kurz, lass sie bestätigen und kündige an, dass du sie per SMS
   weiterleitest.
4. Verwende anschließend die Funktion `create_booking`, um die SMS mit Name, Datum,
   Uhrzeit und Telefonnummer auszulösen.
5. Bestätige knapp, dass die Anfrage übermittelt wurde und sich jemand meldet. Mache
   keine Zusagen zu einer fixen Buchung.

Sprich präzise und auf Deutsch.
""".strip()

DEFAULT_NOTIFICATION_RECIPIENT = "+4915752651227"
BOOKING_TIMEZONE = "Europe/Berlin"
INITIAL_GREETING_INSTRUCTION = (
    "Begrüße den Anrufer freundlich, erkläre kurz, dass du bei "
    "Terminbuchungen hilfst, und frage direkt nach seinem Namen sowie dem "
    "gewünschten Datum/Uhrzeit (volle Stunde)."
)
CALLER_PHONE_INSTRUCTION_TEMPLATE = (
    "Der aktuelle Anrufer wurde über Twilio identifiziert. Verwende für "
    "die SMS automatisch die Nummer {phone} und frage nicht erneut danach."
)
BOOKING_FLOW_REMINDER = (
    "Sammle nur Name sowie Datum/Uhrzeit. Verwende anschließend `create_booking`, "
    "um eine SMS mit den Daten (inklusive der automatisch erkannten Telefonnummer) "
    "an das Team zu schicken und bestätige danach die Weiterleitung."
)


def build_booking_tools_schema() -> ToolsSchema:
    """Create the tool definitions that Gemini should be aware of."""

    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="create_booking",
                description=(
                    "Löst eine SMS an das interne Team aus, sobald Name, Datum"
                    " und Uhrzeit bestätigt wurden."
                ),
                properties={
                    "name": {
                        "type": "string",
                        "description": "Vollständiger Name des Kunden.",
                    },
                    "phone": {
                        "type": "string",
                        "description": (
                            "Optional; wird normalerweise automatisch aus dem"
                            " Anruf übernommen. Frage nicht aktiv danach."
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
                    "notes": {
                        "type": "string",
                        "description": (
                            "Optionale Zusatzinformationen oder Wünsche des"
                            " Anrufers."
                        ),
                    },
                },
                required=["name", "date", "time"],
            ),
        ]
    )


class BookingAPI:
    """Validate slot details and trigger SMS notifications for manual follow-up."""

    def __init__(
        self,
        timezone: str = BOOKING_TIMEZONE,
        *,
        caller_phone: Optional[str] = None,
        twilio_from_number: Optional[str] = None,
        notification_recipient: Optional[str] = None,
    ) -> None:
        self.timezone = timezone
        self._timeout = httpx.Timeout(10.0)
        self._tzinfo = ZoneInfo(self.timezone)

        default_region_env = os.getenv('BOOKING_DEFAULT_REGION', '').strip().upper()
        self.default_region = default_region_env or 'DE'

        self.caller_phone = self._normalize_phone_number(caller_phone, None)
        if caller_phone and not self.caller_phone:
            logger.warning(
                'Anrufernummer konnte nicht normalisiert werden: {}', caller_phone
            )

        caller_region = self._detect_region(self.caller_phone)
        if caller_region:
            self.default_region = caller_region

        from_number = twilio_from_number.strip() if twilio_from_number else None
        if not from_number:
            fallback_from = os.getenv('TWILIO_SMS_FROM_NUMBER')
            from_number = fallback_from.strip() if fallback_from else None

        self.twilio_from_number = self._normalize_phone_number(from_number, None)
        if from_number and not self.twilio_from_number:
            logger.warning(
                'Twilio-Absendernummer konnte nicht normalisiert werden: {}',
                from_number,
            )

        configured_recipient = (
            notification_recipient
            or os.getenv('BOOKING_NOTIFICATION_SMS_RECIPIENT')
            or DEFAULT_NOTIFICATION_RECIPIENT
        )
        normalized_recipient = self._normalize_phone_number(
            configured_recipient, self.default_region
        )
        if configured_recipient and not normalized_recipient:
            logger.warning(
                'SMS-Empfänger konnte nicht normalisiert werden: {}',
                configured_recipient,
            )
        self.notification_recipient = normalized_recipient

        self.twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')

    def _normalize_phone_number(
        self, phone: Optional[str], region_hint: Optional[str]
    ) -> Optional[str]:
        if not phone:
            return None
        candidate = str(phone).strip()
        if not candidate:
            return None

        attempts: List[str] = [candidate]
        digits = ''.join(ch for ch in candidate if ch.isdigit() or ch == '+')
        if digits and digits not in attempts:
            attempts.append(digits)
        if digits.startswith('00'):
            alt = '+' + digits[2:]
            if alt not in attempts:
                attempts.append(alt)

        region_order: List[Optional[str]] = []
        if region_hint:
            region_order.append(region_hint)
        if self.default_region and self.default_region not in region_order:
            region_order.append(self.default_region)
        region_order.append(None)

        for attempt in attempts:
            if not attempt:
                continue
            for region in region_order:
                try:
                    parsed = phonenumbers.parse(attempt, region)
                except NumberParseException:
                    continue
                if not phonenumbers.is_possible_number(parsed):
                    continue
                try:
                    formatted = phonenumbers.format_number(
                        parsed, PhoneNumberFormat.E164
                    )
                except Exception:
                    continue
                if formatted:
                    return formatted

        if digits.startswith('+') and len(digits) >= 8:
            return digits
        return None

    @staticmethod
    def _detect_region(phone: Optional[str]) -> Optional[str]:
        if not phone:
            return None
        try:
            parsed = phonenumbers.parse(phone)
        except NumberParseException:
            return None
        region = region_code_for_number(parsed)
        return region or None

    def _now(self) -> datetime:
        return datetime.now(tz=self._tzinfo)

    def _slot_datetime(self, slot_date: date, time_str: str) -> datetime:
        hour, minute = [int(part) for part in time_str.split(':', 1)]
        return datetime(
            slot_date.year,
            slot_date.month,
            slot_date.day,
            hour,
            minute,
            tzinfo=self._tzinfo,
        )

    def _is_slot_in_past(self, slot_date: date, time_str: str) -> bool:
        return self._slot_datetime(slot_date, time_str) <= self._now()

    @staticmethod
    def _allowed_start_hours(weekday: int) -> List[int]:
        if weekday < 5:
            return list(range(7, 15))
        if weekday == 5:
            return list(range(7, 13))
        return []

    def _validate_slot(
        self, date_str: str, time_str: str
    ) -> tuple[bool, Optional[str], Optional[date], Optional[str]]:
        try:
            date_obj = datetime.strptime(date_str.strip(), '%Y-%m-%d').date()
        except ValueError:
            return False, 'Datum muss im Format YYYY-MM-DD vorliegen.', None, None

        normalized_time = None
        minute = 0
        for fmt in ('%H:%M', '%H:%M:%S'):
            try:
                parsed_time = datetime.strptime(time_str.strip(), fmt).time()
                normalized_time = f"{parsed_time.hour:02d}:00"
                minute = parsed_time.minute
                break
            except ValueError:
                continue
        if normalized_time is None:
            return (
                False,
                'Zeit muss im Format HH:MM (z. B. 09:00) angegeben werden.',
                date_obj,
                None,
            )
        if minute != 0:
            return (
                False,
                'Termine sind nur zur vollen Stunde möglich.',
                date_obj,
                None,
            )

        allowed_hours = self._allowed_start_hours(date_obj.weekday())
        if not allowed_hours:
            return False, 'An diesem Tag werden keine Termine angeboten.', date_obj, None
        hour = int(normalized_time.split(':', 1)[0])
        if hour not in allowed_hours:
            first = f"{allowed_hours[0]:02d}:00" if allowed_hours else ''
            last = f"{allowed_hours[-1]:02d}:00" if allowed_hours else ''
            return (
                False,
                f'Startzeiten müssen zwischen {first} und {last} liegen.',
                date_obj,
                None,
            )

        return True, None, date_obj, normalized_time

    def _build_notification_message(
        self,
        *,
        name: str,
        date_value: str,
        time_value: str,
        phone: Optional[str],
        notes: Optional[str],
    ) -> str:
        lines = [
            'Neue Termin-Anfrage (manuelle Buchung):',
            f'Name: {name}',
            f'Datum/Zeit: {date_value} {time_value} ({self.timezone})',
        ]
        if phone:
            lines.append(f'Telefon: {phone}')
        if notes:
            lines.append(f'Notizen: {notes}')
        return "\n".join(lines)

    async def _send_sms_notification(self, body: str) -> bool:
        if not self.notification_recipient:
            logger.warning(
                'SMS-Benachrichtigung übersprungen: Kein Empfänger konfiguriert.'
            )
            return False
        if not self.twilio_account_sid or not self.twilio_auth_token:
            logger.warning(
                'SMS-Benachrichtigung übersprungen: Twilio-Zugangsdaten fehlen.'
            )
            return False
        if not self.twilio_from_number:
            logger.warning(
                'SMS-Benachrichtigung übersprungen: Absendernummer unbekannt.'
            )
            return False

        url = (
            f'https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/Messages.json'
        )
        data = {
            'To': self.notification_recipient,
            'From': self.twilio_from_number,
            'Body': body,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    url,
                    data=data,
                    auth=(self.twilio_account_sid, self.twilio_auth_token),
                )
        except httpx.RequestError as exc:
            logger.exception('SMS-Benachrichtigung fehlgeschlagen: {}', exc)
            return False

        if response.status_code >= 300:
            logger.error(
                'SMS-Benachrichtigung fehlgeschlagen: {} - {}',
                response.status_code,
                response.text,
            )
            return False

        logger.info(
            'SMS-Benachrichtigung versendet (Status {}).', response.status_code
        )
        return True

    async def handle_create_booking(self, params: FunctionCallParams) -> None:
        args = dict(params.arguments or {})
        name = str(args.get('name') or '').strip()
        date_raw = str(args.get('date') or '').strip()
        time_raw = str(args.get('time') or '').strip()
        notes_raw = str(args.get('notes') or '').strip()
        provided_phone = str(args.get('phone') or '').strip()

        errors: List[str] = []
        if not name:
            errors.append('name: Bitte den vollständigen Namen erfassen.')
        if not date_raw:
            errors.append('date: Bitte ein Datum im Format YYYY-MM-DD angeben.')
        if not time_raw:
            errors.append('time: Bitte eine Startzeit im Format HH:MM angeben.')

        normalized_phone = self._normalize_phone_number(
            provided_phone, self.default_region
        )
        phone_value = normalized_phone or self.caller_phone
        if not phone_value:
            errors.append(
                'phone: Die Anrufernummer konnte nicht ermittelt werden.'
            )

        slot_date = None
        slot_time = None
        if not errors and date_raw and time_raw:
            valid, slot_error, slot_date, slot_time = self._validate_slot(
                date_raw, time_raw
            )
            if not valid:
                errors.append(
                    f'slot: {slot_error}' if slot_error else 'slot: Ungültig.'
                )
            elif slot_date and slot_time and self._is_slot_in_past(slot_date, slot_time):
                errors.append('slot: Der gewünschte Termin liegt in der Vergangenheit.')

        if errors:
            await params.result_callback(
                {
                    'success': False,
                    'error': 'validation_error',
                    'messages': errors,
                }
            )
            return

        assert slot_date is not None and slot_time is not None
        logger.info(
            'Versende Buchungs-SMS für {} am {} {}',
            name,
            slot_date.isoformat(),
            slot_time,
        )

        message = self._build_notification_message(
            name=name,
            date_value=slot_date.isoformat(),
            time_value=slot_time,
            phone=phone_value,
            notes=notes_raw or None,
        )
        sms_sent = await self._send_sms_notification(message)

        payload: Dict[str, Any] = {
            'success': sms_sent,
            'message': (
                'SMS mit den Termindetails wurde versendet.'
                if sms_sent
                else 'SMS konnte nicht versendet werden.'
            ),
            'request': {
                'name': name,
                'date': slot_date.isoformat(),
                'time': slot_time,
                'phone': phone_value,
            },
            'timezone': self.timezone,
        }
        if notes_raw:
            payload['request']['notes'] = notes_raw
        if not sms_sent:
            payload['error'] = 'sms_failed'
        if self.notification_recipient:
            payload['notificationRecipient'] = self.notification_recipient

        await params.result_callback(payload)


async def run_bot(
    transport: BaseTransport,
    handle_sigint: bool,
    caller_phone: Optional[str] = None,
    twilio_number: Optional[str] = None,
):
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

    sms_recipient_env = os.getenv("BOOKING_NOTIFICATION_SMS_RECIPIENT")
    sms_recipient = (sms_recipient_env or DEFAULT_NOTIFICATION_RECIPIENT).strip()
    booking_client = BookingAPI(
        caller_phone=caller_phone,
        twilio_from_number=twilio_number,
        notification_recipient=sms_recipient or None,
    )

    llm.register_function("create_booking", booking_client.handle_create_booking)

    logger.info(
        (
            "SMS-Modus aktiv (recipient_raw: {}, caller_raw: {}, caller_normalized: {}, "
            "twilio_raw: {}, twilio_normalized: {})"
        ),
        sms_recipient or DEFAULT_NOTIFICATION_RECIPIENT,
        caller_phone,
        booking_client.caller_phone,
        twilio_number,
        booking_client.twilio_from_number,
    )

    initial_messages = [
        {"role": "user", "content": INITIAL_GREETING_INSTRUCTION},
        {"role": "user", "content": BOOKING_FLOW_REMINDER},
    ]
    prompt_phone = booking_client.caller_phone or caller_phone
    if prompt_phone:
        initial_messages.append(
            {
                "role": "user",
                "content": CALLER_PHONE_INSTRUCTION_TEMPLATE.format(
                    phone=prompt_phone
                ),
            }
        )

    context = OpenAILLMContext(initial_messages)
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
    caller_phone: Optional[str] = None
    twilio_number: Optional[str] = None
    if isinstance(call_data, Mapping):
        body = call_data.get("body")
        if isinstance(body, Mapping):
            caller_raw = str(body.get("from") or "").strip()
            twilio_raw = str(body.get("to") or "").strip()
            caller_phone = caller_raw or None
            twilio_number = twilio_raw or None

    logger.info(
        "Detected transport: {} (caller: {}, twilio: {})",
        transport_type,
        caller_phone,
        twilio_number,
    )

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

    await run_bot(
        transport,
        runner_args.handle_sigint,
        caller_phone=caller_phone,
        twilio_number=twilio_number,
    )


