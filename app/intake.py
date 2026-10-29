"""Intake: normalize raw input from each channel, then run the pipeline."""
import logging
import os
from pathlib import Path
from typing import Awaitable, Callable

from .alerts import notify_owner
from .classifier import classify, fallback_classification
from .models import Classification, IncomingMessage, ProcessedMessage
from .router import route

log = logging.getLogger("customer_desk.intake")

Classifier = Callable[[IncomingMessage], Awaitable[Classification]]
Notifier = Callable[[ProcessedMessage], Awaitable[None]]

def normalize_telegram(first_name: str | None, text: str, chat_id: int | str) -> IncomingMessage:
    return IncomingMessage(
        name=first_name or "Unknown",
        message=text,
        channel="telegram",
        chat_id=str(chat_id),
    )


def normalize_form(payload: dict) -> IncomingMessage:
    return IncomingMessage(
        name=payload.get("name", ""),
        message=payload.get("message", ""),
        channel="web_form",
    )


def _data_file(name: str) -> Path:
    # Read the env var at call time so tests can point it elsewhere.
    folder = Path(os.getenv("DATA_DIR", "data"))
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def _append(name: str, line: str) -> None:
    with _data_file(name).open("a", encoding="utf-8") as f:
        f.write(line + "\n")


async def handle_message(msg: IncomingMessage, classifier: Classifier | None = None, notifier: Notifier | None = None) -> ProcessedMessage | None:
    
    log.info("[%s] %s: %s", msg.channel, msg.name, msg.message)
    _append("messages.jsonl", msg.model_dump_json())

    if classifier is None:
        if not os.getenv("GEMINI_API_KEY"):
            log.warning("GEMINI_API_KEY not set: skipping classification")
            return None
        classifier = classify

    needs_review = False
    try:
        classification = await classifier(msg)
    except Exception:
        log.exception("Classification failed; using fallback and flagging for review")
        classification = fallback_classification()
        needs_review = True
        
    # after classification, route(processed) produces the decision and attaches it to the message 
    # the result is saved to classified.jsonl and also to a queue file named after the destinations 

    processed = ProcessedMessage(message=msg, classification=classification, needs_review=needs_review)
    
    processed.decision = route(processed)
    decision = processed.decision
    
    _append("classified.jsonl", processed.model_dump_json())
    _append(f"queue_{decision.queue}.jsonl", processed.model_dump_json())
    log.info(
        "routed -> %s%s | %s | %s",
        decision.queue,
        " [ALERT]" if decision.alert_owner else "",
        decision.reason,
        classification.summary,
    )
    if decision.alert_owner:
        try:
            await (notifier or notify_owner)(processed)
        except Exception:
            log.exception("Alert failed; continuing")
    return processed