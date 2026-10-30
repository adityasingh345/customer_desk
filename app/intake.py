"""Intake: normalize raw input from each channel, then run the pipeline."""
import logging
import os
from pathlib import Path
from typing import Awaitable, Callable

from .alerts import notify_owner
from .classifier import classify, fallback_classification
from .models import Classification, IncomingMessage, ProcessedMessage, SupportAnswer
from .rag import default_answerer
from .router import route

log = logging.getLogger("customer_desk.intake")

Classifier = Callable[[IncomingMessage], Awaitable[Classification]]
Notifier = Callable[[ProcessedMessage], Awaitable[None]]
Answerer = Callable[[IncomingMessage], Awaitable[SupportAnswer]]

ESCALATION_REPLY = (
    "I'm not sure about that one, so I've passed your question to our team. They'll get back to you soon."
)


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


async def _answer_from_knowledge_base(processed: ProcessedMessage, answerer: Answerer) -> None:
    """Phase 4: try to answer a support question from the FAQ; otherwise hand it to a human."""
    decision = processed.decision
    try:
        answer = await answerer(processed.message)
    except Exception:
        log.exception("Support answer failed; escalating to a human")
        answer = None

    if answer is not None and answer.answerable:
        decision.reply = answer.answer
        decision.sources = answer.sources
        decision.reason += " | answered from the knowledge base"
    else:
        decision.reply = ESCALATION_REPLY
        decision.alert_owner = True  # a human must follow up: the bot does not know
        decision.reason += " | not in the knowledge base: escalated to a human"


async def handle_message(
    msg: IncomingMessage,
    classifier: Classifier | None = None,
    notifier: Notifier | None = None,
    answerer: Answerer | None = None,
) -> ProcessedMessage | None:
    """Single entry point for every message.

    1. Save the raw message FIRST, so nothing is ever lost.
    2. Classify it with AI (or a fallback if the AI fails).
    3. Route it: pick a queue, a reply, and whether to alert the owner.
    4. Support questions: try to answer from the FAQ (RAG), else escalate to a human.
    5. Save the result, put it in its queue, and alert the owner if needed.
    """
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

    processed = ProcessedMessage(message=msg, classification=classification, needs_review=needs_review)
    processed.decision = route(processed)
    decision = processed.decision

    # Phase 4. Not for urgent messages: a person should read those first, not a bot.
    if decision.queue == "support" and not needs_review and classification.urgency != "high":
        answerer = answerer or default_answerer()
        if answerer is not None:
            await _answer_from_knowledge_base(processed, answerer)

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