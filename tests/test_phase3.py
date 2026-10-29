import json
from types import SimpleNamespace

import pytest

from app import telegram_bot
from app.alerts import format_alert, notify_owner
from app.intake import handle_message, normalize_form, normalize_telegram
from app.models import Classification, IncomingMessage, ProcessedMessage
from app.router import route


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return tmp_path


def processed(category="support", urgency="medium", needs_review=False) -> ProcessedMessage:
    msg = IncomingMessage(name="Asha", message="hello", channel="web_form")
    c = Classification(category=category, urgency=urgency, summary="A summary.")
    return ProcessedMessage(message=msg, classification=c, needs_review=needs_review)


def fixed_classifier(category, urgency):
    async def classifier(m):
        return Classification(category=category, urgency=urgency, summary="A summary.")

    return classifier


def recording_notifier():
    sent = []

    async def notifier(p):
        sent.append(p)

    notifier.sent = sent
    return notifier


# --- the router: a pure decision table --------------------------------------

@pytest.mark.parametrize(
    "category,urgency,alert",
    [
        ("sales", "low", False), ("sales", "medium", False), ("sales", "high", True),
        ("support", "low", False), ("support", "high", True),
        ("billing", "medium", False), ("billing", "high", True),
    ],
)
def test_router_queue_and_alert(category, urgency, alert):
    d = route(processed(category, urgency))
    assert d.queue == category
    assert d.alert_owner is alert
    assert d.reply  # every real category gets an acknowledgement


def test_router_spam_is_silent_and_never_alerts():
    d = route(processed("spam", "high"))
    assert d.queue == "spam" and d.reply is None and d.alert_owner is False


def test_router_ai_failure_alerts_owner():
    d = route(processed("support", "medium", needs_review=True))
    assert d.alert_owner is True and d.queue == "support"
    assert "review" in d.reason


# --- the pipeline: queues and alerts ----------------------------------------

async def test_message_lands_in_its_queue_file(tmp_data):
    msg = normalize_form({"name": "Ravi", "message": "I was charged twice"})
    result = await handle_message(msg, classifier=fixed_classifier("billing", "medium"),
                                  notifier=recording_notifier())
    assert result.decision.queue == "billing"
    line = json.loads((tmp_data / "queue_billing.jsonl").read_text().strip())
    assert line["message"]["name"] == "Ravi"
    assert line["decision"]["queue"] == "billing"
    assert not (tmp_data / "queue_sales.jsonl").exists()


async def test_high_urgency_alerts_owner_exactly_once():
    notifier = recording_notifier()
    msg = normalize_form({"name": "A", "message": "refund me NOW"})
    await handle_message(msg, classifier=fixed_classifier("billing", "high"), notifier=notifier)
    assert len(notifier.sent) == 1


async def test_normal_urgency_does_not_alert():
    notifier = recording_notifier()
    msg = normalize_form({"name": "A", "message": "what are your hours"})
    await handle_message(msg, classifier=fixed_classifier("support", "low"), notifier=notifier)
    assert notifier.sent == []


async def test_spam_is_saved_but_silent(tmp_data):
    notifier = recording_notifier()
    msg = normalize_form({"name": "Bot", "message": "WIN a FREE iPhone"})
    result = await handle_message(msg, classifier=fixed_classifier("spam", "high"), notifier=notifier)
    assert result.decision.reply is None and notifier.sent == []
    assert (tmp_data / "queue_spam.jsonl").exists()  # kept for audit


async def test_ai_failure_still_routes_and_alerts():
    async def broken(m):
        raise RuntimeError("AI down")

    notifier = recording_notifier()
    msg = normalize_form({"name": "A", "message": "hello"})
    result = await handle_message(msg, classifier=broken, notifier=notifier)
    assert result.needs_review and result.decision.queue == "support"
    assert len(notifier.sent) == 1


async def test_broken_alert_does_not_break_pipeline(tmp_data):
    async def exploding_notifier(p):
        raise RuntimeError("Telegram is down")

    msg = normalize_form({"name": "A", "message": "urgent!"})
    result = await handle_message(msg, classifier=fixed_classifier("sales", "high"),
                                  notifier=exploding_notifier)
    assert result is not None
    assert (tmp_data / "classified.jsonl").exists()


# --- alerts -----------------------------------------------------------------

def test_alert_text_contains_the_essentials():
    text = format_alert(processed("billing", "high"))
    assert "BILLING" in text and "HIGH" in text and "Asha" in text and "A summary." in text


async def test_notify_owner_without_config_does_not_crash(monkeypatch):
    monkeypatch.delenv("OWNER_CHAT_ID", raising=False)
    await notify_owner(processed("billing", "high"))  # just logs a warning


# --- Telegram replies come from the router's decision -----------------------

class FakeMessage:
    def __init__(self, text):
        self.text = text
        self.replies = []

    async def reply_text(self, t):
        self.replies.append(t)


def fake_update(text="hi"):
    m = FakeMessage(text)
    return SimpleNamespace(
        effective_user=SimpleNamespace(first_name="Asha"),
        effective_message=m,
        effective_chat=SimpleNamespace(id=555),
    ), m


async def test_telegram_replies_with_router_text(monkeypatch):
    async def fake_handle(msg):
        p = processed("billing", "medium")
        p.decision = route(p)
        return p

    monkeypatch.setattr(telegram_bot, "handle_message", fake_handle)
    update, m = fake_update("charged twice")
    await telegram_bot.on_text(update, None)
    assert m.replies == [route(processed("billing", "medium")).reply]


async def test_telegram_stays_silent_for_spam(monkeypatch):
    async def fake_handle(msg):
        p = processed("spam", "low")
        p.decision = route(p)
        return p

    monkeypatch.setattr(telegram_bot, "handle_message", fake_handle)
    update, m = fake_update("WIN FREE")
    await telegram_bot.on_text(update, None)
    assert m.replies == []


async def test_telegram_default_reply_when_ai_skipped(monkeypatch):
    async def fake_handle(msg):
        return None

    monkeypatch.setattr(telegram_bot, "handle_message", fake_handle)
    update, m = fake_update()
    await telegram_bot.on_text(update, None)
    assert m.replies == [telegram_bot.DEFAULT_REPLY]