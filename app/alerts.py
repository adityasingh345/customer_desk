"""Phase 3: tell the business owner about important messages, on Telegram."""
import logging
import os

from telegram import Bot

from .models import ProcessedMessage

log = logging.getLogger("customer_desk.alerts")


def format_alert(processed: ProcessedMessage) -> str:
    c = processed.classification
    m = processed.message
    flag = "AI FAILED - please review" if processed.needs_review else f"{c.urgency.upper()} urgency"
    lines = [
        f"[{c.category.upper()}] {flag}",
        f"From: {m.name} ({m.channel})",
        f"Summary: {c.summary}",
        f"Message: {m.message[:300]}",
    ]
    if processed.decision:  # Phase 4: say WHY you are being alerted
        lines.append(f"Why: {processed.decision.reason}")
    return "\n".join(lines)


async def notify_owner(processed: ProcessedMessage) -> None:
    """Send an alert to the owner's Telegram chat.

    Never raises: a failed alert must not break message handling.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    owner_chat_id = os.getenv("OWNER_CHAT_ID", "").strip()
    text = format_alert(processed)

    if not token or not owner_chat_id:
        log.warning("OWNER_CHAT_ID or TELEGRAM_BOT_TOKEN not set; alert not sent:\n%s", text)
        return
    try:
        async with Bot(token) as bot:
            await bot.send_message(chat_id=owner_chat_id, text=text)
    except Exception:
        log.exception("Could not send owner alert")