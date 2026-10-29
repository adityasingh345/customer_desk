# tell the buisness owner about importnant message 

import logging
import os 

from telegram import Bot

from .models import ProcessedMessage

log = logging.getLogger("customer_desk.alerts")


def format_alert(processed: ProcessedMessage) -> str:
    c = processed.classification
    m = processed.message
    flag = "AI FAILED - please review" if processed.needs_review else f"{c.urgency.upper()} urgency"
    
    return (
        f"[{c.category.upper()}] {flag}\n"
        f"From: {m.name} ({m.channel})\n"
        f"Summary: {c.summary}\n"
        f"Message: {m.message[:300]}"
    )

async def notify_owner(processed: ProcessedMessage) -> None:
    # sending an alert to the owners telegram bot
    
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