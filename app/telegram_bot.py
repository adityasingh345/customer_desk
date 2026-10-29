# telegram using pooling : no public url tunnel needed

#what we did here ---------------------------- this is the equivalent of n8n's trigger: we ahve used pooling: the library keeps askign telegram is any new messages. thats why we dont need a tunnel or public url, ehich was the biggest problem in n8n's plan 
import logging 
from pydantic import ValidationError
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .intake import handle_message, normalize_telegram

log = logging.getLogger("customer_desk.telegram")

DEFAULT_REPLY = "Got it. Your message has been received."

#on_start runs when someone sends /start and replies with greeting
async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text("Hi! Send us your question and we'll get back to you.")

# on_text runs for every plain text message filters.TEXT & ~filters.COMMAND means “text that isn’t a /command”.

#telegram asks handle message for the decision and sends the router's reply: the billing acknowledgement to billing message 
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    message = update.effective_message
    try: 
        msg = normalize_telegram(
            first_name=user.first_name if user else None,
            text=message.text,
            chat_id=update.effective_chat.id,
        )
    except ValidationError:
        await message.reply_text("Sorry, I couldn't read that. Please send a short text message.")
        return
    result = await handle_message(msg)
    
    if result is None or result.decision is None:
        reply = DEFAULT_REPLY
    elif result.decision.reply is None:
        return 
    else:
        reply = result.decision.reply
    await message.reply_text(reply)
    
def build_application(token: str) -> Application:
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    return app