import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# httpx logs every request URL at INFO level, and Telegram URLs contain the bot token.
logging.getLogger("httpx").setLevel(logging.WARNING)

log = logging.getLogger("customer_desk")

from .telegram_bot import build_application
from .web import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    bot = None
    if token:
        bot = build_application(token)
        await bot.initialize()
        await bot.start()
        await bot.updater.start_polling()
        log.info("Telegram bot started (polling)")
    else:
        log.warning("TELEGRAM_BOT_TOKEN not set: running web form only")
    yield
    if bot:
        await bot.updater.stop()
        await bot.stop()
        await bot.shutdown()
        
app = FastAPI(title="Customer Desk", lifespan=lifespan)
app.include_router(router)