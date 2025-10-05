import os
import asyncio
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

#Import LangGraph pipeline
import mom.pipeline_langgraph as lg

from langchain_community.vectorstores import FAISS
#news
from newsapi import NewsApiClient
import praw

# General Setup
load_dotenv()
TELEGRAM_API = os.getenv("TELEGRAM_API")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    filename="telegram_bot.log",
    filemode="a"
)
logger = logging.getLogger(__name__)


# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I'm your Crypto Risk Bot\n"
        "Use /risk BTC to evaluate risk of BTC (according to Benoît Mandelbrot)\n"
        "Use /help if you need a reminder of what I can do\n"
        "still in beta: Or just ask me free-form questions\n"
        "...\n"
        "One more thing:\n"
        "I'm running on a RaspberryPi server, so depending on the request I may have to do some heavy calculations.\n"
        "A response time of 2-3min is normal, so please be patient with me. If I do not respond, please try again later."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Available commands:\n"
        "/start - Welcome message\n"
        "/help - Show this help\n"
        "/risk BTC - Run the risk pipeline (default BTC)\n"
        "Or type any crypto-related question!"
    )


async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles /risk command -> runs rag_risk tool directly via lg.run_risk()."""
    asset = "BTC"
    if context.args:
        asset = context.args[0].upper()

    await update.message.reply_text(f"⚡ Evaluating risk for {asset}... please wait")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lg.run_risk, asset)
        await update.message.reply_text(result)
    except Exception as e:
        logger.error(f"/risk error: {e}")
        await update.message.reply_text(f"Error in /risk: {e}")


# ---- Freeform Queries ----
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles free-form user text -> runs through LangGraph pipeline."""
    user_text = update.message.text

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lg.run, user_text)
        await update.message.reply_text(result)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        await update.message.reply_text(f"Agent error: {e}")


# ---- Main ----
def main():
    app = ApplicationBuilder().token(TELEGRAM_API).build()

    # Commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("risk", risk_command))

    # Freeform messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
