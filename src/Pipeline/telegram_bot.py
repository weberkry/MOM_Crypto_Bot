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
from deep_translator import GoogleTranslator
from langdetect import detect

# === Import your pipeline + tools ===
from mom.pipeline_agent import agent, rag_risk

# ---- Setup ----
load_dotenv()
TELEGRAM_API = os.getenv("TELEGRAM_API")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    llevel=logging.DEBUG,   
    filename="telegram_bot.log",
    filemode="a"
)
logger = logging.getLogger(__name__)


#  Language translator
def detect_and_translate(text: str, target_lang="en"):
    """Detect input language, translate to target_lang if needed."""
    try:
        detected = detect(text)
    except Exception:
        detected = "en"

    if detected != target_lang:
        translated = GoogleTranslator(source=detected, target=target_lang).translate(text)
    else:
        translated = text

    return detected, translated


# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I’m your Crypto Risk Bot\n"
        "Use /risk BTC to evaluate risk of BTC (according to Benoît Mandelbrot)\n"
        "Or just ask me free-form questions"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Available commands:\n"
        "/start - Welcome message\n"
        "/help - Show this help\n"
        "/risk <ASSET> - Run the risk pipeline (default BTC)\n"
        "Or type any crypto-related question!"
    )


async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    asset = "BTC"
    if context.args:
        asset = context.args[0].upper()

    await update.message.reply_text(f"Evaluating risk for {asset}...")

    try:
        result = rag_risk(asset)  # call tool directly
        await update.message.reply_text(result)
    except Exception as e:
        logger.error(f"/risk error: {e}")
        await update.message.reply_text(f"Error in /risk: {e}")


# Freeform prompts
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    try:
        # 1.Translate to English
        detected_lang, translated_text = detect_and_translate(user_text, target_lang="en")

        # 2.Run agent in a thread (avoid blocking event loop)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, agent.run, translated_text)

        # 3.Translate back
        if detected_lang != "en":
            response = GoogleTranslator(source="en", target=detected_lang).translate(response)

        await update.message.reply_text(response)

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
