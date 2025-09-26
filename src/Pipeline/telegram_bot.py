import sys, os

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.ext import CommandHandler

from mom import pipeline_utils as pipeline


from openai import OpenAI
from sentence_transformers import SentenceTransformer
#import faiss
import hnswlib

#news
from newsapi import NewsApiClient
import praw


#the following entries are expected in the MOM_Crypto_Bot/.env
ASSET = os.getenv("ASSET")
CURRENCY = os.getenv("CURRENCY")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
#OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MODEL = "gpt-4o-mini"
#OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
OPENAI_TEMPERATURE = float("0.0")

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
#REDDIT_ID = os.getenv("REDDIT_CLIENT_ID")
#REDDIT_SECRET = os.getenv("REDDIT_SECRET")
TELEGRAM_API = os.getenv("TELEGRAM_API")

OpenAI.api_key = OPENAI_KEY
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)


async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    asset = "BTC"  # default
    if context.args:  # e.g. /risk ETH
        asset = context.args[0].upper()

    await update.message.reply_text(f"Evaluating risk for {asset}...")

    result = pipeline.run_rag_risk(asset)
    if "error" in result:
        await update.message.reply_text(f"{result['error']}")
    else:
        await update.message.reply_text(result["llm_result"])



# Telegram bot
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    response = agent.run(user_message)
    await update.message.reply_text(response)

app = ApplicationBuilder().token(TELEGRAM_API).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
# Register handler
app.add_handler(CommandHandler("risk", risk_command))


if __name__ == "__main__":
    print("Bot is running...")
    app.run_polling()