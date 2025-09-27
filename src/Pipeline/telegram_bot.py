import os
import asyncio
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from langchain.agents import AgentType, initialize_agent

# === import your pipeline + tools ===
from mom.pipeline_agent import agent, rag_risk  # assumes tools+agent are defined in pipeline_agent.py


# /risk command ----
async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    asset = "BTC"  # default
    if context.args:  # e.g. /risk ETH
        asset = context.args[0].upper()

    await update.message.reply_text(f"Evaluating risk for {asset}...")

    try:
        # run the rag_risk tool directly (not via agent)
        result = rag_risk.run(asset)
        await update.message.reply_text(result)
    except Exception as e:
        await update.message.reply_text(f"Error in /risk: {e}")


# ---- freeform chat ----
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, agent.run, user_message)
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Agent error: {e}")


# ---- main ----
def main():
    TELEGRAM_API = os.getenv("TELEGRAM_API")
    app = ApplicationBuilder().token(TELEGRAM_API).build()

    # register handlers
    app.add_handler(CommandHandler("risk", risk_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("I'm up and running! I'll do my best!")
    #print("...sheesh..the internet is dark and full of terror...")
    #print("ok... enough internet... let me think about it...")
    app.run_polling()


if __name__ == "__main__":
    main()
