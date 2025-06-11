import os
import yaml
#from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI

#from langchain_openai import OpenAI

from langchain.tools import tool
import requests

#telegram bot
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters






#set fixed parameters 


### OPENAI API
with open('/home/christiane/git/stash/API/openai_api.yml', 'r') as file:
    OPENAI_API_KEY = yaml.safe_load(file)["api"]

### TELEGRAM API
with open('/home/christiane/git/stash/API/telegram_api.yml', 'r') as file:
    TELEGRAM_TOKEN = yaml.safe_load(file)["api"]
    



# Initialize LLM

### temperature
#### 0-0.3 -> focused/predictable
#### 0.7-1.0-> creative mode

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# set tools

### Tool 1: Fetch current Crypto Price

@tool
def get_crypto_price_current(symbol: str) -> str:
    """Returns the current price of a given cryptocurrency symbol (e.g., BTC, ETH) using CoinGecko API"""
     #fix for user input BTC&bitcoin
    COIN_LOOKUP = {
        "btc": "bitcoin",
        "bitcoin": "bitcoin",
        "eth": "ethereum",
        "ethereum": "ethereum",
        "sol": "solana",
        "solana": "solana",
        "ada": "cardano",
        "cardano": "cardano",
        "doge": "dogecoin",
        "dogecoin": "dogecoin",
    }
    
    
    try:
        #for coingecko it has to be full name+lowercase
        coin_lowercase = symbol.strip().lower()   #symbol gets generated from package llm (extracted from input)
        coin_id = COIN_LOOKUP[coin_lowercase]     
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=eur"
        response = requests.get(url).json()
        price = response[coin_id]["eur"]
        return f"The current price of {coin_id} is {price}€"
    
    except Exception as e:
        return f"Error fetching price: {e}"

# agent Initialization t

tools = [get_crypto_price_current]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


# Telegram bot
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    response = agent.run(user_message)
    await update.message.reply_text(response)

app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

if __name__ == "__main__":
    print("Bot is running...")
    app.run_polling()