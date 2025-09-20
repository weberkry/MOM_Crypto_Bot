import sys, os
from dotenv import load_dotenv
import numpy as np
from datetime import datetime, timezone, timedelta
import pandas as pd

import time
import json
from typing import List, Dict, Optional


# own functions
from mom import influxDB_utils as influx
from mom import Mandelbrot
#from mom import news_utils as news
from mom import pipeline_utils as pipeline


from openai import OpenAI
from sentence_transformers import SentenceTransformer
#import faiss
import hnswlib

#news
from newsapi import NewsApiClient
import praw

import sys, os
# Load env from project root
load_dotenv()

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
client = OpenAI(api_key=OPENAI_KEY)




if __name__ == "__main__":
    print("Running RAG evaluator for BTC...")
    result = pipeline.run_rag_risk("BTC")
    print(result["llm_result"])