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



def fetch_newsapi_articles(
    query: str,
    range_days: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    page_size: int = 30
) -> List[Dict]:
    """
    Fetch articles from NewsAPI.
    
    Parameters
    ----------
    query : str
        Keyword(s) to search for (e.g. "BTC").
    range : str, optional
        Relative range, e.g. '7d' for 7 days, '24h' for 24 hours.
    from_date : str, optional
        Absolute start date, format 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS'.
    to_date : str, optional
        Absolute end date, same format as from_date.
    page_size : int
        Max results per page (default 50, NewsAPI limit is 100).
    """
    params = {
        "q": query,
        "page_size": page_size,
        "language": "en",
        "sort_by": "publishedAt"
    }

    # --- relative range ---
    if range_days and not from_date:
        unit = range_days[-1]
        value = int(range_days[:-1])
        now = datetime.now(timezone.utc)

        if unit == "d":
            from_dt = now - timedelta(days=value)
        elif unit == "h":
            from_dt = now - timedelta(hours=value)
        else:
            raise ValueError("Unsupported range format. Use like '7d' or '24h'.")

        params["from_param"] = from_dt.strftime("%Y-%m-%dT%H:%M:%S")

    # --- absolute range ---
    if from_date:
        params["from_param"] = from_date
    if to_date:
        params["to"] = to_date

    # --- call NewsAPI ---
    res = newsapi.get_everything(**params)
    articles = res.get("articles", [])

    cleaned = []
    for a in articles:
        cleaned.append({
            "title": a.get("title") or "",
            "description": a.get("description") or "",
            "content": (a.get("content") or "")[:4000],
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
            "source": a.get("source", {}).get("name")
        })
    return cleaned


# --- chunk text for adding vecotrized articles to index
def chunk_text(txt: str, max_words=200):
    
    words = txt.split()
    chunks = []
    
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    
    return chunks

# ---- safe chunking ----
def chunk_text(txt: str, max_words=200):
    words = txt.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

