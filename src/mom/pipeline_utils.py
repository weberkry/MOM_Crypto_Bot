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


# ---- embedding model + index (global)
# >> sentence Transformer : transforms text into numerical vectors of n = 384 (for this case)
#
EMB_MODEL_NAME = "all-MiniLM-L6-v2"  # small & fast
embed_model = SentenceTransformer(EMB_MODEL_NAME)
DIM = embed_model.get_sentence_embedding_dimension()  #384 dim


# ---- hnswlib index (in-memory) (Persist/restore if needed)
# >> stores the vectorized articles
_index = hnswlib.Index(space="cosine", dim=DIM)
_index_initialized = False
_articles_store = {}  # id -> metadata + text

def init_index(max_elements=20000):
    global _index_initialized
    if not _index_initialized:
        _index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        _index.set_ef(50)
        _index_initialized = True


def fetch_newsapi_articles(
    query: str,
    range_days: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    page_size: int = 50
) -> List[Dict]:
    """
    Fetch articles from NewsAPI.
    
    Parameters
    ----------
    query : str
        Keyword(s) to search for (e.g. "BTC").
    range_days : str, optional
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

    # --- relative range
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

    # --- absolute range
    if from_date:
        params["from_param"] = from_date
    if to_date:
        params["to"] = to_date

    # --- call NewsAPI
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
    """Simple chunker by words."""
    words = txt.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks
