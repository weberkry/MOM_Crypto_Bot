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
from mom import news_utils as news


from openai import OpenAI
from sentence_transformers import SentenceTransformer
#import faiss
import hnswlib

#news
from newsapi import NewsApiClient
import praw

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

OpenAI.api_key = OPENAI_KEY
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)


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

