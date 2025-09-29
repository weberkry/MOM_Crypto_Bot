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
from mom import analysis_utils as analysis


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
    """Initialize index only once"""
    global _index_initialized
    if not _index_initialized:
        _index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        _index.set_ef(50)
        _index_initialized = True

def categorize_hurst(h: float):
    if h is None:
        return "unknown"
    if h < 0.45:
        return "high_volatility"
    elif h < 0.55:
        return "random_walk"
    else:
        return "trending"

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

# ---- safe chunking ----
def chunk_text(txt: str, max_words=200):
    words = txt.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
# ---- incremental indexing ----
def add_articles_to_index(articles: List[Dict], prefix_id=0):
    """Index article chunks. Returns dict of id->meta."""
    init_index(max_elements=max(10000, len(articles)*4))
    current_max_id = max(_articles_store.keys())+1 if _articles_store else 0
    idx = current_max_id
    texts = []
    metas = []
    for art in articles:
        full_text = " ".join([art["title"], art["description"], art["content"]])
        chunks = chunk_text(full_text, max_words=200)
        for chunk in chunks:
            texts.append(chunk)
            metas.append({
                "title": art["title"],
                "url": art["url"],
                "source": art["source"],
                "publishedAt": art["publishedAt"]
            })
    if not texts:
        return {}
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    ids = list(range(idx, idx + len(texts)))
    _index.add_items(embeddings, ids)
    for i, m in zip(ids, metas):
        _articles_store[i] = {"text": texts[i-idx], "meta": m}
    return {i: _articles_store[i] for i in ids}
# ---- safe retrieval ----
def retrieve(query: str, k=5):
    if not _index_initialized or not _articles_store:
        print("taking the None road")
        return []
    vec = embed_model.encode([query], convert_to_numpy=True)
    labels, distances = _index.knn_query(vec, k=min(k, len(_articles_store)))
    results = []
    for lid in labels[0]:
        if lid in _articles_store:
            results.append({**_articles_store[lid]["meta"], "text": _articles_store[lid]["text"], "id": lid})
    return results



# ---- LLM prompt + call
def build_prompt(asset: str, 
                 hurst_minute: float, hurst_minute_cat: str, 
                 hurst_daily: float, hurst_daily_cat: str, 
                 cvm_norm_Day: float,
                 cvm_norm_Minute: float, 
                 cvm_cauchy_Day: float, 
                 cvm_cauchy_Minute: float,  
                 retrieved: List[Dict]) -> str:
    # --- Header with hurst values and PDF Fit---
    header = (
        f"Asset: {asset}\n"
        f"Hurst (Minute): {hurst_minute} ({hurst_minute_cat})\n"
        f"Hurst (Daily): {hurst_daily} ({hurst_daily_cat})\n"
        f"Cramer von Mises (Norm Fit Minute): {cvm_norm_Minute}\n"
        f"Cramer von Mises (Norm Fit Day): {cvm_norm_Day}\n"
        f"Cramer von Mises (Cauchy Fit Minute): {cvm_cauchy_Minute}\n"
        f"Cramer von Mises (Cauchy Fit Day): {cvm_cauchy_Day}\n\n"
    )
    
    # --- Instruction ---
    header += (
        "You are a financial risk analyst, statsistic expert and familiar with Misbehavior of Markets (Benoit Mandelbrot)" 
        "Using the Hurst signals (above)"
        "and normal distribution vs cauchy distribution of the deltas between datapoints"
        "fitted with cramer von mises evaluation (above)"
        "for both the Minute and Daily intervals, together with the news snippets below, "
        "produce a concise risk evaluation (1–3 short paragraphs) that includes:\n"
        "- a single-line risk level (LOW / MEDIUM / HIGH)\n"
        "- a short explaination which distribution fits best and what that means for the data"
        "- a short justification referencing both Hurst signals and at least two article snippets (cite source + date)\n"
        "- suggestions for monitoring (what to watch next)\n\n"
    )
    
    # --- Add snippets ---
    prompt = header + "News snippets (most relevant first):\n"
    for r in retrieved:
        pub = r.get("publishedAt")
        src = r.get("source")
        title = r.get("title", "")
        snippet = r.get("text", "")[:800]
        prompt += f"\n[{src} | {pub}] {title}\n{snippet}\n---\n"
    
    # --- Final instruction ---
    prompt += (
        "\nNow provide the risk evaluation. "
        "Be concise and make explicit references to both the Hurst signals and the news snippets.\n"
    )
    return prompt

def ask_llm(prompt: str, model: str = OPENAI_MODEL, temperature=OPENAI_TEMPERATURE, max_tokens=600):
    """Calls OpenAI Chat Completions API using the new client."""
    messages = [
        {"role": "system", "content": "You are a professional quantitative risk analyst."},
        {"role": "user", "content": prompt},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

def run_rag_risk(asset="BTC", lookback_news_days=7, top_k=5):
    """Run end-to-end RAG pipeline for risk analysis on a given asset."""

    # 1) Hurst
    #h = get_latest_hurst(asset)
    print("Step 1 --- Hurst --- ")
    H_Day= influx.query_returns(asset="BTC", interval="Day", start="-30d",field="hurst", bucket="Hurst")
    H_Day = H_Day.hurst.iloc[-1]
    H_Minute = influx.query_returns(asset="BTC", interval="Minute", start="-30d",field="hurst", bucket="Hurst")
    H_Minute = H_Minute.hurst.iloc[-1]
    h_cat_day = categorize_hurst(H_Day)
    h_cat_min = categorize_hurst(H_Minute)
    print("Hurst (Day):",H_Day," ",h_cat_day)
    print("Hurst (Min):",H_Minute," ",h_cat_min)

    # 2) Get PDF FIT
    print("Step 2 --- Prbability Density Function --- ")
    #DF_Day = influx.query_returns(asset=ASSET, interval="Day", start="0", field="delta")
    #DF_Min = influx.query_returns(asset=ASSET, interval="Minute", start="0", field="delta")
    #DF_Min = DF_Min.sample(n=50000, random_state=42) # Full DF is too big
    
    #cvm_Min = analysis.pdf_fit_return(DF_Min)
    #cvm_Day = analysis.pdf_fit_return(DF_Day)
    def get_cvm_values(DF):
        #cvm_cauchy = [(DF["parameter"] == "cvm") & (DF["pdf"] == "cauchy")]
        subset = DF[DF["parameter"].isin(["cvm"])]
        subset_c = subset[subset["pdf"].isin(["cauchy"])]
        subset_g = subset[subset["pdf"].isin(["gauss"])]
        cvm_c = subset_c._value.iloc[0]
        #cvm_gauss = [(DF["parameter"] == "cvm") & (DF["pdf"] == "gauss")]
        cvm_g = subset_g._value.iloc[0]
        return [cvm_g, cvm_c]

    pdf_Min = analysis.get_pdf(interval="Minute")
    pdf_Day = analysis.get_pdf(interval="Day")

    cvm_Min = get_cvm_values(pdf_Min)
    cvm_Day = get_cvm_values(pdf_Day)

    # 23) Fetch news (relative lookback)
    print("Step 3 --- Get recent news --- ")
    since = datetime.now(timezone.utc) - timedelta(days=lookback_news_days)
    from_date = since.strftime("%Y-%m-%dT%H:%M:%S")
    articles = fetch_newsapi_articles(query=asset, from_date=from_date, page_size=50)
    #print(articles)
    if not articles:
        return {
            "asset": asset,
            "error": "No articles found",
            "hurst_value_daily": H_Day,
            "hurst_cat_daily": h_cat_day,
            "hurst_value_minute": H_Minute,
            "hurst_cat_minute": h_cat_min,
            "cvm_norm_Day": cvm_Day[0],
            "cvm_norm_Minute": cvm_Min[0], 
            "cvm_cauchy_Day": cvm_Day[1], 
            "cvm_cauchy_Minute": cvm_Min[1],
            "retrieved": [],
            "llm_result": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # 4) Add articles to index
    added = add_articles_to_index(articles)
    #print(added)
    if not added:
        return {
            "asset": asset,
            "error": "No chunks indexed",
            "hurst_value_daily": H_Day,
            "hurst_cat_daily": h_cat_day,
            "hurst_value_minute": H_Minute,
            "hurst_cat_minute": h_cat_min,
            "cvm_norm_Day": cvm_Day[0],
            "cvm_norm_Minute": cvm_Min[0], 
            "cvm_cauchy_Day": cvm_Day[1], 
            "cvm_cauchy_Minute": cvm_Min[1],  
            "retrieved": [],
            "llm_result": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # 5) Retrieve relevant items
    retrieved = retrieve(asset, k=top_k)
    #print(retrieved)
    if not retrieved:
        return {
            "asset": asset,
            "error": "No retrieval results",
            "hurst_value_daily": H_Day,
            "hurst_cat_daily": h_cat_day,
            "hurst_value_minute": H_Minute,
            "hurst_cat_minute": h_cat_min,
            "cvm_norm_Day": cvm_Day[0],
            "cvm_norm_Minute": cvm_Min[0], 
            "cvm_cauchy_Day": cvm_Day[1], 
            "cvm_cauchy_Minute": cvm_Min[1],
            "retrieved": [],
            "llm_result": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # 6) Build prompt and query LLM
    prompt = build_prompt(asset, H_Minute, h_cat_min,H_Day, h_cat_day,cvm_Day[0],cvm_Min[0],cvm_Day[1],cvm_Min[1], retrieved)
    
    #print(prompt)
    llm_result = ask_llm(prompt)

    return {
        "asset": asset,
        "hurst_value_daily": H_Day,
        "hurst_cat_daily": h_cat_day,
        "hurst_value_minute": H_Minute,
        "hurst_cat_minute": h_cat_min,
        "cvm_norm_Day": cvm_Day[0],
        "cvm_norm_Minute": cvm_Min[0], 
        "cvm_cauchy_Day": cvm_Day[1], 
        "cvm_cauchy_Minute": cvm_Min[1],
        "retrieved": retrieved,
        "llm_result": llm_result,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
