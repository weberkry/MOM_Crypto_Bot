import os
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict

import pandas as pd
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from mom import influxDB_utils as influx
from mom import analysis_utils as analysis
from mom import pipeline_utils as pipeline
from mom import Mandelbrot

from langchain_community.vectorstores import FAISS
#news
from newsapi import NewsApiClient
#import newsapi
import praw


# If you use FAISS or another vectorstore/embeddings, import them here:

# from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
VectorStoreClass = FAISS

# Make sure fetch_newsapi_articles is available
# If defined elsewhere, import it here
# from mom.news_utils import fetch_newsapi_articles

import sys, os
from dotenv import load_dotenv
# Load env from project root
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

def fetch_newsapi_articles(
    query: str,
    from_date: str = None,
    page_size: int = 25
) -> List[Dict]:
    """
    Fetch news articles from NewsAPI for a given query since from_date.
    """
    params = {
        "q": query,
        "language": "en",
        "sort_by": "publishedAt",
        "page_size": page_size,
    }

    if from_date:
        params["from_param"] = from_date

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

# Tools for agent
#@tool("fetch_news", return_direct=False)
def fetch_news_tool(query: str) -> str:
    """
    Fetch news articles from the last 24 hours for a given asset symbol.
    Returns a list of cleaned article dictionaries with keys: title, content, source, publishedAt.
    rewritten to avoid zeroshot error --> only 1 input allowed
    hardcoing to fetch articles from the last 24h
    """
  
    # Compute 24h ago in UTC
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    from_date = since.strftime("%Y-%m-%dT%H:%M:%S")

    # Fetch articles
    articles = fetch_newsapi_articles(
        query=query,
        from_date=from_date,
        page_size=25    #hard coded to 25 pages
    )

    if not articles:
        return "No articles found in the last 24 hours."

    # Format as simple text for LLM
    return "\n".join(
        f"{a['source']} ({a['publishedAt']}): {a['title']}" 
        for a in articles
    )


# Fetch news (timedelta fixed)
def fetch_news_tool(query: str) -> str:
    """
    Fetch news articles from the last 24 hours for a given asset symbol.

    Returns a plain text summary of articles (title + source + time).
    Input: query string (asset symbol, e.g. "BTC").
    """
    since = datetime.now(timezone.utc) - timedelta(days=5)
    from_date = since.strftime("%Y-%m-%dT%H:%M:%S")

    articles = fetch_newsapi_articles(query=query, from_date=from_date, page_size=25)

    if not articles:
        return "No articles found in the last 24 hours."

    return "\n".join(
        f"{a['source']} ({a['publishedAt']}): {a['title']}"
        for a in articles
    )


fetch_news_single = tool(
    fetch_news_tool,
    description="Fetches news articles from the last 24 hours for a given asset. Input: asset symbol like 'BTC'.",
    return_direct=False,
)


# Hurst exponent
@tool("get_hurst", return_direct=False)
def get_hurst(asset: str) -> str:
    """Get the most recent daily and minute Hurst values for an asset."""
    H_Day = influx.query_returns(asset=asset, interval="Day", start="-30d", field="hurst", bucket="Hurst")
    H_Day = H_Day.hurst.iloc[-1]

    H_Minute = influx.query_returns(asset=asset, interval="Minute", start="-30d", field="hurst", bucket="Hurst")
    H_Minute = H_Minute.hurst.iloc[-1]

    h_cat_day = analysis.categorize_hurst(H_Day)
    h_cat_min = analysis.categorize_hurst(H_Minute)

    return json.dumps({
        "daily": H_Day,
        "minute": H_Minute,
        "daily_cat": h_cat_day,
        "minute_cat": h_cat_min,
    })


# PDF fit
@tool("get_pdf_fit", return_direct=False)
def get_pdf_fit(asset: str) -> str:
    """Fetch PDF fitting statistics (Cramer–von Mises) for Day and Minute intervals."""
    def get_cvm_values(DF):
        subset = DF[DF["parameter"] == "cvm"]
        return {
            "gauss": subset[subset["pdf"] == "gauss"]._value.iloc[0],
            "cauchy": subset[subset["pdf"] == "cauchy"]._value.iloc[0],
        }

    pdf_Min = analysis.get_pdf(interval="Minute")
    pdf_Day = analysis.get_pdf(interval="Day")

    return json.dumps({
        "Day": get_cvm_values(pdf_Day),
        "Minute": get_cvm_values(pdf_Min),
    })


# RAG Risk Pipeline
@tool("rag_risk", return_direct=True)
def rag_risk(asset: str) -> str:
    """Run the full risk pipeline: Hurst, PDF fits, News RAG → risk evaluation."""
    # Hurst
    H_Day = influx.query_returns(asset=asset, interval="Day", start="-30d", field="hurst", bucket="Hurst")
    H_Day = H_Day.hurst.iloc[-1]

    H_Minute = influx.query_returns(asset=asset, interval="Minute", start="-30d", field="hurst", bucket="Hurst")
    H_Minute = H_Minute.hurst.iloc[-1]

    h_cat_day = analysis.categorize_hurst(H_Day)
    h_cat_min = analysis.categorize_hurst(H_Minute)

    # PDF
    def get_cvm_values(DF):
        subset = DF[DF["parameter"] == "cvm"]
        return [
            subset[subset["pdf"] == "gauss"]._value.iloc[0],
            subset[subset["pdf"] == "cauchy"]._value.iloc[0],
        ]

    pdf_Min = analysis.get_pdf(interval="Minute")
    pdf_Day = analysis.get_pdf(interval="Day")
    cvm_Min = get_cvm_values(pdf_Min)
    cvm_Day = get_cvm_values(pdf_Day)

    # News (7 days back for more context)
    since = datetime.now(timezone.utc) - timedelta(days=7)
    from_date = since.strftime("%Y-%m-%dT%H:%M:%S")
    articles = fetch_newsapi_articles(query=asset, from_date=from_date, page_size=50)

    docs = [
        Document(
            page_content=" ".join([a["title"], a.get("description", ""), a.get("content", "")]),
            metadata={"title": a["title"], "source": a["source"], "publishedAt": a["publishedAt"]}
        )
        for a in articles
    ]

    if not docs:
        return "No articles found."

    # Vector index (requires embeddings + VectorStoreClass set globally/imported)
    vectorstore = VectorStoreClass.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.get_relevant_documents(asset)

    retrieved = [{**d.metadata, "text": d.page_content} for d in retrieved_docs]

    # Build + run prompt
    prompt = pipeline.build_prompt(
        asset,
        H_Minute, h_cat_min,
        H_Day, h_cat_day,
        cvm_Day[0], cvm_Min[0],
        cvm_Day[1], cvm_Min[1],
        retrieved,
    )

    return pipeline.ask_llm(prompt)


#toolbox list
toolbox = [fetch_news_single, get_hurst, get_pdf_fit, rag_risk]
