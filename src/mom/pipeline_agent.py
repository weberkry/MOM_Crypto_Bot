import os
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict

from mom import influxDB_utils as influx
from mom import analysis_utils as analysis
from mom import pipeline_utils as pipeline
from mom import Mandelbrot

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType

from langchain_community.vectorstores import FAISS
#news
from newsapi import NewsApiClient
import praw

import sys, os
from dotenv import load_dotenv
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

#OpenAI.api_key = OPENAI_KEY
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
#client = OpenAI(api_key=OPENAI_KEY)


# ---- Embeddings ----
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
VectorStoreClass = FAISS

# ---- LLM ----
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE,openai_api_key=OPENAI_KEY)


# ---- NewsAPI fetch function ----
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

# ---- Tools ----
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


@tool("get_hurst", return_direct=False)
def get_hurst(asset: str) -> str:
    """Get the most recent daily and minute Hurst values for an asset."""
    H_Day= influx.query_returns(asset="BTC", interval="Day", start="-30d",field="hurst", bucket="Hurst")
    H_Day = H_Day.hurst.iloc[-1]
    H_Minute = influx.query_returns(asset="BTC", interval="Minute", start="-30d",field="hurst", bucket="Hurst")
    H_Minute = H_Minute.hurst.iloc[-1]
    h_cat_day = analysis.categorize_hurst(H_Day)
    h_cat_min = analysis.categorize_hurst(H_Minute)
    return json.dumps({
        "daily": H_Day,
        "minute": H_Minute,
        "daily_cat": h_cat_day,
        "minute_cat": h_cat_min,
    })

@tool("get_pdf_fit", return_direct=False)
def get_pdf_fit(asset: str) -> str:
    """Fetch PDF fitting statistics (Cramer von Mises) for Day and Minute intervals."""
    def get_cvm_values(DF):
        subset = DF[DF["parameter"] == "cvm"]
        return {
            "gauss": subset[subset["pdf"] == "gauss"]._value.iloc[0],
            "cauchy": subset[subset["pdf"] == "cauchy"]._value.iloc[0],
        }
    pdf_Min = analysis.get_pdf(interval="Minute")
    pdf_Day = analysis.get_pdf(interval="Day")
    return json.dumps({"Day": get_cvm_values(pdf_Day), "Minute": get_cvm_values(pdf_Min)})


@tool("rag_risk", return_direct=True)
def rag_risk(asset: str) -> str:
    """Run the full risk pipeline: Hurst, PDF fits, News RAG → risk evaluation."""
    # Hurst
    H_Day= influx.query_returns(asset="BTC", interval="Day", start="-30d",field="hurst", bucket="Hurst")
    H_Day = H_Day.hurst.iloc[-1]
    H_Minute = influx.query_returns(asset="BTC", interval="Minute", start="-30d",field="hurst", bucket="Hurst")
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

    # News
    since = datetime.now(timezone.utc) - timedelta(days=7)
    from_date = since.strftime("%Y-%m-%dT%H:%M:%S")
    articles = fetch_newsapi_articles(query=asset, from_date=from_date, page_size=50)

    # Vector index
    docs = [
        Document(page_content=" ".join([a["title"], a["description"], a["content"]]),
                 metadata={"title": a["title"], "source": a["source"], "publishedAt": a["publishedAt"]})
        for a in articles
    ]
    if not docs:
        return "No articles found"
    vectorstore = VectorStoreClass.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.get_relevant_documents(asset)
    retrieved = [{**d.metadata, "text": d.page_content} for d in retrieved_docs]

    # Build prompt
    prompt = pipeline.build_prompt(
        asset,
        H_Minute, h_cat_min,
        H_Day, h_cat_day,
        cvm_Day[0], cvm_Min[0],
        cvm_Day[1], cvm_Min[1],
        retrieved
    )
    
    resp = pipeline.ask_llm(prompt)
    
    return resp

# New 24h news fetch function
#def fetch_news_tool(asset: str):
#    """Fetches news articles from the last 24 hours for a given asset."""
#    from datetime import datetime, timezone, timedelta
#    since = datetime.now(timezone.utc) - timedelta(days=1)
#    from_date = since.strftime("%Y-%m-%dT%H:%M:%S")
#    return fetch_newsapi_articles(query=asset, from_date=from_date, page_size=25)

# Wrap with the tool decorator (latest LangChain style)
fetch_news_single = tool(
    fetch_news_tool,
    description="Fetches news articles from the last 24 hours for a given asset. Input: asset symbol like 'BTC'.",
    return_direct=False
)




tools = [fetch_news_single, get_hurst, get_pdf_fit, rag_risk]  # replace old fetch_news
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


# ---- Example usage ----
if __name__ == "__main__":
    # Structured call
    print(agent.run("Run rag_risk on BTC"))

    # Freeform Q&A
    print(agent.run("What is the risk level for ETH right now based on news and Hurst?"))
