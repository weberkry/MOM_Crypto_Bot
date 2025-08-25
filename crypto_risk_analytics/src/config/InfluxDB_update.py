import sys, os

# go up one level from the notebook folder
sys.path.append(os.path.abspath("../analysis"))
sys.path.append(os.path.abspath("../config"))

import pandas as pd
import yfinance as yf
import ccxt
from datetime import datetime, timezone
from influxDB_utils import write_dataframe, get_query_api  #my file


exchange = ccxt.binance()

def get_last_timestamp(asset, currency, interval):
    """
    Query InfluxDB for the most recent timestamp of a series.
    """
    query = f"""
    from(bucket: "TEST")
      |> range(start: -10y)
      |> filter(fn: (r) => r._measurement == "crypto_price")
      |> filter(fn: (r) => r.asset == "{asset}")
      |> filter(fn: (r) => r.currency == "{currency}")
      |> filter(fn: (r) => r.interval == "{interval}")
      |> keep(columns: ["_time"])
      |> sort(columns: ["_time"], desc: true)
      |> limit(n:1)
    """
    tables = get_query_api.query(query)
    if tables and tables[0].records:
        return tables[0].records[0].get_time()
    return None


# --- DAILY + WEEKLY via yfinance ---
def update_yfinance(asset, currency, symbol):
    print(f"Updating {asset}-{currency} (yfinance)...")

    last_ts = get_last_timestamp(asset, currency, "Day")
    start_date = last_ts.strftime("%Y-%m-%d") if last_ts else "2014-01-01"

    df = yf.download(symbol, start=start_date, interval="1d")
    df = df.reset_index()

    # Daily
    df_day = pd.DataFrame({
        "time": pd.to_datetime(df["Date"]).dt.tz_localize("UTC"),
        "price": df["Close"].astype(float),
        "asset": asset,
        "currency": currency,
        "interval": "Day"
    })

    # Weekly resample
    df_week = df_day.resample("W-MON", on="time").last().dropna().reset_index()
    df_week["asset"] = asset
    df_week["currency"] = currency
    df_week["interval"] = "Week"

    write_dataframe(df_day)
    write_dataframe(df_week)

    print(f"Update {asset}-{currency} (daily + weekly) complete")


# --- MINUTE + HOURLY via Binance ---
def update_binance(asset, currency, symbol, timeframe="1m", interval="Minute", limit=1000):
    print(f"Updating {asset}-{currency} ({interval}) from Binance...")

    last_ts = get_last_timestamp(asset, currency, interval)
    since = int(last_ts.timestamp() * 1000) if last_ts else None

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    if not ohlcv:
        print("No new data from Binance")
        return

    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)

    df_out = pd.DataFrame({
        "time": df["time"],
        "price": df["close"].astype(float),
        "asset": asset,
        "currency": currency,
        "interval": interval
    })

    write_dataframe(df_out)
    print(f"Update {asset}-{currency} ({interval}) complete")


def update_crypto(asset="BTC", currency="EUR"):
    symbol_yf = f"{asset}-{currency}" if currency != "USDT" else f"{asset}-USD"
    symbol_binance = f"{asset}/{currency}"

    # YFinance → daily + weekly
    try:
        update_yfinance(asset, currency, symbol_yf)
    except Exception as e:
        print(f"yfinance error: {e}")

    # Binance → minute + hourly
    try:
        update_binance(asset, currency, symbol_binance, timeframe="1m", interval="Minute")
        update_binance(asset, currency, symbol_binance, timeframe="1h", interval="Hour")
    except Exception as e:
        print(f"Binance error: {e}")


