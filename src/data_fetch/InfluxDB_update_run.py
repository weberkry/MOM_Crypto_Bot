
import pandas as pd
import yfinance as yf
import ccxt
from datetime import datetime, timezone, timedelta
from mom import influxDB_utils as influx
from mom import Mandelbrot

import os
from dotenv import load_dotenv

# Load env from project root
load_dotenv()

# the following entries are expected in the MOM_Crypto_Bot/.env
ASSET = os.getenv("ASSET")
CURRENCY = os.getenv("CURRENCY")

# define interval
GRANULARITIES = {
    "Day": "1d",
    "Minute": "1m",
    # "Week": not directly supported in yfinance → use 1d and resample
}

client = influx.get_client()
query_api = client.query_api()

'''
def get_last_timestamp(measurement: str, asset: str, currency: str) -> datetime:
    """
    Query InfluxDB to get the most recent timestamp for a measurement/asset/currency.
    """
    query = f"""
    from(bucket: "{influx.INFLUX_BUCKET}")
      |> range(start: 0)
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> filter(fn: (r) => r.asset == "{asset}")
      |> filter(fn: (r) => r.currency == "{currency}")
      |> filter(fn: (r) => r._field == "high")
      |> last()
    """
    tables = query_api.query(org=influx.INFLUX_ORG, query=query)
    if not tables or not tables[0].records:
        return None
    
    print(measurement,"last entry:",tables[0].records[0].get_time())
    return tables[0].records[0].get_time()

def fetch_data(asset: str, currency: str, interval: str, start: datetime) -> pd.DataFrame:
    """
    Fetch OHLC data from yfinance from 'start' until now.
    - For Minute data: limited to 7 days history (yfinance restriction).
    - For Day data (and higher): full range supported.
    """
    ticker = yf.Ticker(f"{asset}-{currency}")
    granularity = GRANULARITIES.get(interval)

    if interval == "Minute":
        # yfinance only supports up to 7d of 1m data
        earliest = datetime.now(timezone.utc) - timedelta(days=7)
        adj_start = max(start, earliest)
        print(f"Fetching {interval} data since {adj_start} (limited to 7d)")
        df = ticker.history(interval=granularity, start=adj_start)
    else:
        print(f"Fetching {interval} data since {start}")
        df = ticker.history(interval=granularity, start=start)

    if df.empty:
        return df

    # Standard processing
    df = Mandelbrot.reset_index(df)
    df = Mandelbrot.process_raw_DF(df)
    df = df.set_index("date")
    df["asset"] = asset
    df["currency"] = currency
    df["interval"] = interval

    # Ensure float types
    for col in ["volume", "high", "low", "delta", "delta_log", "return", "return_log"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    df = Mandelbrot.fill_nas(df)
    print(df.head())
    return df


    """
    Fetch OHLC data from yfinance from 'start' until now.
    """
    ticker = yf.Ticker(f"{asset}-{currency}")
    print(GRANULARITIES.get(interval))
    if interval == "Minute":
        df = ticker.history(interval="1m", period="7d")
    else:
        df = ticker.history(interval=GRANULARITIES.get(interval), start=start)
  
    if df.empty:
        return df
    
    df = Mandelbrot.reset_index(df)
    df = Mandelbrot.process_raw_DF(df)
    df = df.set_index("date")
    df["asset"] = asset
    df["currency"] = currency
    df["interval"] = interval
    for i in ["volume", "high","low","delta","delta_log","return","return_log"]:
        df[i] = df[i].astype(float)

    df = Mandelbrot.fill_nas(df)
    print(df)
    return df


def update(max_retries: int = 3):
    """
    Update InfluxDB with new data for all configured assets/currencies/intervals.
    Retries fetching data up to `max_retries` times if Yahoo Finance returns empty.
    """
    for curr in [CURRENCY]:
        for asset in [ASSET]:
            for interval, gran in GRANULARITIES.items():
                measurement = interval
                last_time = get_last_timestamp(measurement, asset, curr)
                start = last_time + timedelta(seconds=1) if last_time else datetime(2015, 1, 1, tzinfo=timezone.utc)

                print(f"Updating {measurement} for {asset} since {start}")

                retries = 0
                df = pd.DataFrame()

                while retries < max_retries:
                    df = fetch_data(asset, curr, interval, start)
                    if not df.empty:
                        break
                    retries += 1
                    wait_time = 5 * retries  # backoff (5s, 10s, 15s...)
                    print(f"Empty dataframe (attempt {retries}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)

                if df.empty:
                    print(f"No new data for {asset} ({interval}) after {max_retries} retries.")
                    continue

                influx.write_dataframe(df, measurement=measurement)
                print(f"Updated {asset} ({interval}) with {len(df)} new rows")

'''

def update():
    MIN, DAY = Mandelbrot.update_datasets(crypto=ASSET,curr = CURRENCY)
    print(MIN.head())
    influx.write_dataframe(MIN, measurement="Minute")
    influx.write_dataframe(DAY, measurement="Day")


if __name__ == "__main__":
    update()