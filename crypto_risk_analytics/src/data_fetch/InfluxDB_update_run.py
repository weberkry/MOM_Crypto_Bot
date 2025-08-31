import sys, os

# go up one level from the notebook folder
sys.path.append(os.path.abspath("../analysis"))
sys.path.append(os.path.abspath("../config"))

import pandas as pd
import yfinance as yf
import ccxt
from datetime import datetime, timezone, timedelta
import influxDB_utils as influx  #my file
import Mandelbrot


## Config
ASSETS = ["BTC", "ETH"]
CURRENCIES = ["EUR"]
GRANULARITIES = {
    "Day": "1d",
    "Hour": "1h",
    "Minute": "1m",
    # "Week": not directly supported in yfinance → use 1d and resample
}

client = influx.get_client()
query_api = client.query_api()


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
    return tables[0].records[0].get_time()


def fetch_data(asset: str,currency: str, interval: str, start: datetime) -> pd.DataFrame:
    """
    Fetch OHLC data from yfinance from 'start' until now.
    """
    ticker = yf.Ticker(f"{asset}-{currency}")
    print(GRANULARITIES.get(interval))
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
    #print(df.head())
    return df


def update():
    for curr in CURRENCIES:
        for asset in ASSETS:
            for interval, gran in GRANULARITIES.items():
                measurement = f"{interval}"
                print(gran, interval)
                last_time = get_last_timestamp(measurement, asset, curr)
                start = last_time + timedelta(seconds=1) if last_time else datetime(2015, 1, 1, tzinfo=timezone.utc)

                print(f"Updating {measurement} for {asset} since {start}")

                df = fetch_data(asset,curr , interval, start)
                if df.empty:
                    print(f"No new data for {asset} ({interval})")
                    continue

                influx.write_dataframe(df)
                print(f"Updated {asset} ({interval}) with {len(df)} new rows")


if __name__ == "__main__":
    update()