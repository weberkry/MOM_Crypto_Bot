import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

# Load env from project root
load_dotenv()

INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")


def get_client():
    if not all([INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET]):
        raise ValueError("Missing InfluxDB env vars. Check your .env file.")
    return InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)


def get_write_api(client):
    return client.write_api(write_options=SYNCHRONOUS)

def get_query_api(client):
    return client.query_api()


def write_dataframe(df):
    """
    Write a pandas DataFrame to InfluxDB.
    Expects columns: time, price, asset, currency, interval
    """
    client = get_client()
    write_api = get_write_api(client)

    # Ensure time column is datetime with timezone
    #
    if not df.index.dtype.kind == "M":
        raise ValueError("DataFrame must have a datetime64[ns, UTC] column named 'time'")

    write_api.write(
        bucket=INFLUX_BUCKET,
        org=INFLUX_ORG,
        record=df,
        data_frame_measurement_name=df.interval.iloc[0],
        data_frame_tag_columns=["asset", "currency", "interval"],
    )
    print(f"Wrote {len(df)} rows to InfluxDB bucket '{INFLUX_BUCKET}'")


def query_last_timestamp(asset, currency, granularity, measurement="crypto_price"):
    """
    Returns the most recent timestamp for a given asset/currency/granularity.
    """
    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -10y)
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> filter(fn: (r) => r.asset == "{asset}")
      |> filter(fn: (r) => r.currency == "{currency}")
      |> filter(fn: (r) => r.granularity == "{granularity}")
      |> keep(columns: ["_time"])
      |> sort(columns: ["_time"], desc: true)
      |> limit(n:1)
    """
    tables = query_api.query(query, org=INFLUX_ORG)
    if tables and tables[0].records:
        return tables[0].records[0].get_time()
    return None
