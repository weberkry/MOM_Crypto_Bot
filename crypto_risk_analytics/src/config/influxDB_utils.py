import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
from datetime import datetime, timezone, timedelta

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


#def write_dataframe(df):
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


def write_dataframe(df, measurement="crypto_price_", chunk_size=10_000):
    """
    Write a pandas DataFrame to InfluxDB in safe chunks to avoid timeouts.
    Expects:
      - index: datetime64[ns, UTC]
      - columns: price, asset, currency, interval (+ any other fields)
    """
    client = get_client()

    # Use asynchronous batched writing
    write_api = client.write_api(
        write_options=WriteOptions(
            batch_size=5000,          # number of rows per batch
            flush_interval=10_000,    # flush buffer every 10s
            jitter_interval=2_000,    # randomize flush a little
            retry_interval=5_000,     # retry wait if failed
            max_retries=5,            # max retries
            max_retry_delay=30_000,   # max retry backoff
            exponential_base=2
        )
    )

    if not df.index.dtype.kind == "M":
        raise ValueError("DataFrame index must be datetime64[ns, UTC]")

    # Split DF into chunks
    total_rows = len(df)
    for start in range(0, total_rows, chunk_size):
        chunk = df.iloc[start:start+chunk_size]

        write_api.write(
            bucket=INFLUX_BUCKET,
            org=INFLUX_ORG,
            record=chunk,
            data_frame_measurement_name=interval.iloc[0],
            data_frame_tag_columns=["asset", "currency", "interval"],
        )

        print(f"Wrote rows {start}–{start+len(chunk)-1} "
              f"({len(chunk)}) to InfluxDB bucket '{INFLUX_BUCKET}'")

    print(f"Finished writing {total_rows} rows to InfluxDB")


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


#def backup_to_csv(output_dir="backup", bucket=INFLUX_BUCKET):

    """
    Query all data from a bucket and save separate CSVs
    for each (asset, interval) combination.
    """
    client = get_client()
    query_api = client.query_api()

    # --- Step 1: get unique tag values ---
    flux_assets = f'''
    import "influxdata/influxdb/schema"
    schema.tagValues(bucket: "{bucket}", tag: "asset")
    '''
    assets = []
    for table in query_api.query(flux_assets):
        for record in table.records:
            assets.append(record.get_value())

    flux_intervals = f'''
    import "influxdata/influxdb/schema"
    schema.tagValues(bucket: "{bucket}", tag: "interval")
    '''
    intervals = []
    for table in query_api.query(flux_intervals):
        for record in table.records:
            intervals.append(record.get_value())

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    # Step 2: Loop over tag combos
    for asset in assets:
        for interval in intervals:
            flux = f'''
            from(bucket: "{bucket}")
              |> range(start: 0)
              |> filter(fn: (r) => r["asset"] == "{asset}" and r["interval"] == "{interval}")
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
              |> keep(columns: ["_time","_measurement","asset","currency","interval","price","volume"])
            '''
            df = query_api.query_data_frame(flux)

            if df.empty:
                print(f"No data for {asset} @ {interval}")
                continue

            df["_time"] = pd.to_datetime(df["_time"])
            filename = f"{bucket}_{asset}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"Saved {filepath}")
            saved_files.append(filepath)

    return saved_files


#def backup_to_csv_by_year(output_dir="backup", bucket=INFLUX_BUCKET):

    """
    Query all data from a bucket and save separate CSVs per asset/interval/year
    to avoid timeouts for large datasets (Minute data).
    """
    client = get_client()
    query_api = client.query_api()

    # Step 1: get unique tag values
    flux_assets = '''
    import "influxdata/influxdb/schema"
    schema.tagValues(bucket: "%s", tag: "asset")
    ''' % bucket
    assets = [record.get_value() for table in query_api.query(flux_assets) for record in table.records]

    flux_intervals = '''
    import "influxdata/influxdb/schema"
    schema.tagValues(bucket: "%s", tag: "interval")
    ''' % bucket
    intervals = [record.get_value() for table in query_api.query(flux_intervals) for record in table.records]

    os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    for asset in assets:
        for interval in intervals:
            # Determine date range for iteration
            current_year = 2015  # or the first year of your data
            this_year = datetime.now().year

            while current_year <= this_year:
                start = datetime(current_year, 1, 1, tzinfo=timezone.utc)
                end = datetime(current_year + 1, 1, 1, tzinfo=timezone.utc)

                flux = f'''
                from(bucket: "{bucket}")
                  |> range(start: {start.isoformat()}, stop: {end.isoformat()})
                  |> filter(fn: (r) => r["asset"] == "{asset}" and r["interval"] == "{interval}")
                  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                  |> keep(columns: ["_time","asset","currency","interval","price","volume"])
                '''
                df = query_api.query_data_frame(flux)

                if not df.empty:
                    df["_time"] = pd.to_datetime(df["_time"])
                    filename = f"{bucket}_{asset}_{interval}_{current_year}.csv"
                    filepath = os.path.join(output_dir, filename)
                    df.to_csv(filepath, index=False)
                    print(f"Saved {filepath} ({len(df)} rows)")
                    saved_files.append(filepath)

                current_year += 1

    return saved_files



def backup_csv(output_dir="backup", bucket=INFLUX_BUCKET):
    """
    Backup InfluxDB data into CSVs.
    - Day/Hour/Week: yearly
    - Minute: monthly (to avoid timeouts)
    """
    client = get_client()
    query_api = client.query_api()

    # Get unique tag values
    flux_assets = f'''
    import "influxdata/influxdb/schema"
    schema.tagValues(bucket: "{bucket}", tag: "asset")
    '''
    assets = [r.get_value() for t in query_api.query(flux_assets) for r in t.records]

    flux_intervals = f'''
    import "influxdata/influxdb/schema"
    schema.tagValues(bucket: "{bucket}", tag: "interval")
    '''
    intervals = [r.get_value() for t in query_api.query(flux_intervals) for r in t.records]

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    for asset in assets:
        for interval in intervals:
            # Determine start/end periods
            now = datetime.now().year
            if interval == "Minute":
                # Monthly chunks
                start_year = 2015
                start_month = 1
                while start_year <= now:
                    for month in range(1, 13):
                        start = datetime(start_year, month, 1, tzinfo=timezone.utc)
                        if month == 12:
                            end = datetime(start_year+1, 1, 1, tzinfo=timezone.utc)
                        else:
                            end = datetime(start_year, month+1, 1, tzinfo=timezone.utc)

                        flux = f'''
                        from(bucket: "{bucket}")
                          |> range(start: {start.isoformat()}, stop: {end.isoformat()})
                          |> filter(fn: (r) => r["asset"] == "{asset}" and r["interval"] == "{interval}")
                          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                          |> keep(columns: ["_time","asset","currency","interval","price","volume"])
                        '''
                        df = query_api.query_data_frame(flux)
                        if not df.empty:
                            df["_time"] = pd.to_datetime(df["_time"])
                            filename = f"{bucket}_{asset}_{interval}_{start_year}_{month:02d}.csv"
                            filepath = os.path.join(output_dir, filename)
                            df.to_csv(filepath, index=False)
                            print(f"✅ Saved {filepath} ({len(df)} rows)")
                            saved_files.append(filepath)
                    start_year += 1
            else:
                # Yearly chunks
                for year in range(2015, now+1):
                    start = datetime(year, 1, 1, tzinfo=timezone.utc)
                    end = datetime(year+1, 1, 1, tzinfo=timezone.utc)
                    flux = f'''
                    from(bucket: "{bucket}")
                      |> range(start: {start.isoformat()}, stop: {end.isoformat()})
                      |> filter(fn: (r) => r["asset"] == "{asset}" and r["interval"] == "{interval}")
                      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                      |> keep(columns: ["_time","asset","currency","interval","price","volume"])
                    '''
                    df = query_api.query_data_frame(flux)
                    if not df.empty:
                        df["_time"] = pd.to_datetime(df["_time"])
                        filename = f"{bucket}_{asset}_{interval}_{year}.csv"
                        filepath = os.path.join(output_dir, filename)
                        df.to_csv(filepath, index=False)
                        print(f"Saved {filepath} ({len(df)} rows)")
                        saved_files.append(filepath)

    return saved_files
