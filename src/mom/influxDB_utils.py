import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
from datetime import datetime, timezone, timedelta
from mom import Mandelbrot

# Load env from project root
load_dotenv()

#the following entries are expected in the MOM_Crypto_Bot/.env
INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")

BACKUP_DIR = os.getenv("BACKUP_DIR")


def get_client():
    if not all([INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET]):
        raise ValueError("Missing InfluxDB env vars. Check your .env file.")
    return InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)


def get_write_api(client):
    return client.write_api(write_options=SYNCHRONOUS)

def get_query_api(client):
    return client.query_api()


def write_dataframe(df, in_bucket=INFLUX_BUCKET, measurement = "Hurst"):
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
    df.index.name = "_time"

    print(f"Writing to InfluxDB bucket '{in_bucket}-{measurement}'")
    write_api.write(
        bucket=in_bucket,
        org=INFLUX_ORG,
        record=df,
        data_frame_measurement_name=measurement,
        data_frame_tag_columns=["asset", "currency", "interval"],
    )
    print(f"Wrote {len(df)} rows to InfluxDB bucket '{in_bucket}-{measurement}'")



def query_returns(asset="BTC", interval="Day", start="-30d",field="returns"):
    """
    Query 'return' field for a specific asset & interval from InfluxDB.
    
    :param asset: e.g. "BTC"
    :param interval: e.g. "Day", "Hour", "Minute"
    :param start: time range, e.g. "-30d", "-1y", "1970-01-01T00:00:00Z"
    :return: pandas DataFrame with time + return
    """
    #client = get_client()
    #Q = client.get_query_api()
    Q = get_query_api(get_client())

    flux = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {start})
      |> filter(fn: (r) => r["_measurement"] == "{interval}")
      |> filter(fn: (r) => r["asset"] == "{asset}")
      |> filter(fn: (r) => r["_field"] == "{field}")
      |> keep(columns: ["_time", "_value", "asset", "currency", "interval"])
    '''

    tables = Q.query(org=INFLUX_ORG, query=flux)

    # convert FluxTable → pandas DataFrame
    records = []
    for table in tables:
        for record in table.records:  # type: FluxRecord
            records.append({
                "time": record["_time"],
                field: record["_value"],
                "asset": record["asset"],
                "currency": record["currency"],
                "interval": record["interval"],
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df.set_index("time", inplace=True)

    return df


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



def backup_csv(output_dir=BACKUP_DIR, bucket=INFLUX_BUCKET):
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
                          |> keep(columns: ["_time","asset","currency","interval","high","delta","delta_log","return","return_log","volume"])
                        '''
                        df = query_api.query_data_frame(flux)
                        if not df.empty:
                            df["_time"] = pd.to_datetime(df["_time"])
                            filename = f"{bucket}_{asset}_{interval}_{start_year}_{month:02d}.csv"
                            filepath = os.path.join(output_dir, filename)
                            df.to_csv(filepath, index=False)
                            print(f"Saved {filepath} ({len(df)} rows)")
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
                      |> keep(columns: ["_time","asset","currency","interval","high","volume","return","return_log","delta","delta_log"])
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


def write_from_backup(directory = BACKUP_DIR):
    all_entries = os.listdir(directory)
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for f in files:
        print(f)
        
        DF = pd.read_csv(directory+"/"+f)
        DF = DF.set_index("_time")
        DF.index = pd.to_datetime(DF.index, utc=True)
        #print(DF.columns)
        for col in ["high","low","volume", "delta", "delta_log", "return", "return_log"]:
            if col in DF.columns:
                DF[col] = pd.to_numeric(DF[col], errors="coerce")
        # Ensure tag columns are strings
        for col in ["asset", "currency", "interval"]:
            if col in DF.columns:
                DF[col] = DF[col].astype(str)

        DF = Mandelbrot.fill_nas(DF)
        
    
        write_dataframe(DF, measurement=DF.interval.iloc[0])