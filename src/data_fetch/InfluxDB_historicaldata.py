from mom import influxDB_utils as influx
from mom import Mandelbrot
from influxdb_client import Point
from datetime import datetime, timezone

for c in ["BTC"]:
    MIN, HOUR, DAY, WEEK, MONTH = Mandelbrot.historical_datasets(crypto=c, curr="EUR")
    #for INTERVAL in [WEEK,MONTH]:
    #    influx.write_dataframe(INTERVAL, measurement=INTERVAL.interval.iloc[0])
    for INTERVAL in [DAY,MIN]:
        chunk_size = 1000
        total_rows = len(INTERVAL)

        for start in range(0, total_rows, chunk_size):
            chunk = INTERVAL.iloc[start:start + chunk_size]
            influx.write_dataframe(chunk, measurement=INTERVAL.interval.iloc[0])
            print(f"Wrote rows {start}-{start+len(chunk)-1} ")
	
	
