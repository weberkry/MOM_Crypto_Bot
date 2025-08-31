import sys, os

# go up one level from the notebook folder
sys.path.append(os.path.abspath("/home/weberkry/git/MOM_Crypto_Bot/crypto_risk_analytics/src/analysis"))
sys.path.append(os.path.abspath("/home/weberkry/git/MOM_Crypto_Bot/crypto_risk_analytics/src/config"))

import Mandelbrot #defined/customized functions from Mandelbrot.py
import influxDB_utils as influx
from influxdb_client import Point
from datetime import datetime, timezone

for c in ["ETH"]:
    MIN, HOUR, DAY, WEEK, MONTH = Mandelbrot.historical_datasets(crypto=c, curr="EUR")
    for INTERVAL in [WEEK,MONTH]:
        influx.write_dataframe(INTERVAL)
    for INTERVAL in [DAY,HOUR,MIN]:
        chunk_size = 1000
        total_rows = len(INTERVAL)

        for start in range(0, total_rows, chunk_size):
            chunk = INTERVAL.iloc[start:start + chunk_size]
            influx.write_dataframe(chunk)
            print(f"Wrote rows {start}-{start+len(chunk)-1} ")
	
	
