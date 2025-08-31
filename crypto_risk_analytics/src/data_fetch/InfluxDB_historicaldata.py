import sys, os

# go up one level from the notebook folder
sys.path.append(os.path.abspath("../analysis"))
sys.path.append(os.path.abspath("../config"))

import Mandelbrot #defined/customized functions from Mandelbrot.py
import influxDB_utils as influx
from influxdb_client import Point
from datetime import datetime, timezone

for c in ["BTC","ETH"]:
    MIN, HOUR, DAY, WEEK, MONTH = Mandelbrot.historical_datasets(crypto=c, curr="EUR")
    for INTERVAL in [MIN,HOUR,DAY,WEEK,MONTH]:
        influx.write_dataframe(INTERVAL)