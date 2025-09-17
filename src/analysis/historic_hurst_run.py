import pandas as pd
from mom import influxDB_utils as influx
from mom import analysis_utils as analysis
from mom import Mandelbrot

import os

from dotenv import load_dotenv

# Load env from project root
load_dotenv()

#the following entries are expected in the MOM_Crypto_Bot/.env
ASSET = os.getenv("ASSET")
CURRENCY = os.getenv("CURRENCY")


if __name__ == "__main__":
        analysis.get_hurst_historic(asset=ASSET,currency = CURRENCY)
        #print(influx.get_last_timestamp(measurement="Day", bucket = "Hurst", asset=ASSET, currency=CURRENCY, field="hurst").date().isoformat())