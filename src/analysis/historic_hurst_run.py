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

import sys

y = sys.argv[1]

endyear = int(y)+1

end = f"{endyear}-01-01"

print("using ", end)


if __name__ == "__main__":
    if len(end)>0:
        analysis.get_hurst_historic(asset=ASSET,currency = CURRENCY, end=end)
    else:
        analysis.get_hurst_historic(asset=ASSET,currency = CURRENCY)