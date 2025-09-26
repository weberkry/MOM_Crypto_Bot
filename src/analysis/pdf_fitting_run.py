from mom import Mandelbrot #defined/customized functions from Mandelbrot.py
from mom import influxDB_utils as influx
from mom import analysis_utils as analysis

import scipy
from scipy.stats import norm, cauchy, lognorm, expon, kstest

import os
from dotenv import load_dotenv

# Load env from project root
load_dotenv()

#the following entries are expected in the MOM_Crypto_Bot/.env
ASSET = os.getenv("ASSET")
CURRENCY = os.getenv("CURRENCY")


#DF_Day = influx.query_returns(asset=ASSET, interval="Day", start="0", field="delta")
#DF_Min = influx.query_returns(asset=ASSET, interval="Minute", start="0", field="delta")

#DF_Min = DF_Min.sample(n=50000, random_state=42) # Full DF is too big





if __name__ == "__main__":
    cvm_Min = analysis.get_pdf(interval="Minute")
    cvm_Day = analysis.get_pdf(interval="Day")
    print("CVM Day: ", cvm_Day)
    print("CVM MIN: ", cvm_Min)


