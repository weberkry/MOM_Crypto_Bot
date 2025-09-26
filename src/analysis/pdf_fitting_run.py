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

def get_pdf(interval = "Minute"):

    DF = influx.query_returns(asset="BTC", interval=interval, start="0",field="value", bucket="PDF")
    
    print(f"PDF fit for {interval} data")

    if len(DF) > 0:
        
        if "cauchy" in DF["pdf"]:
            print("cauchy pdf already calculated")
        else:
            DF_prices = influx.query_returns(asset="BTC", interval=interval, start="0", field="delta")
            analysis.PDF(DF_prices, pdf="cauchy", interval=interval)
        if "gauss" in DF["pdf"]:
            print("gauss pdf already calculated")
        else:
            DF_prices = influx.query_returns(asset="BTC", interval=interval, start="0", field="delta")
            analysis.PDF(DF_prices, pdf="gauss", interval=interval)
    
    else:
        DF_prices = influx.query_returns(asset="BTC", interval=interval, start="0", field="delta")
        #print("PRICES DF ---------------------------------------")
        ##print(DF_prices.head())
        if len(DF_prices) > 50000:
            DF_prices = DF_prices.sample(n=50000, random_state=42)
        print("analyszing PDF cauchy")
        analysis.PDF(DF_prices, pdf="cauchy", interval=interval)
        print("analyszing PDF gauss")
        analysis.PDF(DF_prices, pdf="gauss", interval=interval)
        
        DF = influx.query_returns(asset="BTC", interval=interval, start="0",field="value", bucket="PDF")

    subset = DF[DF["parameter"].isin(["cvm", "p_value"])]
    #print(DF.head())
    
    return subset




if __name__ == "__main__":
    cvm_Min = get_pdf(interval="Minute")
    cvm_Day = get_pdf(interval="Day")
    print("CVM Day: ", cvm_Day)
    print("CVM MIN: ", cvm_Min)


