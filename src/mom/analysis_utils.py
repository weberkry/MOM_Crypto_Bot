from mom import influxDB_utils as influx
from mom import Mandelbrot
from influxdb_client import Point
from datetime import datetime, timezone
import pandas as pd
import os

from dotenv import load_dotenv

# Load env from project root
load_dotenv()

#the following entries are expected in the MOM_Crypto_Bot/.env
ASSET = os.getenv("ASSET")
CURRENCY = os.getenv("CURRENCY")

def get_hurst(asset=ASSET, currency=CURRENCY):
    '''
    calculates Minute and Hour hurst with all data up to most recent
    '''
    hurst_list=[]
    for x in ["Minute","Day"]:
        DF = influx.query_returns(asset=asset, interval=x, start=0, field="delta")

        #rescaled range method
        h = Mandelbrot.hurst()

        if x == "Day":
            h.fit(data=DF["delta"], power=8, rolling_window="false")
        #elif x == "Hour":
            #h.fit(data=DF["delta"], power=9, rolling_window="true")
        else:
            h.fit(data=DF["delta"], power=11, rolling_window="true")

        hurst_list.append(h.hurst)

    df = pd.DataFrame({
        "asset": [asset, asset],
        "currency": [currency, currency],
        "interval": ["Minute","Day"],
        "hurst": hurst_list
    })

    df["date"] = datetime.now(timezone.utc)
    df = df.set_index("date")
    influx.write_hurst(df)
    
    return df

def get_hurst_historic(asset=ASSET,currency = CURRENCY, end= influx.get_last_timestamp(measurement="Day", asset=ASSET, currency=CURRENCY).date().isoformat()):
    
    start = influx.get_last_timestamp(measurement="Day", bucket = "Hurst", asset=ASSET, currency=CURRENCY, field="hurst").date().isoformat()
    
    dates = pd.date_range(start=start,    #divide dates for MIN and DAY (DAY data starts at 2015)
                          end=end, 
                          freq="D")
    print("----calculating for dates:")
    print("from:",start," to:", end)
    #print(dates[-1],dates[0])
    
    
    print("----reading Influx data")
    DAY = influx.query_returns(asset=asset, interval="Day", start=0, field="delta")
    print("----Day data complete")
    MIN = influx.query_returns(asset=asset, interval="Minute", start=0, field="delta")
    print("----Minute data complete")
    #HOUR = influx.query_returns(asset=a, interval="Hour", start=0, field="delta")
    #DAY = influx.query_returns(asset=asset, interval="Day", start=0, field="delta")

    
    
    #
    #maybe adjust for minute in min_before
    #
    for day in dates:
        print("calculating Hurst for: ",day)
        min_before = MIN[MIN.index < day.isoformat()]
        #hour_before = HOUR[HOUR.index < day]
        day_before = DAY[DAY.index < day.isoformat()]
        #print("MIN: ",min_before.head())
        #print("DAY: ",day_before.head())
        #rescaled range method
        m = Mandelbrot.hurst()
        m.fit(data=min_before["delta"], power=11, rolling_window="true")
        
        #h = Mandelbrot.hurst()
        #h.fit(data=hour_before["delta"], power=8, rolling_window="true")
        
        d = Mandelbrot.hurst()
        d.fit(data=day_before["delta"], power=8, rolling_window="false")


        MIN_hurst = pd.DataFrame({
            "asset": [asset],
            "currency": [currency],
            "interval": ["Minute"],
            "hurst": [m.hurst],
            "date": [day]})
        MIN_hurst = MIN_hurst.set_index("date")
        
        DAY_hurst = pd.DataFrame({
            "asset": [asset],
            "currency": [currency],
            "interval": ["Day"],
            "hurst": [d.hurst],
            "date": [day]})
        DAY_hurst = DAY_hurst.set_index("date")

        
        influx.write_hurst(MIN_hurst)
        influx.write_hurst(DAY_hurst)
    
    
