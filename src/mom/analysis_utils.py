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
    
def PDF(DF, pdf="cauchy", interval="Minute"):
    
    if pdf == "cauchy":
        params, pdf_fitted, x = Mandelbrot.cauchy_pdf(DF["delta"])
        cvm = Mandelbrot.calculate_pdf_fit(DF["delta"],params, pdf)
    

    elif pdf == "gauss":
        params, pdf_fitted, x = Mandelbrot.gauss_pdf(DF["delta"])
        cvm = Mandelbrot.calculate_pdf_fit(DF["delta"], params, "norm")
        
    else:
        print("pick a Probability density function")
        
    #arg = params[:-2]
    loc = params[-2]   #sigma (variance) for gauss/ #x0 (peak value) for cauchy
    scale = params[-1] #mu (mean) for gauss / 
                       #gamma (the distance from the peak x0 where the PDF drops to half its maximum value) for cauchy

        
    #x values for pdf fit
    df_x_values = Mandelbrot.list_to_df_with_dates(x,start_date="1970-01-01", parameter="x_values")
    df_x_values["pdf"] = pdf
    df_x_values["interval"] = interval
    influx.write_dataframe(df_x_values, in_bucket="PDF")
    
    #y values for df fit
    df_y_values = Mandelbrot.list_to_df_with_dates(pdf_fitted,start_date="1970-01-01", parameter="pdf_values")
    df_y_values["pdf"] = pdf
    df_y_values["interval"] = interval
    influx.write_dataframe(df_y_values, in_bucket="PDF")
    
    # p value of cvm statistic
    df_p_value = Mandelbrot.list_to_df_with_dates([float(cvm[0].pvalue)],start_date="1970-01-01", parameter="p_value")
    #print([float(cvm[0].pvalue)])
    df_p_value["pdf"] = pdf
    df_p_value["interval"] = interval
    influx.write_dataframe(df_p_value, in_bucket="PDF")
    
    #cvm statisitc
    df_cvm = Mandelbrot.list_to_df_with_dates([cvm[0].statistic],start_date="1970-01-01", parameter="cvm")
    df_cvm["pdf"] = pdf
    df_cvm["interval"] = interval
    influx.write_dataframe(df_cvm, in_bucket="PDF")
    
    #pdf loc
    #sigma (variance) for gauss
    #x0 (peak value) for cauchy
    df_loc = Mandelbrot.list_to_df_with_dates([float(loc)],start_date="1970-01-01", parameter="loc")
    df_loc["pdf"] = pdf
    df_loc["interval"] = interval
    influx.write_dataframe(df_loc, in_bucket="PDF")
    
    #pdf scale
    #mu (mean) for gauss / 
    #gamma (the distance from the peak x0 where the PDF drops to half its maximum value) for cauchy
    df_scale = Mandelbrot.list_to_df_with_dates([float(scale)],start_date="1970-01-01", parameter="scale")
    df_scale["pdf"] = pdf
    df_scale["interval"] = interval
    influx.write_dataframe(df_scale, in_bucket="PDF")

    return cvm


def pdf_fit_return(DF):
    gauss_params, gauss_pdf_fitted, gauss_x = Mandelbrot.gauss_pdf(DF["delta"])
    cauchy_params, cauchy_pdf_fitted, cauchy_x = Mandelbrot.cauchy_pdf(DF["delta"])
    cauchy_cvm = Mandelbrot.calculate_pdf_fit(DF["delta"],cauchy_params, "cauchy")
    gauss_cvm = Mandelbrot.calculate_pdf_fit(DF["delta"], gauss_params, "norm")

    cvm = [gauss_cvm, cauchy_cvm]

    return cvm

def categorize_hurst(h: float):
    if h is None:
        return "unknown"
    if h < 0.45:
        return "high_volatility"
    elif h < 0.55:
        return "random_walk"
    else:
        return "trending"
    

def get_pdf(interval = "Minute"):

    DF = influx.query_returns(asset="BTC", interval=interval, start="0",field="value", bucket="PDF")
    
    print(f"PDF fit for {interval} data")

    if len(DF) > 0:
        
        if "cauchy" in DF["pdf"].values:
            print("cauchy pdf already calculated")
        else:
            DF_prices = influx.query_returns(asset="BTC", interval=interval, start="0", field="delta")
            PDF(DF_prices, pdf="cauchy", interval=interval)
        if "gauss" in DF["pdf"].values:
            print("gauss pdf already calculated")
        else:
            DF_prices = influx.query_returns(asset="BTC", interval=interval, start="0", field="delta")
            PDF(DF_prices, pdf="gauss", interval=interval)
    
    else:
        DF_prices = influx.query_returns(asset="BTC", interval=interval, start="0", field="delta")
        #print("PRICES DF ---------------------------------------")
        ##print(DF_prices.head())
        if len(DF_prices) > 50000:
            DF_prices = DF_prices.sample(n=30000, random_state=42)
        print("analyszing PDF cauchy")
        PDF(DF_prices, pdf="cauchy", interval=interval)
        print("analyszing PDF gauss")
        PDF(DF_prices, pdf="gauss", interval=interval)
        
        DF = influx.query_returns(asset="BTC", interval=interval, start="0",field="value", bucket="PDF")

    subset = DF[DF["parameter"].isin(["cvm", "p_value"])]
    #print(DF.head())
    
    return subset

    
