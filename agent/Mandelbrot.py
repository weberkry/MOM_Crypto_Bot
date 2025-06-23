import numpy as np
import pandas as pd
import requests
from datetime import datetime
import time
import matplotlib.pyplot as plt


#collect historical data via cryptodatadownload

def historical_data(crypto="BTC",
                    curr = "USD",
                    gran= "d"):
    
    #gran = ["minute", "hour", "dailyd"]
    
    if curr == "USD":
        curr = "USDT"

    url_hist_data = f"https://www.cryptodatadownload.com/cdd/Binance_{crypto}{curr}_{gran}.csv"

    print(curr)

    print(url_hist_data)

    hist_data = pd.read_csv(url_hist_data, skiprows=1) 

    hist_data_delta = process_raw_DF(hist_data)
    
    return hist_data_delta


# Processing Output from historical_data() for further analysis
def process_raw_DF(DF):
    #convert Unix timestamp to readable date-time format
    #DF["Date"] = [datetime.utcfromtimestamp(DF["Unix Timestamp"][ts]).strftime('%Y-%m-%d %H:%M:%S') for ts in range(0,len(DF["Unix Timestamp"]))]
    #Calculate deltas for data points
    DF["Delta_High"] = DF["High"].diff()
    DF["Delta_Low"] = DF["Low"].diff()
    
    return DF


def log10(df):
    data = np.array(df).astype(float)
    y = []
    for i in range(0,len(data)):
        x = data[i]
    
        if x > 0:
            y.append(np.log10(x))
        elif x == 0:
            y.append(0) 
        else:
            y.append(-np.log10(np.absolute(x)))
            #print(i,x,y)
            
    return y


##### NO LONGER IN USE --> only relevant for coinbase api, but historical data gets now downloaded via cryptodatadownload.com
#calculates the gran for units in datapoints in historical charts (gran = 1 = 1sec)
def gran_calc(unit="d"):
    if unit == "d": #day
        gran = 60*60*24
    elif unit == "h": #hour
        gran = 60*60
    elif unit == "m": #min
        gran = 60
    else:
        print("no valid unit")
        
    return gran




def calculate_pdf_fits(data, PDF):
    from scipy.stats import norm, cauchy, lognorm, expon, kstest
    #get parameter
    dist = getattr(scipy.stats, PDF)
    params = dist.fit(data)
    
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    
    #fit density
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    ks = kstest(data, 'norm', args=(loc, scale))
    
    if arg:
        pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale)
    else:
        pdf_fitted = dist.pdf(x, loc=loc, scale=scale)
    
    return params, pdf_fitted, x, ks



# calculate hurst exponent

def hurst_exp(DF, interval):
    
    lags = range(1, interval)

    tau = [np.sqrt(np.std(np.subtract(np.array(DF[lag:]), np.array(DF[:-lag])))) for lag in lags]
    
    reg = linregress(np.log(lags),np.log(tau))
    hurst = reg.slope*2
    
    return hurst, tau, lags

def hurst_per_interval(DF, iteration =1, maximum= 1000, gran = "day"):
    
    hurst_per_interval = pd.DataFrame(columns=["Asset","interval","Hurst","gran"])
    Curr = DF["Asset"][1]

    for i in range(2,maximum, iteration):
        
        hurst, tau , lags = hurst_exp(DF["High"], i)
        row = [Curr,i,hurst, DF["gran"][1]]
        #print(row)
        hurst_per_interval.loc[len(hurst_per_interval)] = row
    
    return hurst_per_interval