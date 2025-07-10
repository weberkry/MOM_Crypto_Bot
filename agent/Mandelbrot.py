import numpy as np
import pandas as pd
import requests
from datetime import datetime
import time
import matplotlib.pyplot as plt
import scipy


#collect historical data via cryptodatadownload an yfinace and merge them together

def historical_data(crypto="BTC",
                    curr = "USD",
                    gran= "d"):
    
    #gran = ["minute", "hour", "d"]
    gran_LOOKUP_cdd = {
        "d": "d",
        "day": "d",
        "h": "60m",
        "m": "minute"
    }
    
    col = ["unix", "date", "high", "low"]

    #1. collect data from cryptodatadownload

    if curr == "USD":
        curr = "USDT"

    url_cdd = f"https://www.cryptodatadownload.com/cdd/Binance_{crypto}{curr}_{gran_LOOKUP_cdd[gran]}.csv"

    print(f"downloading the {curr} price/{gran} for {crypto} from {url_cdd}....")

    hist_data_cdd = pd.read_csv(url_cdd, skiprows=1) 
    hist_data_cdd.columns = map(str.lower, hist_data_cdd.columns)
    hist_data_cdd["unix"] = hist_data_cdd['unix'] // 1000    #unix is displayed in milliseconds. To match yfinance it must be displayed in sec.
    hist_data_cdd["volume"] = hist_data_cdd[f"volume {crypto.lower()}"]

    #hist_data_delta_cdd = process_raw_DF(hist_data_cdd)
    #print("---cdd---")
    #print(hist_data_cdd.head())

    #2. collect data from yfinance
    import yfinance as yf

    gran_LOOKUP = {
        "d": "1d",
        "h": "60m",
        "m": "1m"
    }

    crypto_chart = yf.Ticker(f"{crypto}-{curr}")
    hist_data_yf = reset_index(crypto_chart.history(period="max", interval=gran_LOOKUP[gran]))

    print(f"downloading the {curr} price/{gran} for {crypto} from yfinance....")


    #print("---yf---")
    #print(hist_data_yf.head())


    #merge cdd and yf
    hist_data = pd.merge(hist_data_yf, hist_data_cdd, on='unix', how='outer').sort_values('unix').reset_index(drop=True)
    hist_data = process_raw_DF(hist_data)
    print("merged yf and cdd")

    return hist_data


def historical_datasets(crypto="BTC",
                        curr = "USD"):
    MIN  = historical_data(crypto=crypto, curr = curr, gran = "m")
    print('-----------------------------DF Tail Minute')
    print(MIN.tail())
    DAY  = historical_data(crypto=crypto, curr = curr, gran = "d")

    #calculate Hour
    HOUR = MIN[MIN.date.dt.minute == 0 ]
    HOUR.loc[:,"delta_high"] = HOUR["high"].diff()
    HOUR.loc[:,"delta_low"] = HOUR["low"].diff()

    #calculate week
    WEEK = DAY[DAY.date.dt.weekday == 0]
    #.loc[row_indexer,col_indexer] 
    WEEK.loc[:,"delta_high"] = WEEK["high"].diff()
    WEEK.loc[:,"delta_low"] = WEEK["low"].diff()

    #calculate month
    MONTH = DAY[DAY.date.dt.is_month_start]
    MONTH.loc[:,"delta_high"] = MONTH["high"].diff()
    MONTH.loc[:,"delta_low"]= MONTH["low"].diff()


    return MIN, HOUR, DAY, WEEK, MONTH 

    


def reset_index(df):
    df = df.reset_index()
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends','stock splits']
    df['unix'] = df['date'].dt.floor('D').astype('int64') // 10**9  #correcting date for midnight
    return df


# Processing Output from historical_data() for further analysis
def process_raw_DF(DF):
    col = ["unix", "date","volume", "high", "low","delta_high","delta_low"]
    print("-----------------------------------------------------------------------")
    print(DF.head())
    # merge different highs and lows for the same timepoint:
    DF['high'] = DF[['high_x', 'high_y']].mean(axis=1, skipna=True)
    DF['low'] = DF[['low_x', 'low_y']].mean(axis=1, skipna=True)
    DF['volume'] = DF[['volume_x','volume_y']].mean(axis=1, skipna=True)

    #convert Unix timestamp to readable date-time format
    DF['date'] = pd.to_datetime(DF['unix'], unit='s')
    #Calculate deltas for data points

    DF["delta_high"] = DF["high"].diff()
    DF["delta_low"] = DF["low"].diff()

    print("-----------------------------------------------------------------------")
    print(DF.head())
    
    return DF[col]


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
##############################################################

def process_csv(gran="min"):
    DF = pd.read_csv(f"{gran}.csv")
    DF.loc[:,'date'] = pd.to_datetime(DF['unix'],unit='s')
    DF = DF[DF['date'] < pd.Timestamp('2024-01-01 00:00:00')] #cutoff ---> incomplete data
    DF['date'] = pd.to_datetime(DF['date'])
    DF['timedelta'] = DF['date'].diff()
    
    return DF


def gauss_pdf(data):
    dist = getattr(scipy.stats, "norm")
    params = dist.fit(data,method="mle")

    #params
    sigma = params[-2]
    mu = params[-1]


    #loglik = np.sum(norm.logpdf(data, *params))
    #aic = aic(loglik, 2)
    #cvm = scipy.stats.cramervonmises(data, 'norm')

    #fit density
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    pdf_fitted = dist.pdf(x, loc=sigma, scale=mu)

    return params, pdf_fitted, x



def cauchy_pdf(data):
    dist = getattr(scipy.stats, "cauchy")
    params = dist.fit(data,method="mle")

    #params
    x0 = params[-2]
    gamma = params[-1]


    #loglik = np.sum(cauchy.logpdf(data, *params))
    #aic = aic(loglik, 2)
    #cvm = scipy.stats.cramervonmises(data, 'cauchy')

    #fit density
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    pdf_fitted = dist.pdf(x, loc=x0, scale=gamma)

    return params, pdf_fitted, x



def aic(loglik, num_params):
    return 2 * num_params - 2 * loglik

def calculate_pdf(data, PDF):
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

    
    
    if arg:
        pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale)
    else:
        pdf_fitted = dist.pdf(x, loc=loc, scale=scale)
    
    return params, pdf_fitted, x

def calculate_pdf_fit(data, params):
    from scipy.stats import norm, cauchy, lognorm, expon, kstest
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    fit_test = []
    
    #kolmogorov-smirnoff test
    
    
    #Cramér von mises Test
    cvm = scipy.stats.cramervonmises(data, 'norm')
    fit_test.append(cvm)

    return fit_test



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