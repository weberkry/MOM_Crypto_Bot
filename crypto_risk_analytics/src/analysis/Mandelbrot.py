import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone

import time
import matplotlib.pyplot as plt
import scipy
from scipy.stats import linregress



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
    hist_data["currency"] = curr
    hist_data["asset"] = crypto
    #hist_data = hist_data.set_index("date")
    print("merged yf and cdd")

    return hist_data


def historical_datasets(crypto="BTC",
                        curr = "USD"):
    MIN  = historical_data(crypto=crypto, curr = curr, gran = "m")
    #print('-----------------------------DF Tail Minute')
    #print(MIN.tail())
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

    MIN["interval"] = "Minute"
    MIN = MIN.set_index("date")
    HOUR["interval"] = "Hour"
    HOUR = HOUR.set_index("date")
    DAY["interval"] = "Day"
    DAY = DAY.set_index("date")
    WEEK["interval"] = "Week"
    WEEK = WEEK.set_index("date")
    MONTH["interval"] = "Month"
    MONTH = MONTH.set_index("date")

    return MIN, HOUR, DAY, WEEK, MONTH 

    


def reset_index(df):
    df = df.reset_index()
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends','stock splits']
    df['unix'] = df['date'].dt.floor('D').astype('int64') // 10**9  #correcting date for midnight
    return df

#def mean_xy(df)


# Processing Output from historical_data() for further analysis
def process_raw_DF(DF):
    from datetime import datetime, timezone
    col = ["unix", "date","volume", "high","low","delta","delta_log","return","return_log"]
    #print("-----------------------------------------------------------------------")
    #print(DF.head())
    if "high_x" in DF.columns:
    # merge different highs and lows for the same timepoint:
        DF['high'] = DF[['high_x', 'high_y']].mean(axis=1, skipna=True)
        DF['low'] = DF[['low_x', 'low_y']].mean(axis=1, skipna=True)
        DF['volume'] = DF[['volume_x','volume_y']].mean(axis=1, skipna=True)

    #convert Unix timestamp to readable date-time format
    DF['date'] = pd.to_datetime(DF['unix'], unit='s')
    rfc3339 = []
    for i in DF["unix"]:
        dt = datetime.fromtimestamp(i, tz= timezone.utc)
        date = dt.isoformat().replace("+00:00","Z")
        rfc3339.append(date)
    DF["date_rfc"] = rfc3339

    #Calculate deltas for data points

    DF["delta"] = DF["high"].diff()
    DF["delta_log"] = log10(DF["delta"])
    DF["return"] = compute_returns(DF["high"])[0]
    DF["return_log"] = compute_returns(DF["high"])[0]

    #print("-----------------------------------------------------------------------")
    #print(DF.head())
    
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

def compute_returns(series):
    return series.pct_change(), np.log(series / series.shift(1)) #simple_return,log return
  


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



class hurst:
    def __init__(self):
        self.hurst = None
        self.Y = None
        self.linfit = None
        self.rs_range = None
        self.subset_length = None

    def rescaled_range(self, data, power=9, rolling_window=False):
        #the len should be at least 512 /power =9
        l = 2 ** power  # length of each subset
        #l = power
        n = int(len(data) / l)  # number of subsets
        R_S_per_segment = []

        if rolling_window == True:
            print(f"Data gets divided into {len(data) - l + 1} rolling windows of length {l}")
            for k in range(l, len(data) + 1):
                subset = data[k - l:k]
                mean = np.mean(subset)
                std = np.std(subset)
                Y_cumsum = np.cumsum(subset - mean)
                R = np.max(Y_cumsum) - np.min(Y_cumsum)
                R_S = R / std if std != 0 else 0
                R_S_per_segment.append(R_S)
        else:
            print(f"Data gets divided into {n} fixed windows of length {l}")
            for k in range(n):
                subset = data[k * l:(k + 1) * l]
                mean = np.mean(subset)
                std = np.std(subset)
                Y_cumsum = np.cumsum(subset - mean)
                R = np.max(Y_cumsum) - np.min(Y_cumsum)
                R_S = R / std if std != 0 else 0
                R_S_per_segment.append(R_S)

        R_S_mean = np.mean(R_S_per_segment)
        return R_S_mean, l

    def fit(self, data, power, rolling_window=False):
        L = []
        R_S_mean_list = []
        
        #loop through different lengths relating to the base 2
        for p in range(6, power + 1):
            r_s_mean, l = self.rescaled_range(data=data, power=p, rolling_window=rolling_window)
            L.append(l)
            R_S_mean_list.append(r_s_mean)

        #fit linear regression
        slope, intercept, r_value, p_value, std_err = linregress(np.log(L), np.log(R_S_mean_list))

        #reconstruct fitted values
        Y = [slope * np.log(l) + intercept for l in L]

        #store values in the instance
        self.hurst = slope
        self.Y = Y
        self.linfit = [slope, intercept, p_value]
        self.rs_range = R_S_mean_list
        self.subset_length = L

        return slope  #hurst
    
    def plot_fit(self):
        plt.scatter(np.log(self.subset_length),np.log(self.rs_range))
        plt.plot(np.log(self.subset_length), self.Y, label=f'linear Fit: Hurst={self.linfit[0]:.3f}', color='red')
        #plt.text(x=0.9,y=0.1,s=f"Y={H}*x+{fit[1]}")
        plt.ylabel("log(R/S)")
        plt.xlabel("log(length)")
        plt.title('Hurst Exponent via Rescaled Range Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()
        