import numpy as np
from scipy import stats


def hurst_exponent(ts, max_lag=100):
    """Estimate Hurst exponent using aggregated variances (Rescaled range or variance scaling).
    ts: 1D array-like of prices
    Returns: H in (0,1). H=0.5 random walk, H>0.5 persistent, H<0.5 anti-persistent
    """
    ts = np.asarray(ts, dtype=float)
    if ts.size < 20:
        return np.nan

    # convert to log returns
    returns = np.diff(np.log(ts))
    lags = np.arange(2, min(max_lag, len(returns)//2))
    tau = [np.sqrt(np.std(returns[lag:] - returns[:-lag])**2) for lag in lags]
    # fit log(tau) = m*log(lag) + c
    m, c, r, p, se = stats.linregress(np.log(lags), np.log(tau))
    H = m
    # depending on method, sometimes H = m*0.5 ; keep conservative
    return H


def hill_tail_index(returns, k=50):
    """Estimate tail index (alpha) using Hill estimator on absolute returns.
    returns: 1D array of returns (non-zero)
    k: number of upper order statistics to use
    Returns alpha (tail exponent). heavy tails => small alpha (~2-4 typical for returns)
    """
    x = np.abs(np.asarray(returns))
    x = x[x > 0]
    if x.size < k + 1:
        return np.nan
    x_sorted = np.sort(x)[::-1]
    x_k = x_sorted[:k]
    x_k_plus1 = x_sorted[k]
    hill = (1 / k) * np.sum(np.log(x_k / x_k_plus1))
    alpha = 1 / hill if hill != 0 else np.nan
    return alpha


def fractal_dimension_from_hurst(H):
    # Relationship D = 2 - H for time series graph
    try:
        return 2.0 - float(H)
    except Exception:
        return np.nan
