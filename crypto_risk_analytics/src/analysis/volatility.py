# Simple volatility helpers

import numpy as np


def realized_volatility(returns):
    return np.std(returns, ddof=1) * np.sqrt(len(returns))
