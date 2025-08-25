import sys
import time
from pathlib import Path

from config import settings
from data_fetch.market_data import fetch_ohlcv
from analysis.fractal_risk import hurst_exponent, hill_tail_index, fractal_dimension_from_hurst
from ai_agents.mcp_agent import MCPAgent
from storage.influxdb_writer import write_risk_metrics, write_sentiment
import numpy as np


def run_for_symbol(symbol: str):
    print(f'Running pipeline for {symbol}')
    # 1) Fetch hourly data as an example
    df = fetch_ohlcv(symbol, timeframe='1h', limit=1000)
    if df is None or df.empty:
        print('No market data; aborting')
        return

    close = df['close'].values

    # 2) Compute risk metrics
    H = hurst_exponent(close)
    returns = np.diff(np.log(close))
    alpha = hill_tail_index(returns, k=100)
    D = fractal_dimension_from_hurst(H)

    metrics = {
        'hurst': float(H) if H is not None else None,
        'tail_alpha': float(alpha) if alpha is not None else None,
        'fractal_dimension': float(D) if D is not None else None,
    }

    write_risk_metrics(symbol, metrics)

    # 3) Run MCP agent
    agent = MCPAgent(api_key=settings.MCP_API_KEY)
    ai_out = agent.run_pipeline(symbol)
    write_sentiment(symbol, ai_out.get('sentiment_score', 0.0), ai_out.get('summary', ''))


if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTC/USDT'
    run_for_symbol(symbol)
