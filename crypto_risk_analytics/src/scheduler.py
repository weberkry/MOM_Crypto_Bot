import schedule
import time
from .main import run_for_symbol


def schedule_every_15_minutes(symbol='BTC/USDT'):
    schedule.every(15).minutes.do(run_for_symbol, symbol)
    while True:
        schedule.run_pending()
        time.sleep(1)
