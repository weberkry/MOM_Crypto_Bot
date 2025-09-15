import pandas as pd
from mom import influxDB_utils as influx

import sys

y = sys.argv[1]

print("using ", y)



if __name__ == "__main__":
    influx.backup_csv(year = y)
    influx.backup_hurst()