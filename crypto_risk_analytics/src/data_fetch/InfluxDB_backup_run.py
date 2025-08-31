import sys, os

# go up one level from the notebook folder
sys.path.append(os.path.abspath("../analysis"))
sys.path.append(os.path.abspath("../config"))

import pandas as pd
import influxDB_utils as influx  #my file


if __name__ == "__main__":
    influx.backup_csv()