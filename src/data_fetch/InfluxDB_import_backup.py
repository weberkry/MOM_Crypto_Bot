import pandas as pd
from mom import influxDB_utils as influx


if __name__ == "__main__":
    influx.write_from_backup()