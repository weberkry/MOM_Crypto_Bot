import os
from dotenv import load_dotenv

load_dotenv()

# General
#DATA_DIR = os.getenv('DATA_DIR', 'data')

# Exchanges (example: Binance)
#EXCHANGE = os.getenv('EXCHANGE', 'binance')

# InfluxDB
INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")


# MCP / AI
#MCP_API_KEY = os.getenv('MCP_API_KEY', '')
