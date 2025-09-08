#!/bin/bash
# File: /home/weberkry/git/MOM_Crypto_Bot/src/data_fetch/InfluxDB_update_run.sh

# Load environment variables
export $(grep -v '^#' /home/weberkry/git/MOM_Crypto_Bot/.env | xargs)

# Activate virtual environment
source /home/weberkry/git/MOM_Crypto_Bot/venv/bin/activate

# Run Python script
python /home/weberkry/git/MOM_Crypto_Bot/src/data_fetch/InfluxDB_update_run.py
