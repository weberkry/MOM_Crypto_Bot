#!/bin/bash

# --- Setup paths ---
PROJECT_DIR=$PROJECT_ROOT
VENV_DIR="$PROJECT_DIR/venv"
ENV_FILE="$PROJECT_DIR/.env"

# --- Load virtualenv ---
source "$VENV_DIR/bin/activate"

# --- Load .env file ---
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "[ERROR] .env file not found: $ENV_FILE"
    exit 1
fi

# --- Setup logging ---
LOG_FILE="${LOG_DIR}/influx_update_$(date +'%Y%m%d').log"
mkdir -p "$LOG_DIR"

# --- Run Python script ---
python "$PROJECT_DIR/src/data_fetch/InfluxDB_update_run.py" >> "$LOG_FILE" 2>&1

