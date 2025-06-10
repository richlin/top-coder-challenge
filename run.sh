#!/bin/bash

# Black Box Challenge - Your Implementation
# This script should take three parameters and output the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed"
    exit 1
fi

# Check if required Python packages are installed
if ! python3 -c "import numpy, pandas, sklearn" &> /dev/null; then
    echo "Installing required Python packages..."
    pip3 install -r requirements.txt
fi

# Run the Python script with the provided arguments and output only the result
python3 calculate_reimbursement.py "$1" "$2" "$3"
