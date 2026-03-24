#!/bin/bash

# Black Box Challenge - Reimbursement Calculator
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>
# No external dependencies required - uses only Python 3 standard library

python3 calculate_reimbursement.py "$1" "$2" "$3"
