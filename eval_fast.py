#!/usr/bin/env python3
"""Fast evaluation: imports the model directly instead of 1000 subprocess calls."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force reimport by removing cached module
if 'calculate_reimbursement_rules' in sys.modules:
    del sys.modules['calculate_reimbursement_rules']

from simple_rules_model import calculate_reimbursement

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'public_cases.json')) as f:
    cases = json.load(f)

total_error = 0.0
exact = 0
max_error = 0.0

for c in cases:
    d = c['input']['trip_duration_days']
    m = c['input']['miles_traveled']
    r = c['input']['total_receipts_amount']
    e = c['expected_output']
    pred = calculate_reimbursement(d, m, r)
    err = abs(pred - e)
    total_error += err
    if err < 0.01:
        exact += 1
    if err > max_error:
        max_error = err

n = len(cases)
avg_error = total_error / n
score = avg_error * 100 + (n - exact) * 0.1

print(f"score:{score:.4f}")
print(f"avg_error:{avg_error:.4f}")
print(f"exact:{exact}")
print(f"max_error:{max_error:.4f}")
