#!/usr/bin/env python3
"""
Try fitting separate models per day count to understand per-day structure.
If each day count has its own formula, this will reveal it.
"""

import json
import os
import math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'public_cases.json')

def load_data():
    with open(DATA_PATH) as f:
        cases = json.load(f)
    return [(c['input']['trip_duration_days'],
             c['input']['miles_traveled'],
             c['input']['total_receipts_amount'],
             c['expected_output']) for c in cases]

def is_special_cents(r):
    cents = round(r * 100) % 100
    return cents == 49 or cents == 99

def build_features(miles, receipts):
    """Features for a single day-group model (days are fixed)."""
    m, r = miles, receipts
    sp = 1.0 if is_special_cents(r) else 0.0
    return [
        1.0,                           # intercept
        m,                             # linear miles
        math.log(m + 1),               # log miles
        math.sqrt(m),                  # sqrt miles
        min(m, 100),                   # tier 1
        min(max(0, m - 100), 200),     # tier 2
        min(max(0, m - 300), 500),     # tier 3
        max(0, m - 800),              # tier 4
        r,                             # linear receipts
        math.log(r + 1),               # log receipts
        math.sqrt(r),                  # sqrt receipts
        min(r, 300),                   # tier 1
        min(max(0, r - 300), 300),     # tier 2
        min(max(0, r - 600), 600),     # tier 3
        max(0, r - 1200),             # tier 4
        max(0, r - 1800),             # tier 5
        m * r / 10000,                 # interaction
        math.sqrt(m) * math.sqrt(r),   # sqrt interaction
        math.log(m + 1) * math.log(r + 1),  # log interaction
        sp,                            # bug
        sp * r,                        # bug × receipts
        sp * r * r / 10000,            # bug × receipts²
        sp * m,                        # bug × miles
    ]

def main():
    data = load_data()

    total_error = 0.0
    total_cases = 0
    all_errors = []

    for day_count in range(1, 15):
        subset = [(m, r, e) for d, m, r, e in data if d == day_count]
        if len(subset) < 10:
            continue

        n = len(subset)
        X = np.zeros((n, len(build_features(500, 1000))))
        y = np.zeros(n)

        for i, (m, r, e) in enumerate(subset):
            X[i] = build_features(m, r)
            y[i] = e

        # Ridge regression
        alpha = 1.0
        XtX = X.T @ X + alpha * np.eye(X.shape[1])
        Xty = X.T @ y
        coeffs = np.linalg.solve(XtX, Xty)

        pred = np.maximum(0, X @ coeffs)
        errors = np.abs(pred - y)

        print(f"Day {day_count:2d}: n={n:3d}, MAE=${errors.mean():6.2f}, "
              f"median=${np.median(errors):6.2f}, max=${errors.max():6.2f}, "
              f"<$10={int((errors < 10).sum()):3d}, <$50={int((errors < 50).sum()):3d}")

        total_error += errors.sum()
        total_cases += n
        all_errors.extend(errors)

    all_errors = np.array(all_errors)
    print(f"\nOverall: MAE=${total_error/total_cases:.2f}, "
          f"median=${np.median(all_errors):.2f}, max=${all_errors.max():.2f}")
    print(f"  <$10: {(all_errors < 10).sum()}, <$50: {(all_errors < 50).sum()}, "
          f"<$100: {(all_errors < 100).sum()}")

if __name__ == '__main__':
    main()
