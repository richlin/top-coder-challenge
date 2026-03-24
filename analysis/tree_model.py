#!/usr/bin/env python3
"""
Gradient boosted tree model to find the error floor and discover breakpoints.
XGBoost can naturally find thresholds, interactions, and non-linearities.
"""

import json
import os
import math
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb

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

def build_features(d, m, r):
    """Minimal features — let the tree find the breakpoints."""
    mi_pd = m / max(d, 1)
    r_pd = r / max(d, 1)
    sp = 1.0 if is_special_cents(r) else 0.0
    return [
        d, m, r,                    # raw inputs
        mi_pd, r_pd,               # key ratios
        sp,                         # bug indicator
        d * m, d * r, m * r,        # interactions
        math.log(d + 1), math.log(m + 1), math.log(r + 1),
    ]

FEATURE_NAMES = [
    'days', 'miles', 'receipts',
    'miles_per_day', 'receipts_per_day',
    'special_cents',
    'days_x_miles', 'days_x_receipts', 'miles_x_receipts',
    'log_days', 'log_miles', 'log_receipts',
]

def main():
    data = load_data()
    n = len(data)

    X = np.zeros((n, len(FEATURE_NAMES)))
    y = np.zeros(n)

    for i, (d, m, r, e) in enumerate(data):
        X[i] = build_features(d, m, r)
        y[i] = e

    # Cross-validation to estimate generalization
    print("Cross-validation (5-fold):")
    for max_depth in [3, 5, 7, 10]:
        for n_est in [100, 500, 1000]:
            model = xgb.XGBRegressor(
                n_estimators=n_est, max_depth=max_depth,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=42,
                objective='reg:absoluteerror'
            )
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in kf.split(X):
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[val_idx])
                mae = np.abs(pred - y[val_idx]).mean()
                scores.append(mae)
            avg_mae = np.mean(scores)
            print(f"  depth={max_depth:2d}, n_est={n_est:4d}: CV MAE=${avg_mae:.2f}")

    # Fit on all data to see training error floor
    print("\nTraining on all data:")
    for max_depth in [3, 5, 7, 10, 15, 20]:
        for n_est in [100, 500, 1000, 2000]:
            model = xgb.XGBRegressor(
                n_estimators=n_est, max_depth=max_depth,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=42
            )
            model.fit(X, y)
            pred = model.predict(X)
            errors = np.abs(pred - y)
            print(f"  depth={max_depth:2d}, n_est={n_est:4d}: "
                  f"train MAE=${errors.mean():.2f}, max=${errors.max():.2f}, "
                  f"<$1={int((errors < 1).sum())}, <$10={int((errors < 10).sum())}")

    # Best model with feature importances
    print("\n\nBest model feature importances:")
    model = xgb.XGBRegressor(
        n_estimators=1000, max_depth=7,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42
    )
    model.fit(X, y)
    importances = model.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        print(f"  {name:20s}: {imp:.4f}")

    # Show worst cases for best model
    pred = model.predict(X)
    errors = np.abs(pred - y)
    worst = np.argsort(errors)[-10:]
    print(f"\nWorst 10 (train, depth=7, n=1000):")
    for idx in worst:
        d, m, r, e = data[idx]
        sp = " *BUG*" if is_special_cents(r) else ""
        print(f"  {d}d, {m:.0f}mi, ${r:.2f} -> expected ${e:.2f}, "
              f"got ${pred[idx]:.2f}, err=${errors[idx]:.2f}{sp}")

if __name__ == '__main__':
    main()
