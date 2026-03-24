#!/usr/bin/env python3
"""
Extract actual split points from XGBoost to discover the legacy system's thresholds.
"""

import json
import os
import math
import numpy as np
import xgboost as xgb
from collections import Counter

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

FEATURE_NAMES = [
    'days', 'miles', 'receipts',
    'miles_per_day', 'receipts_per_day',
    'special_cents',
    'days_x_miles', 'days_x_receipts', 'miles_x_receipts',
    'log_days', 'log_miles', 'log_receipts',
]

def build_features(d, m, r):
    mi_pd = m / max(d, 1)
    r_pd = r / max(d, 1)
    sp = 1.0 if is_special_cents(r) else 0.0
    return [d, m, r, mi_pd, r_pd, sp, d*m, d*r, m*r,
            math.log(d+1), math.log(m+1), math.log(r+1)]

def extract_splits(model, feature_names):
    """Extract all split points from the XGBoost model trees."""
    splits = {name: [] for name in feature_names}
    trees = model.get_booster().get_dump(dump_format='json')

    import json as j
    def walk_tree(node):
        if 'split' in node:
            feat_name = feature_names[int(node['split'].replace('f', ''))]
            splits[feat_name].append(float(node['split_condition']))
            if 'children' in node:
                for child in node['children']:
                    walk_tree(child)

    for tree_json in trees:
        tree = j.loads(tree_json)
        walk_tree(tree)

    return splits

def main():
    data = load_data()
    n = len(data)

    X = np.zeros((n, len(FEATURE_NAMES)))
    y = np.zeros(n)
    for i, (d, m, r, e) in enumerate(data):
        X[i] = build_features(d, m, r)
        y[i] = e

    # Train a shallow model to find the most important splits
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=5,
        learning_rate=0.1, random_state=42
    )
    model.fit(X, y)

    splits = extract_splits(model, FEATURE_NAMES)

    print("Most common split points per feature:")
    print("=" * 60)
    for name in FEATURE_NAMES:
        if not splits[name]:
            continue
        values = splits[name]
        # Bin similar values and count
        rounded = [round(v, 1) for v in values]
        counts = Counter(rounded)
        top = counts.most_common(15)
        if top:
            vals_str = ', '.join(f'{v}({c})' for v, c in top)
            print(f"\n{name} ({len(values)} total splits):")
            print(f"  Top splits: {vals_str}")
            # Also show histogram
            arr = np.array(values)
            pcts = np.percentile(arr, [10, 25, 50, 75, 90])
            print(f"  Percentiles: p10={pcts[0]:.1f}, p25={pcts[1]:.1f}, "
                  f"p50={pcts[2]:.1f}, p75={pcts[3]:.1f}, p90={pcts[4]:.1f}")

if __name__ == '__main__':
    main()
