#!/usr/bin/env python3
"""
Compare rule-based vs KNN+ridge models on held-out data.
This is the key test: which generalizes better to unseen cases?
"""

import json
import os
import sys
import math
import numpy as np
from sklearn.model_selection import KFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))

from calculate_reimbursement_rules import build_features, is_special_cents

def load_data():
    path = os.path.join(SCRIPT_DIR, '..', 'public_cases.json')
    with open(path) as f:
        cases = json.load(f)
    return [(c['input']['trip_duration_days'],
             c['input']['miles_traveled'],
             c['input']['total_receipts_amount'],
             c['expected_output']) for c in cases]

# ─── KNN model (from calculate_reimbursement.py) ──────────────
def knn_predict(days, miles, receipts, train_data):
    """Simplified KNN from the original model."""
    DAY_SCALE, MILE_SCALE, RCPT_SCALE = 14.0, 1400.0, 2600.0

    dists = []
    for td, tm, tr, expected in train_data:
        dd = (days - td) / DAY_SCALE * 2.0
        dm = (miles - tm) / MILE_SCALE
        dr = (receipts - tr) / RCPT_SCALE
        sp_penalty = 0.3 if is_special_cents(receipts) != is_special_cents(tr) else 0.0
        d = math.sqrt(dd*dd + dm*dm + dr*dr) + sp_penalty
        dists.append((d, expected))
    dists.sort()

    if dists[0][0] < 1e-10:
        return dists[0][1]

    K = 15
    total_w, total_v = 0.0, 0.0
    for dist, val in dists[:K]:
        w = 1.0 / (dist ** 2 + 1e-8)
        total_w += w
        total_v += w * val
    return total_v / total_w

# ─── Ridge regression features (from original model) ──────────
def original_regression_features(days, miles, receipts):
    """The 108-feature set from the original model."""
    d, m, r = days, miles, receipts
    mi_pd = m / d
    r_pd = r / d
    sp = 1.0 if is_special_cents(r) else 0.0
    return [
        1, d, m, r,
        d*d, d*d*d, m*m/10000,
        math.log(d+1), math.log(m+1), math.log(r+1),
        math.sqrt(m), math.sqrt(r), math.sqrt(d),
        mi_pd, r_pd, r_pd*r_pd/10000,
        d*r/1000, m*r/100000, d*m/100,
        math.log(m+1)*math.log(r+1),
        min(r, 300), min(max(0, r-300), 300),
        min(max(0, r-600), 600), max(0, r-1200), max(0, r-1800),
        min(m, 100), min(max(0, m-100), 200),
        max(0, m-300), max(0, m-800),
        min(mi_pd, 200), max(0, mi_pd-200),
        1.0 if mi_pd > 150 and d <= 3 else 0.0,
        1.0 if d >= 7 and m > 800 and r > 800 else 0.0,
        1.0 if d >= 7 and m > 600 else 0.0,
        1.0 if 4 <= d <= 6 else 0.0,
        1.0 if 7 <= d <= 9 else 0.0,
        1.0 if d >= 10 else 0.0,
        1.0 if d <= 2 else 0.0,
        1.0 if d == 1 else 0.0,
        d*math.log(m+1), mi_pd*r_pd/10000,
        max(0, d-10), max(0, d-10)*r/1000,
        math.sqrt(d)*math.log(m+1),
        d*math.sqrt(r), math.sqrt(m)*math.sqrt(r),
        m*math.log(r+1)/1000, r*math.log(m+1)/1000,
        d*d*m/10000, d*d*r/10000,
        min(d, 3), min(d, 3)*m/100, min(d, 3)*r/1000,
        1.0 if 6 <= d <= 9 and m > 900 else 0.0,
        (1.0 if 6 <= d <= 9 and m > 900 else 0.0) * m / 1000,
        (1.0 if 6 <= d <= 9 and m > 900 else 0.0) * r / 1000,
        1.0 if d >= 12 and m > 800 else 0.0,
        max(0, d-12)*m/1000,
        1.0 if d >= 12 and m < 200 else 0.0,
        m**3/1e9, mi_pd*mi_pd/100000,
        d*mi_pd/100, r**2/1e7,
        max(0, r-2200), min(max(0, r-300), 300),
        1.0 if r_pd > 300 else 0.0,
        (1.0 if r_pd > 300 else 0.0) * r / 1000,
        1.0 if r_pd < 30 else 0.0,
        (1.0 if r_pd < 30 else 0.0) * d,
        1.0 if d <= 2 and r > 1500 else 0.0,
        (1.0 if d <= 2 and r > 1500 else 0.0) * r / 1000,
        mi_pd*r_pd/10000, r_pd*d/1000,
        sp, sp*r, sp*d, sp*m, sp*r*r/10000, sp*r_pd,
        sp*d*r/1000,
        m*math.log(d+1)/1000,
        1.0 if d >= 5 and m > 700 and r < 1300 else 0.0,
        (1.0 if d >= 5 and m > 700 else 0.0) * m / 1000,
        d*m*r/1e7,
        math.log(d+1)*math.log(m+1)*math.log(r+1),
        min(mi_pd, 150)*d/100,
        max(0, m-500)*d/1000,
        (1.0 if d >= 10 and m > 600 else 0.0) * m / 1000,
        mi_pd*d/100,
        1.0 if d >= 7 and m > 700 and 600 < r < 1400 else 0.0,
        (1.0 if d >= 7 and m > 700 and 600 < r < 1400 else 0.0) * m / 1000,
        (1.0 if d >= 7 and m > 700 and 600 < r < 1400 else 0.0) * r / 1000,
        d*m*math.log(m+1)/10000,
        d*r*math.log(r+1)/100000,
        m*r*d/10000000,
        math.sqrt(d)*math.sqrt(r),
        math.sqrt(d)*math.sqrt(m),
        math.log(d+1)*math.log(r+1),
        math.log(d+1)*math.log(m+1),
        math.sqrt(m)*math.log(r+1)/10,
        math.sqrt(r)*math.log(m+1)/10,
        d*mi_pd*mi_pd/1000000,
        1.0 if 50 < r_pd < 100 else 0.0,
        1.0 if 100 < r_pd < 200 else 0.0,
        1.0 if 200 < r_pd < 300 else 0.0,
        (1.0 if d >= 5 and d <= 8 else 0.0) * m / 1000,
        (1.0 if d >= 9 and d <= 12 else 0.0) * m / 1000,
        (1.0 if d >= 13 else 0.0) * m / 1000,
    ]

def main():
    data = load_data()
    n = len(data)

    # Build feature matrices
    rules_feat_count = len(build_features(5, 500, 1000))
    orig_feat_count = len(original_regression_features(5, 500, 1000))

    X_rules = np.zeros((n, rules_feat_count))
    X_orig = np.zeros((n, orig_feat_count))
    y = np.zeros(n)

    for i, (d, m, r, e) in enumerate(data):
        X_rules[i] = build_features(d, m, r)
        X_orig[i] = original_regression_features(d, m, r)
        y[i] = e

    # Cross-validation comparison
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("5-Fold Cross-Validation Comparison")
    print("=" * 70)

    for name, X in [("Rules (54 features)", X_rules), ("Original Ridge (108 features)", X_orig)]:
        fold_maes = []
        fold_max_errors = []
        all_val_errors = []

        for train_idx, val_idx in kf.split(X):
            # Ridge fit (alpha=1)
            alpha = 1.0
            XtX = X[train_idx].T @ X[train_idx] + alpha * np.eye(X.shape[1])
            Xty = X[train_idx].T @ y[train_idx]
            coeffs = np.linalg.solve(XtX, Xty)
            pred = np.maximum(0, X[val_idx] @ coeffs)
            errors = np.abs(pred - y[val_idx])
            fold_maes.append(errors.mean())
            fold_max_errors.append(errors.max())
            all_val_errors.extend(errors)

        all_val_errors = np.array(all_val_errors)
        print(f"\n{name}:")
        print(f"  CV MAE:       ${np.mean(fold_maes):.2f} ± ${np.std(fold_maes):.2f}")
        print(f"  CV Max Error: ${np.mean(fold_max_errors):.2f}")
        print(f"  Within $10:   {(all_val_errors < 10).sum()}/{n}")
        print(f"  Within $50:   {(all_val_errors < 50).sum()}/{n}")
        print(f"  Within $100:  {(all_val_errors < 100).sum()}/{n}")

    # KNN cross-validation
    print(f"\nKNN (K=15, no regression):")
    fold_maes = []
    all_val_errors = []
    for train_idx, val_idx in kf.split(X_rules):
        train_data = [data[i] for i in train_idx]
        errors = []
        for i in val_idx:
            d, m, r, e = data[i]
            pred = knn_predict(d, m, r, train_data)
            errors.append(abs(pred - e))
        errors = np.array(errors)
        fold_maes.append(errors.mean())
        all_val_errors.extend(errors)
    all_val_errors = np.array(all_val_errors)
    print(f"  CV MAE:       ${np.mean(fold_maes):.2f} ± ${np.std(fold_maes):.2f}")
    print(f"  Within $10:   {(all_val_errors < 10).sum()}/{n}")
    print(f"  Within $50:   {(all_val_errors < 50).sum()}/{n}")
    print(f"  Within $100:  {(all_val_errors < 100).sum()}/{n}")

    # KNN + Ridge hybrid (original approach)
    print(f"\nKNN + Ridge Hybrid (original approach):")
    fold_maes = []
    all_val_errors = []
    for train_idx, val_idx in kf.split(X_orig):
        train_data = [data[i] for i in train_idx]
        # Fit ridge on training
        alpha = 1.0
        XtX = X_orig[train_idx].T @ X_orig[train_idx] + alpha * np.eye(orig_feat_count)
        Xty = X_orig[train_idx].T @ y[train_idx]
        coeffs = np.linalg.solve(XtX, Xty)

        errors = []
        for i in val_idx:
            d, m, r, e = data[i]
            knn_pred = knn_predict(d, m, r, train_data)
            reg_pred = max(0, np.dot(X_orig[i], coeffs))

            # Find nearest distance for blending
            DAY_SCALE, MILE_SCALE, RCPT_SCALE = 14.0, 1400.0, 2600.0
            min_dist = float('inf')
            for td, tm, tr, _ in train_data:
                dd = (d - td) / DAY_SCALE * 2.0
                dm = (m - tm) / MILE_SCALE
                dr = (r - tr) / RCPT_SCALE
                sp = 0.3 if is_special_cents(r) != is_special_cents(tr) else 0.0
                dist = math.sqrt(dd*dd + dm*dm + dr*dr) + sp
                if dist < min_dist:
                    min_dist = dist

            knn_weight = 1.0 / (1.0 + math.exp((min_dist - 0.05) * 60))
            pred = knn_weight * knn_pred + (1 - knn_weight) * reg_pred
            errors.append(abs(pred - e))
        errors = np.array(errors)
        fold_maes.append(errors.mean())
        all_val_errors.extend(errors)
    all_val_errors = np.array(all_val_errors)
    print(f"  CV MAE:       ${np.mean(fold_maes):.2f} ± ${np.std(fold_maes):.2f}")
    print(f"  Within $10:   {(all_val_errors < 10).sum()}/{n}")
    print(f"  Within $50:   {(all_val_errors < 50).sum()}/{n}")
    print(f"  Within $100:  {(all_val_errors < 100).sum()}/{n}")

if __name__ == '__main__':
    main()
