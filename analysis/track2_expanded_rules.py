#!/usr/bin/env python3
"""
Track 2: Fit ridge regression on the exact 108 features from calculate_reimbursement.py.
Compare global model vs per-day-count models.
Print best coefficients for hardcoding into simple_rules_model.py.
"""

import json
import math
import os
import numpy as np

def is_special_cents(r):
    cents = round(r * 100) % 100
    return cents == 49 or cents == 99

def build_features(days, miles, receipts):
    """Exact copy of 108 features from calculate_reimbursement.py lines 59-133."""
    mi_pd = miles / days
    r_pd = receipts / days
    sp = 1.0 if is_special_cents(receipts) else 0.0
    return [
        1, days, miles, receipts,
        days*days, days*days*days, miles*miles/10000,
        math.log(days+1), math.log(miles+1), math.log(receipts+1),
        math.sqrt(miles), math.sqrt(receipts), math.sqrt(days),
        mi_pd, r_pd, r_pd*r_pd/10000,
        days*receipts/1000, miles*receipts/100000, days*miles/100,
        math.log(miles+1)*math.log(receipts+1),
        min(receipts, 300), min(max(0, receipts-300), 300),
        min(max(0, receipts-600), 600), max(0, receipts-1200), max(0, receipts-1800),
        min(miles, 100), min(max(0, miles-100), 200),
        max(0, miles-300), max(0, miles-800),
        min(mi_pd, 200), max(0, mi_pd-200),
        1.0 if mi_pd > 150 and days <= 3 else 0.0,
        1.0 if days >= 7 and miles > 800 and receipts > 800 else 0.0,
        1.0 if days >= 7 and miles > 600 else 0.0,
        1.0 if 4 <= days <= 6 else 0.0,
        1.0 if 7 <= days <= 9 else 0.0,
        1.0 if days >= 10 else 0.0,
        1.0 if days <= 2 else 0.0,
        1.0 if days == 1 else 0.0,
        days*math.log(miles+1), mi_pd*r_pd/10000,
        max(0, days-10), max(0, days-10)*receipts/1000,
        math.sqrt(days)*math.log(miles+1),
        days*math.sqrt(receipts), math.sqrt(miles)*math.sqrt(receipts),
        miles*math.log(receipts+1)/1000, receipts*math.log(miles+1)/1000,
        days*days*miles/10000, days*days*receipts/10000,
        min(days, 3), min(days, 3)*miles/100, min(days, 3)*receipts/1000,
        1.0 if 6 <= days <= 9 and miles > 900 else 0.0,
        (1.0 if 6 <= days <= 9 and miles > 900 else 0.0) * miles / 1000,
        (1.0 if 6 <= days <= 9 and miles > 900 else 0.0) * receipts / 1000,
        1.0 if days >= 12 and miles > 800 else 0.0,
        max(0, days-12)*miles/1000,
        1.0 if days >= 12 and miles < 200 else 0.0,
        miles**3/1e9, mi_pd*mi_pd/100000,
        days*mi_pd/100, receipts**2/1e7,
        max(0, receipts-2200), min(max(0, receipts-300), 300),
        1.0 if r_pd > 300 else 0.0,
        (1.0 if r_pd > 300 else 0.0) * receipts / 1000,
        1.0 if r_pd < 30 else 0.0,
        (1.0 if r_pd < 30 else 0.0) * days,
        1.0 if days <= 2 and receipts > 1500 else 0.0,
        (1.0 if days <= 2 and receipts > 1500 else 0.0) * receipts / 1000,
        mi_pd*r_pd/10000, r_pd*days/1000,
        sp, sp*receipts, sp*days, sp*miles, sp*receipts*receipts/10000, sp*r_pd,
        sp*days*receipts/1000,
        miles*math.log(days+1)/1000,
        1.0 if days >= 5 and miles > 700 and receipts < 1300 else 0.0,
        (1.0 if days >= 5 and miles > 700 else 0.0) * miles / 1000,
        days*miles*receipts/1e7,
        math.log(days+1)*math.log(miles+1)*math.log(receipts+1),
        min(mi_pd, 150)*days/100,
        max(0, miles-500)*days/1000,
        (1.0 if days >= 10 and miles > 600 else 0.0) * miles / 1000,
        mi_pd*days/100,
        1.0 if days >= 7 and miles > 700 and 600 < receipts < 1400 else 0.0,
        (1.0 if days >= 7 and miles > 700 and 600 < receipts < 1400 else 0.0) * miles / 1000,
        (1.0 if days >= 7 and miles > 700 and 600 < receipts < 1400 else 0.0) * receipts / 1000,
        days*miles*math.log(miles+1)/10000,
        days*receipts*math.log(receipts+1)/100000,
        miles*receipts*days/10000000,
        math.sqrt(days)*math.sqrt(receipts),
        math.sqrt(days)*math.sqrt(miles),
        math.log(days+1)*math.log(receipts+1),
        math.log(days+1)*math.log(miles+1),
        math.sqrt(miles)*math.log(receipts+1)/10,
        math.sqrt(receipts)*math.log(miles+1)/10,
        days*mi_pd*mi_pd/1000000,
        1.0 if 50 < r_pd < 100 else 0.0,
        1.0 if 100 < r_pd < 200 else 0.0,
        1.0 if 200 < r_pd < 300 else 0.0,
        (1.0 if days >= 5 and days <= 8 else 0.0) * miles / 1000,
        (1.0 if days >= 9 and days <= 12 else 0.0) * miles / 1000,
        (1.0 if days >= 13 else 0.0) * miles / 1000,
    ]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'public_cases.json')
    with open(data_path) as f:
        cases = json.load(f)

    # Build feature matrix
    X = []
    y = []
    days_list = []
    for c in cases:
        d = c['input']['trip_duration_days']
        m = c['input']['miles_traveled']
        r = c['input']['total_receipts_amount']
        X.append(build_features(d, m, r))
        y.append(c['expected_output'])
        days_list.append(d)

    X = np.array(X)
    y = np.array(y)
    days_arr = np.array(days_list)

    print(f"Dataset: {len(y)} cases, {X.shape[1]} features")
    print()

    # --- Global models with various alpha values ---
    print("=== GLOBAL RIDGE REGRESSION ===")
    alphas = [0, 0.01, 0.1, 1, 10, 100]
    best_global_mae = float('inf')
    best_global_coeffs = None
    best_global_alpha = None

    for alpha in alphas:
        if alpha == 0:
            # OLS
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        else:
            # Ridge: (X^T X + alpha I) ^ -1 X^T y
            XtX = X.T @ X
            Xty = X.T @ y
            I = np.eye(X.shape[1])
            I[0, 0] = 0  # don't regularize intercept
            coeffs = np.linalg.solve(XtX + alpha * I, Xty)

        preds = X @ coeffs
        preds = np.maximum(preds, 0)
        mae = np.mean(np.abs(preds - y))
        print(f"  alpha={alpha:>6}: MAE = ${mae:.2f}")

        if mae < best_global_mae:
            best_global_mae = mae
            best_global_coeffs = coeffs
            best_global_alpha = alpha

    print(f"\n  Best global: alpha={best_global_alpha}, MAE=${best_global_mae:.2f}")

    # --- Per-day models ---
    print("\n=== PER-DAY RIDGE REGRESSION (alpha=10) ===")
    per_day_coeffs = {}
    per_day_preds = np.zeros_like(y)
    total_per_day_error = 0.0

    for day in range(1, 15):
        mask = days_arr == day
        n_day = mask.sum()
        if n_day == 0:
            print(f"  day={day:>2}: no cases")
            continue

        X_day = X[mask]
        y_day = y[mask]

        alpha_day = 10
        XtX = X_day.T @ X_day
        Xty = X_day.T @ y_day
        I = np.eye(X_day.shape[1])
        I[0, 0] = 0
        coeffs = np.linalg.solve(XtX + alpha_day * I, Xty)

        preds_day = np.maximum(X_day @ coeffs, 0)
        mae_day = np.mean(np.abs(preds_day - y_day))
        total_per_day_error += np.sum(np.abs(preds_day - y_day))
        per_day_preds[mask] = preds_day
        per_day_coeffs[day] = coeffs

        print(f"  day={day:>2}: n={n_day:>3}, MAE=${mae_day:.2f}")

    overall_per_day_mae = total_per_day_error / len(y)
    print(f"\n  Overall per-day MAE: ${overall_per_day_mae:.2f}")

    # --- Also try per-day with different alphas ---
    print("\n=== PER-DAY ALPHA SEARCH ===")
    best_pd_alpha = 10
    best_pd_mae = overall_per_day_mae
    for alpha_test in [0.01, 0.1, 1, 5, 10, 50, 100]:
        total_err = 0.0
        for day in range(1, 15):
            mask = days_arr == day
            if mask.sum() == 0:
                continue
            X_day = X[mask]
            y_day = y[mask]
            XtX = X_day.T @ X_day
            Xty = X_day.T @ y_day
            I = np.eye(X_day.shape[1])
            I[0, 0] = 0
            coeffs = np.linalg.solve(XtX + alpha_test * I, Xty)
            preds_day = np.maximum(X_day @ coeffs, 0)
            total_err += np.sum(np.abs(preds_day - y_day))
        mae = total_err / len(y)
        print(f"  alpha={alpha_test:>5}: overall MAE=${mae:.2f}")
        if mae < best_pd_mae:
            best_pd_mae = mae
            best_pd_alpha = alpha_test

    # Refit per-day models with best alpha
    print(f"\n=== REFIT PER-DAY WITH BEST ALPHA={best_pd_alpha} ===")
    per_day_coeffs = {}
    per_day_preds = np.zeros_like(y)
    total_per_day_error = 0.0
    for day in range(1, 15):
        mask = days_arr == day
        if mask.sum() == 0:
            continue
        X_day = X[mask]
        y_day = y[mask]
        XtX = X_day.T @ X_day
        Xty = X_day.T @ y_day
        I = np.eye(X_day.shape[1])
        I[0, 0] = 0
        coeffs = np.linalg.solve(XtX + best_pd_alpha * I, Xty)
        preds_day = np.maximum(X_day @ coeffs, 0)
        mae_day = np.mean(np.abs(preds_day - y_day))
        total_per_day_error += np.sum(np.abs(preds_day - y_day))
        per_day_preds[mask] = preds_day
        per_day_coeffs[day] = coeffs
        print(f"  day={day:>2}: n={mask.sum():>3}, MAE=${mae_day:.2f}")
    overall_per_day_mae = total_per_day_error / len(y)
    print(f"  Overall: MAE=${overall_per_day_mae:.2f}")

    # --- Comparison ---
    print("\n=== COMPARISON ===")
    print(f"  Global model (alpha={best_global_alpha}):  MAE=${best_global_mae:.2f}")
    print(f"  Per-day models (alpha={best_pd_alpha}):  MAE=${overall_per_day_mae:.2f}")

    # Decide which to use
    if overall_per_day_mae < best_global_mae - 2.0:
        print("\n  >>> Per-day models are significantly better. Using 14 separate models.")
        use_per_day = True
    else:
        print(f"\n  >>> Global model is comparable or better. Using single global model.")
        use_per_day = False

    # --- Print coefficients ---
    print("\n=== COEFFICIENTS FOR HARDCODING ===")
    if use_per_day:
        print("\n# Per-day coefficient arrays")
        print("PER_DAY_COEFFS = {")
        for day in range(1, 15):
            if day in per_day_coeffs:
                c = per_day_coeffs[day]
                coeffs_str = ", ".join(f"{v:.10f}" for v in c)
                print(f"    {day}: [{coeffs_str}],")
        print("}")
    else:
        print(f"\n# Global coefficients (alpha={best_global_alpha})")
        coeffs_str = ", ".join(f"{v:.10f}" for v in best_global_coeffs)
        print(f"COEFFS = [{coeffs_str}]")

    # --- Compute eval_fast.py equivalent score ---
    preds_final = per_day_preds if use_per_day else np.maximum(X @ best_global_coeffs, 0)

    errors = np.abs(preds_final - y)
    avg_error = np.mean(errors)
    exact = np.sum(errors < 0.01)
    score = avg_error * 100 + (len(y) - exact) * 0.1
    print(f"\n=== ESTIMATED SCORE ===")
    print(f"  avg_error: ${avg_error:.4f}")
    print(f"  exact: {exact}")
    print(f"  score: {score:.4f}")


if __name__ == '__main__':
    main()
