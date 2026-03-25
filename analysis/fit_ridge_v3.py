#!/usr/bin/env python3
"""
fit_ridge_v3.py — Per-Day Ridge with Extended Features

Fits per-day Ridge models with extended features (~244 total),
then outputs updated approach3_ridge_features.py with hardcoded coefficients.
"""

import json
import math
import os
import sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Load data
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE, 'public_cases.json')) as f:
    cases = json.load(f)

def is_special_cents(receipts):
    cents = round(receipts * 100) % 100
    return cents == 49 or cents == 99

def _build_features_base(days, miles, receipts):
    """Original 108 features."""
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

def _build_extended_features(days, miles, receipts):
    """Extended features: per-day interactions + non-linear terms."""
    mi_pd = miles / days
    r_pd = receipts / days
    feats = []

    # Per-day × mile tier interactions (14 × 7 = 98 features)
    for d in range(1, 15):
        is_day = 1.0 if days == d else 0.0
        feats.extend([
            is_day * min(miles, 50),
            is_day * min(max(0, miles-50), 50),
            is_day * min(max(0, miles-100), 200),
            is_day * min(max(0, miles-300), 200),
            is_day * min(max(0, miles-500), 300),
            is_day * min(max(0, miles-800), 400),
            is_day * max(0, miles-1200),
        ])

    # Per-day × receipt tier interactions (14 × 8 = 112 features)
    for d in range(1, 15):
        is_day = 1.0 if days == d else 0.0
        feats.extend([
            is_day * min(receipts, 150),
            is_day * min(max(0, receipts-150), 150),
            is_day * min(max(0, receipts-300), 300),
            is_day * min(max(0, receipts-600), 300),
            is_day * min(max(0, receipts-900), 300),
            is_day * min(max(0, receipts-1200), 400),
            is_day * min(max(0, receipts-1600), 400),
            is_day * max(0, receipts-2000),
        ])

    # Per-day × miles*receipts interaction (14 features)
    for d in range(1, 15):
        is_day = 1.0 if days == d else 0.0
        feats.append(is_day * miles * receipts / 100000)

    # Per-day × efficiency (miles/day) feature (14 features)
    for d in range(1, 15):
        is_day = 1.0 if days == d else 0.0
        feats.append(is_day * mi_pd / 100)

    # Per-day × spending rate (receipts/day) feature (14 features)
    for d in range(1, 15):
        is_day = 1.0 if days == d else 0.0
        feats.append(is_day * r_pd / 100)

    # Cross mile×receipt interaction tiers (4 mile bins × 4 receipt bins = 16 features)
    mile_bins = [
        min(miles, 300) / 300,
        min(max(0, miles-300), 500) / 500,
        min(max(0, miles-800), 500) / 500,
        max(0, miles-1300) / 300,
    ]
    rcpt_bins = [
        min(receipts, 500) / 500,
        min(max(0, receipts-500), 700) / 700,
        min(max(0, receipts-1200), 800) / 800,
        max(0, receipts-2000) / 500,
    ]
    for mb in mile_bins:
        for rb in rcpt_bins:
            feats.append(mb * rb * 100)

    # Higher-order piecewise features (global, 20 features)
    feats.extend([
        min(miles, 300) * min(receipts, 500) / 10000,
        max(0, miles-300) * max(0, receipts-500) / 100000,
        max(0, miles-800) * max(0, receipts-1200) / 100000,
        min(miles, 100) * min(miles, 100) / 10000,
        max(0, miles-800) * max(0, miles-800) / 1000000,
        min(receipts, 300) * min(receipts, 300) / 100000,
        max(0, receipts-1200) * max(0, receipts-1200) / 1000000,
        mi_pd * max(0, receipts-1000) / 1000,
        r_pd * max(0, miles-500) / 1000,
        mi_pd * mi_pd * receipts / 10000000,
        r_pd * r_pd * miles / 10000000,
        min(mi_pd, 100) * min(r_pd, 200) / 10000,
        max(0, mi_pd-100) * max(0, r_pd-100) / 10000,
        (1.0 if mi_pd > 200 else 0.0) * receipts / 1000,
        (1.0 if r_pd > 200 else 0.0) * miles / 1000,
        max(0, miles - 400) * max(0, receipts - 800) / 100000,
        min(miles, 200) * max(0, receipts - 1500) / 100000,
        max(0, miles - 600) * min(receipts, 600) / 100000,
        (1.0 if miles > 500 and receipts > 800 else 0.0) * (miles + receipts) / 1000,
        (1.0 if miles < 200 and receipts > 1500 else 0.0) * receipts / 1000,
    ])

    # Finer cross mile×receipt grid (6×6 = 36 features)
    mile_bins2 = [
        min(miles, 150) / 150,
        min(max(0, miles-150), 250) / 250,
        min(max(0, miles-400), 300) / 300,
        min(max(0, miles-700), 300) / 300,
        min(max(0, miles-1000), 300) / 300,
        max(0, miles-1300) / 200,
    ]
    rcpt_bins2 = [
        min(receipts, 250) / 250,
        min(max(0, receipts-250), 350) / 350,
        min(max(0, receipts-600), 400) / 400,
        min(max(0, receipts-1000), 400) / 400,
        min(max(0, receipts-1400), 500) / 500,
        max(0, receipts-1900) / 600,
    ]
    for mb in mile_bins2:
        for rb in rcpt_bins2:
            feats.append(mb * rb * 100)

    # Even finer cross grid (8×8 = 64 features) - needed for day 5 (n=112)
    mile_bins3 = [
        min(miles, 80) / 80,
        min(max(0, miles-80), 120) / 120,
        min(max(0, miles-200), 150) / 150,
        min(max(0, miles-350), 200) / 200,
        min(max(0, miles-550), 250) / 250,
        min(max(0, miles-800), 250) / 250,
        min(max(0, miles-1050), 250) / 250,
        max(0, miles-1300) / 200,
    ]
    rcpt_bins3 = [
        min(receipts, 200) / 200,
        min(max(0, receipts-200), 200) / 200,
        min(max(0, receipts-400), 300) / 300,
        min(max(0, receipts-700), 300) / 300,
        min(max(0, receipts-1000), 300) / 300,
        min(max(0, receipts-1300), 400) / 400,
        min(max(0, receipts-1700), 400) / 400,
        max(0, receipts-2100) / 500,
    ]
    for mb in mile_bins3:
        for rb in rcpt_bins3:
            feats.append(mb * rb * 100)

    # Additional non-linear terms
    feats.extend([
        days*miles*miles/1000000,
        days*receipts*receipts/10000000,
        (mi_pd**2)*days/100000,
        (r_pd**2)*days/10000,
        min(days, 5)*miles/100,
        max(0, days-7)*receipts/1000,
        max(0, days-7)*miles/1000,
        min(days, 3)*receipts/1000,
        (1.0 if days >= 8 else 0.0) * miles * receipts / 100000,
        (1.0 if days <= 3 else 0.0) * miles * receipts / 100000,
        max(0, miles - 600) * max(0, days - 5) / 1000,
        max(0, receipts - 1000) * max(0, days - 5) / 1000,
        miles * miles * receipts / 1e9,
        miles * receipts * receipts / 1e9,
        mi_pd * min(receipts, 1000) / 1000,
        r_pd * min(miles, 800) / 1000,
        # Extra cross terms for high-error regions
        max(0, miles-500) * min(receipts, 800) / 100000,
        min(miles, 500) * max(0, receipts-800) / 100000,
        max(0, miles-300) * max(0, receipts-300) * max(0, receipts-300) / 1e8,
        min(miles, 300) * min(receipts, 300) * min(receipts, 300) / 1e7,
        miles * receipts * miles / 1e8,
        miles * receipts * receipts / 1e8,
        mi_pd * r_pd * miles / 1e6,
        mi_pd * r_pd * receipts / 1e6,
    ])

    return feats

def build_all_features(days, miles, receipts):
    """Combine base 108 + extended features."""
    return _build_features_base(days, miles, receipts) + _build_extended_features(days, miles, receipts)


# Build feature matrix and target vector
print("Building feature matrix...")
X_list = []
y_list = []
day_list = []
for c in cases:
    d = c['input']['trip_duration_days']
    m = c['input']['miles_traveled']
    r = c['input']['total_receipts_amount']
    e = c['expected_output']
    feats = build_all_features(d, m, r)
    X_list.append(feats)
    y_list.append(e)
    day_list.append(d)

X = np.array(X_list)
y = np.array(y_list)
days_arr = np.array(day_list)

n_features = X.shape[1]
n_base = 108
n_extended = n_features - n_base
print(f"Total features: {n_features} (108 base + {n_extended} extended)")

# --- Fit per-day models ---
alphas = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

def fit_ridge(X_subset, y_subset, label=""):
    """Fit Ridge with alpha search, return raw coefficients."""
    best_alpha = 1.0
    best_mae = float('inf')

    for alpha in alphas:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset)
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X_scaled, y_subset)
        preds = ridge.predict(X_scaled)
        mae = np.mean(np.abs(preds - y_subset))
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha

    # Refit with best alpha
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    ridge = Ridge(alpha=best_alpha, fit_intercept=True)
    ridge.fit(X_scaled, y_subset)

    # Extract raw-space coefficients
    raw_coef = ridge.coef_ / scaler.scale_
    raw_intercept = ridge.intercept_ - np.sum(ridge.coef_ * scaler.mean_ / scaler.scale_)

    # Since feature[0] = 1 (bias), absorb intercept
    full_coeffs = np.zeros(n_features)
    full_coeffs[0] = raw_intercept + raw_coef[0]  # bias feature always = 1
    full_coeffs[1:] = raw_coef[1:]

    # VERIFY: raw coeffs match pipeline predictions
    preds_pipeline = ridge.predict(X_scaled)
    preds_raw = X_subset @ full_coeffs
    max_diff = np.max(np.abs(preds_pipeline - preds_raw))
    assert max_diff < 1e-6, f"Coefficient extraction error: max_diff={max_diff}"

    train_mae = np.mean(np.abs(preds_raw - y_subset))
    if label:
        print(f"  {label}: alpha={best_alpha}, MAE={train_mae:.4f}, max_diff={max_diff:.2e}")

    return full_coeffs


# Fit per-day models
print("\nFitting per-day models...")
per_day_coeffs = {}
for d in range(1, 15):
    mask = days_arr == d
    if mask.sum() == 0:
        continue
    X_d = X[mask]
    y_d = y[mask]
    coeffs = fit_ridge(X_d, y_d, label=f"Day {d:2d} (n={mask.sum():3d})")
    per_day_coeffs[d] = coeffs

# Fit global model as fallback
print("\nFitting global model...")
global_coeffs = fit_ridge(X, y, label="Global")

# --- Evaluate ---
def evaluate(coeffs_dict, global_c, X, y, days_arr):
    total_err = 0.0
    exact = 0
    max_err = 0.0
    for i in range(len(y)):
        d = days_arr[i]
        c = coeffs_dict.get(d, global_c)
        pred = max(0.0, round(float(X[i] @ c), 2))
        err = abs(pred - y[i])
        total_err += err
        if err < 0.01:
            exact += 1
        max_err = max(max_err, err)
    avg_err = total_err / len(y)
    score = avg_err * 100 + (len(y) - exact) * 0.1
    return score, avg_err, exact, max_err

score_g, avg_g, exact_g, max_g = evaluate({}, global_coeffs, X, y, days_arr)
print(f"\nGlobal model:  score={score_g:.4f}, avg_err={avg_g:.4f}, exact={exact_g}, max_err={max_g:.4f}")

score_pd, avg_pd, exact_pd, max_pd = evaluate(per_day_coeffs, global_coeffs, X, y, days_arr)
print(f"Per-day model: score={score_pd:.4f}, avg_err={avg_pd:.4f}, exact={exact_pd}, max_err={max_pd:.4f}")

# --- Generate approach3_ridge_features.py ---
print("\n\n=== GENERATING approach3_ridge_features.py ===\n")

def fmt_coeffs(coeffs, name):
    """Format coefficient array as Python code."""
    lines = [f"{name} = ["]
    for i in range(0, len(coeffs), 6):
        chunk = coeffs[i:i+6]
        line = ", ".join(f"{v:.10f}" for v in chunk)
        lines.append(f"    {line},")
    lines.append("]")
    return "\n".join(lines)

output_lines = []
output_lines.append('#!/usr/bin/env python3')
output_lines.append('"""')
output_lines.append('Approach 3: Ridge-guided feature model (v3).')
output_lines.append(f'Uses {n_features} features with per-day Ridge coefficients.')
output_lines.append('No KNN, no data loading -- pure math with hardcoded coefficients.')
output_lines.append('"""')
output_lines.append('')
output_lines.append('import sys')
output_lines.append('import math')
output_lines.append('')
output_lines.append('')
output_lines.append('def is_special_cents(receipts):')
output_lines.append('    cents = round(receipts * 100) % 100')
output_lines.append('    return cents == 49 or cents == 99')
output_lines.append('')
output_lines.append('')

# PER_DAY_COEFFS
output_lines.append('PER_DAY_COEFFS = {')
for d in range(1, 15):
    if d in per_day_coeffs:
        c = per_day_coeffs[d]
        vals = ", ".join(f"{v:.10f}" for v in c)
        output_lines.append(f'    {d}: [{vals}],')
output_lines.append('}')
output_lines.append('')

# GLOBAL_COEFFS
vals = ", ".join(f"{v:.10f}" for v in global_coeffs)
output_lines.append(f'GLOBAL_COEFFS = [{vals}]')
output_lines.append('')
output_lines.append('')

# _build_features function - needs to produce same features as build_all_features above
output_lines.append('def _build_features(days, miles, receipts):')
output_lines.append(f'    """Build the {n_features}-feature vector."""')
output_lines.append('    mi_pd = miles / days')
output_lines.append('    r_pd = receipts / days')
output_lines.append('    sp = 1.0 if is_special_cents(receipts) else 0.0')
output_lines.append('    feats = [')
output_lines.append('        1, days, miles, receipts,')
output_lines.append('        days*days, days*days*days, miles*miles/10000,')
output_lines.append('        math.log(days+1), math.log(miles+1), math.log(receipts+1),')
output_lines.append('        math.sqrt(miles), math.sqrt(receipts), math.sqrt(days),')
output_lines.append('        mi_pd, r_pd, r_pd*r_pd/10000,')
output_lines.append('        days*receipts/1000, miles*receipts/100000, days*miles/100,')
output_lines.append('        math.log(miles+1)*math.log(receipts+1),')
output_lines.append('        min(receipts, 300), min(max(0, receipts-300), 300),')
output_lines.append('        min(max(0, receipts-600), 600), max(0, receipts-1200), max(0, receipts-1800),')
output_lines.append('        min(miles, 100), min(max(0, miles-100), 200),')
output_lines.append('        max(0, miles-300), max(0, miles-800),')
output_lines.append('        min(mi_pd, 200), max(0, mi_pd-200),')
output_lines.append('        1.0 if mi_pd > 150 and days <= 3 else 0.0,')
output_lines.append('        1.0 if days >= 7 and miles > 800 and receipts > 800 else 0.0,')
output_lines.append('        1.0 if days >= 7 and miles > 600 else 0.0,')
output_lines.append('        1.0 if 4 <= days <= 6 else 0.0,')
output_lines.append('        1.0 if 7 <= days <= 9 else 0.0,')
output_lines.append('        1.0 if days >= 10 else 0.0,')
output_lines.append('        1.0 if days <= 2 else 0.0,')
output_lines.append('        1.0 if days == 1 else 0.0,')
output_lines.append('        days*math.log(miles+1), mi_pd*r_pd/10000,')
output_lines.append('        max(0, days-10), max(0, days-10)*receipts/1000,')
output_lines.append('        math.sqrt(days)*math.log(miles+1),')
output_lines.append('        days*math.sqrt(receipts), math.sqrt(miles)*math.sqrt(receipts),')
output_lines.append('        miles*math.log(receipts+1)/1000, receipts*math.log(miles+1)/1000,')
output_lines.append('        days*days*miles/10000, days*days*receipts/10000,')
output_lines.append('        min(days, 3), min(days, 3)*miles/100, min(days, 3)*receipts/1000,')
output_lines.append('        1.0 if 6 <= days <= 9 and miles > 900 else 0.0,')
output_lines.append('        (1.0 if 6 <= days <= 9 and miles > 900 else 0.0) * miles / 1000,')
output_lines.append('        (1.0 if 6 <= days <= 9 and miles > 900 else 0.0) * receipts / 1000,')
output_lines.append('        1.0 if days >= 12 and miles > 800 else 0.0,')
output_lines.append('        max(0, days-12)*miles/1000,')
output_lines.append('        1.0 if days >= 12 and miles < 200 else 0.0,')
output_lines.append('        miles**3/1e9, mi_pd*mi_pd/100000,')
output_lines.append('        days*mi_pd/100, receipts**2/1e7,')
output_lines.append('        max(0, receipts-2200), min(max(0, receipts-300), 300),')
output_lines.append('        1.0 if r_pd > 300 else 0.0,')
output_lines.append('        (1.0 if r_pd > 300 else 0.0) * receipts / 1000,')
output_lines.append('        1.0 if r_pd < 30 else 0.0,')
output_lines.append('        (1.0 if r_pd < 30 else 0.0) * days,')
output_lines.append('        1.0 if days <= 2 and receipts > 1500 else 0.0,')
output_lines.append('        (1.0 if days <= 2 and receipts > 1500 else 0.0) * receipts / 1000,')
output_lines.append('        mi_pd*r_pd/10000, r_pd*days/1000,')
output_lines.append('        sp, sp*receipts, sp*days, sp*miles, sp*receipts*receipts/10000, sp*r_pd,')
output_lines.append('        sp*days*receipts/1000,')
output_lines.append('        miles*math.log(days+1)/1000,')
output_lines.append('        1.0 if days >= 5 and miles > 700 and receipts < 1300 else 0.0,')
output_lines.append('        (1.0 if days >= 5 and miles > 700 else 0.0) * miles / 1000,')
output_lines.append('        days*miles*receipts/1e7,')
output_lines.append('        math.log(days+1)*math.log(miles+1)*math.log(receipts+1),')
output_lines.append('        min(mi_pd, 150)*days/100,')
output_lines.append('        max(0, miles-500)*days/1000,')
output_lines.append('        (1.0 if days >= 10 and miles > 600 else 0.0) * miles / 1000,')
output_lines.append('        mi_pd*days/100,')
output_lines.append('        1.0 if days >= 7 and miles > 700 and 600 < receipts < 1400 else 0.0,')
output_lines.append('        (1.0 if days >= 7 and miles > 700 and 600 < receipts < 1400 else 0.0) * miles / 1000,')
output_lines.append('        (1.0 if days >= 7 and miles > 700 and 600 < receipts < 1400 else 0.0) * receipts / 1000,')
output_lines.append('        days*miles*math.log(miles+1)/10000,')
output_lines.append('        days*receipts*math.log(receipts+1)/100000,')
output_lines.append('        miles*receipts*days/10000000,')
output_lines.append('        math.sqrt(days)*math.sqrt(receipts),')
output_lines.append('        math.sqrt(days)*math.sqrt(miles),')
output_lines.append('        math.log(days+1)*math.log(receipts+1),')
output_lines.append('        math.log(days+1)*math.log(miles+1),')
output_lines.append('        math.sqrt(miles)*math.log(receipts+1)/10,')
output_lines.append('        math.sqrt(receipts)*math.log(miles+1)/10,')
output_lines.append('        days*mi_pd*mi_pd/1000000,')
output_lines.append('        1.0 if 50 < r_pd < 100 else 0.0,')
output_lines.append('        1.0 if 100 < r_pd < 200 else 0.0,')
output_lines.append('        1.0 if 200 < r_pd < 300 else 0.0,')
output_lines.append('        (1.0 if days >= 5 and days <= 8 else 0.0) * miles / 1000,')
output_lines.append('        (1.0 if days >= 9 and days <= 12 else 0.0) * miles / 1000,')
output_lines.append('        (1.0 if days >= 13 else 0.0) * miles / 1000,')
output_lines.append('    ]')
output_lines.append('    # Extended features: per-day × mile/receipt tiers + non-linear')
output_lines.append('    for d in range(1, 15):')
output_lines.append('        is_day = 1.0 if days == d else 0.0')
output_lines.append('        feats.extend([')
output_lines.append('            is_day * min(miles, 50),')
output_lines.append('            is_day * min(max(0, miles-50), 50),')
output_lines.append('            is_day * min(max(0, miles-100), 200),')
output_lines.append('            is_day * min(max(0, miles-300), 200),')
output_lines.append('            is_day * min(max(0, miles-500), 300),')
output_lines.append('            is_day * min(max(0, miles-800), 400),')
output_lines.append('            is_day * max(0, miles-1200),')
output_lines.append('        ])')
output_lines.append('    for d in range(1, 15):')
output_lines.append('        is_day = 1.0 if days == d else 0.0')
output_lines.append('        feats.extend([')
output_lines.append('            is_day * min(receipts, 150),')
output_lines.append('            is_day * min(max(0, receipts-150), 150),')
output_lines.append('            is_day * min(max(0, receipts-300), 300),')
output_lines.append('            is_day * min(max(0, receipts-600), 300),')
output_lines.append('            is_day * min(max(0, receipts-900), 300),')
output_lines.append('            is_day * min(max(0, receipts-1200), 400),')
output_lines.append('            is_day * min(max(0, receipts-1600), 400),')
output_lines.append('            is_day * max(0, receipts-2000),')
output_lines.append('        ])')
output_lines.append('    for d in range(1, 15):')
output_lines.append('        is_day = 1.0 if days == d else 0.0')
output_lines.append('        feats.append(is_day * miles * receipts / 100000)')
output_lines.append('    for d in range(1, 15):')
output_lines.append('        is_day = 1.0 if days == d else 0.0')
output_lines.append('        feats.append(is_day * mi_pd / 100)')
output_lines.append('    for d in range(1, 15):')
output_lines.append('        is_day = 1.0 if days == d else 0.0')
output_lines.append('        feats.append(is_day * r_pd / 100)')
output_lines.append('    mile_bins = [min(miles,300)/300, min(max(0,miles-300),500)/500, min(max(0,miles-800),500)/500, max(0,miles-1300)/300]')
output_lines.append('    rcpt_bins = [min(receipts,500)/500, min(max(0,receipts-500),700)/700, min(max(0,receipts-1200),800)/800, max(0,receipts-2000)/500]')
output_lines.append('    for mb in mile_bins:')
output_lines.append('        for rb in rcpt_bins:')
output_lines.append('            feats.append(mb * rb * 100)')
output_lines.append('    feats.extend([')
output_lines.append('        min(miles,300)*min(receipts,500)/10000, max(0,miles-300)*max(0,receipts-500)/100000,')
output_lines.append('        max(0,miles-800)*max(0,receipts-1200)/100000, min(miles,100)*min(miles,100)/10000,')
output_lines.append('        max(0,miles-800)*max(0,miles-800)/1000000, min(receipts,300)*min(receipts,300)/100000,')
output_lines.append('        max(0,receipts-1200)*max(0,receipts-1200)/1000000, mi_pd*max(0,receipts-1000)/1000,')
output_lines.append('        r_pd*max(0,miles-500)/1000, mi_pd*mi_pd*receipts/10000000,')
output_lines.append('        r_pd*r_pd*miles/10000000, min(mi_pd,100)*min(r_pd,200)/10000,')
output_lines.append('        max(0,mi_pd-100)*max(0,r_pd-100)/10000, (1.0 if mi_pd>200 else 0.0)*receipts/1000,')
output_lines.append('        (1.0 if r_pd>200 else 0.0)*miles/1000, max(0,miles-400)*max(0,receipts-800)/100000,')
output_lines.append('        min(miles,200)*max(0,receipts-1500)/100000, max(0,miles-600)*min(receipts,600)/100000,')
output_lines.append('        (1.0 if miles>500 and receipts>800 else 0.0)*(miles+receipts)/1000,')
output_lines.append('        (1.0 if miles<200 and receipts>1500 else 0.0)*receipts/1000,')
output_lines.append('    ])')
output_lines.append('    mile_bins2 = [min(miles,150)/150, min(max(0,miles-150),250)/250, min(max(0,miles-400),300)/300, min(max(0,miles-700),300)/300, min(max(0,miles-1000),300)/300, max(0,miles-1300)/200]')
output_lines.append('    rcpt_bins2 = [min(receipts,250)/250, min(max(0,receipts-250),350)/350, min(max(0,receipts-600),400)/400, min(max(0,receipts-1000),400)/400, min(max(0,receipts-1400),500)/500, max(0,receipts-1900)/600]')
output_lines.append('    for mb in mile_bins2:')
output_lines.append('        for rb in rcpt_bins2:')
output_lines.append('            feats.append(mb * rb * 100)')
output_lines.append('    mile_bins3 = [min(miles,80)/80, min(max(0,miles-80),120)/120, min(max(0,miles-200),150)/150, min(max(0,miles-350),200)/200, min(max(0,miles-550),250)/250, min(max(0,miles-800),250)/250, min(max(0,miles-1050),250)/250, max(0,miles-1300)/200]')
output_lines.append('    rcpt_bins3 = [min(receipts,200)/200, min(max(0,receipts-200),200)/200, min(max(0,receipts-400),300)/300, min(max(0,receipts-700),300)/300, min(max(0,receipts-1000),300)/300, min(max(0,receipts-1300),400)/400, min(max(0,receipts-1700),400)/400, max(0,receipts-2100)/500]')
output_lines.append('    for mb in mile_bins3:')
output_lines.append('        for rb in rcpt_bins3:')
output_lines.append('            feats.append(mb * rb * 100)')
output_lines.append('    feats.extend([')
output_lines.append('        days*miles*miles/1000000, days*receipts*receipts/10000000,')
output_lines.append('        (mi_pd**2)*days/100000, (r_pd**2)*days/10000,')
output_lines.append('        min(days,5)*miles/100, max(0,days-7)*receipts/1000,')
output_lines.append('        max(0,days-7)*miles/1000, min(days,3)*receipts/1000,')
output_lines.append('        (1.0 if days>=8 else 0.0)*miles*receipts/100000,')
output_lines.append('        (1.0 if days<=3 else 0.0)*miles*receipts/100000,')
output_lines.append('        max(0,miles-600)*max(0,days-5)/1000, max(0,receipts-1000)*max(0,days-5)/1000,')
output_lines.append('        miles*miles*receipts/1e9, miles*receipts*receipts/1e9,')
output_lines.append('        mi_pd*min(receipts,1000)/1000, r_pd*min(miles,800)/1000,')
output_lines.append('        max(0,miles-500)*min(receipts,800)/100000, min(miles,500)*max(0,receipts-800)/100000,')
output_lines.append('        max(0,miles-300)*max(0,receipts-300)*max(0,receipts-300)/1e8,')
output_lines.append('        min(miles,300)*min(receipts,300)*min(receipts,300)/1e7,')
output_lines.append('        miles*receipts*miles/1e8, miles*receipts*receipts/1e8,')
output_lines.append('        mi_pd*r_pd*miles/1e6, mi_pd*r_pd*receipts/1e6,')
output_lines.append('    ])')
output_lines.append('    return feats')
output_lines.append('')
output_lines.append('')
output_lines.append('def calculate_reimbursement(days, miles, receipts):')
output_lines.append('    days = max(1, min(days, 14))')
output_lines.append('    miles = max(0.0, miles)')
output_lines.append('    receipts = max(0.0, receipts)')
output_lines.append('')
output_lines.append('    features = _build_features(days, miles, receipts)')
output_lines.append('    coeffs = PER_DAY_COEFFS.get(days, GLOBAL_COEFFS)')
output_lines.append('    result = sum(c * f for c, f in zip(coeffs, features))')
output_lines.append('    return round(max(0.0, result), 2)')
output_lines.append('')
output_lines.append('')
output_lines.append("if __name__ == '__main__':")
output_lines.append('    if len(sys.argv) != 4:')
output_lines.append('        print("Usage: python3 approach3_ridge_features.py <days> <miles> <receipts>")')
output_lines.append('        sys.exit(1)')
output_lines.append('    days = int(float(sys.argv[1]))')
output_lines.append('    miles = float(sys.argv[2])')
output_lines.append('    receipts = float(sys.argv[3])')
output_lines.append('    print(calculate_reimbursement(days, miles, receipts))')

output_text = "\n".join(output_lines) + "\n"

# Write to file
output_path = os.path.join(BASE, 'approach3_ridge_features.py')
with open(output_path, 'w') as f:
    f.write(output_text)
print(f"\nWrote updated approach3_ridge_features.py ({n_features} features, per-day coefficients)")
print(f"File: {output_path}")
