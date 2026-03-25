#!/usr/bin/env python3
"""
generalization_study.py — Test feature counts vs generalization.

Builds progressively larger feature sets, measures training fit vs
5-fold cross-validation MAE, and generates approach3_generalized.py
using the best-generalizing configuration.
"""

import json
import math
import os
import sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Feature builder functions copied from fit_ridge_v3.py to avoid running its top-level code

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
    for d in range(1, 15):
        is_day = 1.0 if days == d else 0.0
        feats.append(is_day * miles * receipts / 100000)
    for d in range(1, 15):
        is_day = 1.0 if days == d else 0.0
        feats.append(is_day * mi_pd / 100)
    for d in range(1, 15):
        is_day = 1.0 if days == d else 0.0
        feats.append(is_day * r_pd / 100)
    mile_bins = [
        min(miles, 300) / 300, min(max(0, miles-300), 500) / 500,
        min(max(0, miles-800), 500) / 500, max(0, miles-1300) / 300,
    ]
    rcpt_bins = [
        min(receipts, 500) / 500, min(max(0, receipts-500), 700) / 700,
        min(max(0, receipts-1200), 800) / 800, max(0, receipts-2000) / 500,
    ]
    for mb in mile_bins:
        for rb in rcpt_bins:
            feats.append(mb * rb * 100)
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
    mile_bins2 = [
        min(miles, 150) / 150, min(max(0, miles-150), 250) / 250,
        min(max(0, miles-400), 300) / 300, min(max(0, miles-700), 300) / 300,
        min(max(0, miles-1000), 300) / 300, max(0, miles-1300) / 200,
    ]
    rcpt_bins2 = [
        min(receipts, 250) / 250, min(max(0, receipts-250), 350) / 350,
        min(max(0, receipts-600), 400) / 400, min(max(0, receipts-1000), 400) / 400,
        min(max(0, receipts-1400), 500) / 500, max(0, receipts-1900) / 600,
    ]
    for mb in mile_bins2:
        for rb in rcpt_bins2:
            feats.append(mb * rb * 100)
    mile_bins3 = [
        min(miles, 80) / 80, min(max(0, miles-80), 120) / 120,
        min(max(0, miles-200), 150) / 150, min(max(0, miles-350), 200) / 200,
        min(max(0, miles-550), 250) / 250, min(max(0, miles-800), 250) / 250,
        min(max(0, miles-1050), 250) / 250, max(0, miles-1300) / 200,
    ]
    rcpt_bins3 = [
        min(receipts, 200) / 200, min(max(0, receipts-200), 200) / 200,
        min(max(0, receipts-400), 300) / 300, min(max(0, receipts-700), 300) / 300,
        min(max(0, receipts-1000), 300) / 300, min(max(0, receipts-1300), 400) / 400,
        min(max(0, receipts-1700), 400) / 400, max(0, receipts-2100) / 500,
    ]
    for mb in mile_bins3:
        for rb in rcpt_bins3:
            feats.append(mb * rb * 100)
    feats.extend([
        days*miles*miles/1000000, days*receipts*receipts/10000000,
        (mi_pd**2)*days/100000, (r_pd**2)*days/10000,
        min(days, 5)*miles/100, max(0, days-7)*receipts/1000,
        max(0, days-7)*miles/1000, min(days, 3)*receipts/1000,
        (1.0 if days >= 8 else 0.0) * miles * receipts / 100000,
        (1.0 if days <= 3 else 0.0) * miles * receipts / 100000,
        max(0, miles - 600) * max(0, days - 5) / 1000,
        max(0, receipts - 1000) * max(0, days - 5) / 1000,
        miles * miles * receipts / 1e9, miles * receipts * receipts / 1e9,
        mi_pd * min(receipts, 1000) / 1000, r_pd * min(miles, 800) / 1000,
        max(0, miles-500) * min(receipts, 800) / 100000,
        min(miles, 500) * max(0, receipts-800) / 100000,
        max(0, miles-300) * max(0, receipts-300) * max(0, receipts-300) / 1e8,
        min(miles, 300) * min(receipts, 300) * min(receipts, 300) / 1e7,
        miles * receipts * miles / 1e8, miles * receipts * receipts / 1e8,
        mi_pd * r_pd * miles / 1e6, mi_pd * r_pd * receipts / 1e6,
    ])
    return feats

def build_all_features(days, miles, receipts):
    return _build_features_base(days, miles, receipts) + _build_extended_features(days, miles, receipts)

# Load data
with open(os.path.join(BASE, 'public_cases.json')) as f:
    cases = json.load(f)

# Build full feature matrix
X_list, y_list, day_list = [], [], []
for c in cases:
    d = c['input']['trip_duration_days']
    m = c['input']['miles_traveled']
    r = c['input']['total_receipts_amount']
    feats = build_all_features(d, m, r)
    X_list.append(feats)
    y_list.append(c['expected_output'])
    day_list.append(d)

X_full = np.array(X_list)
y = np.array(y_list)
days_arr = np.array(day_list)

N_BASE = 108

# Feature level definitions: (name, total_feature_count)
# Extended features are ordered: mile_tiers(98), receipt_tiers(112),
# per-day interactions(42), 4x4_grid(16), piecewise(20), 6x6_grid(36),
# 8x8_grid(64), non-linear(24)
LEVELS = [
    ("Base only",                   108),   # 108
    ("+ per-day mile tiers",        206),   # +98
    ("+ per-day receipt tiers",     318),   # +112
    ("+ per-day interactions",      360),   # +42
    ("+ 4x4 grid + piecewise",     396),   # +16+20
    ("+ 6x6 grid",                  432),   # +36
    ("+ 8x8 grid + non-linear",    520),   # +64+24
]

ALPHAS = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0, 10.0]


def fit_perday_ridge(X_train, y_train, days_train, alpha):
    """Fit per-day Ridge models with given alpha. Returns dict of raw coefficients."""
    n_feat = X_train.shape[1]
    coeffs = {}
    for d in range(1, 15):
        mask = days_train == d
        if mask.sum() < 2:
            continue
        Xd = X_train[mask]
        yd = y_train[mask]
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xd)
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(Xs, yd)
        # Extract raw-space coefficients
        raw_coef = ridge.coef_ / scaler.scale_
        raw_intercept = ridge.intercept_ - np.sum(ridge.coef_ * scaler.mean_ / scaler.scale_)
        full_c = np.zeros(n_feat)
        full_c[0] = raw_intercept + raw_coef[0]
        full_c[1:] = raw_coef[1:]
        coeffs[d] = full_c
    return coeffs


def predict_perday(X, days, coeffs_dict):
    """Predict using per-day coefficients. Falls back to global if day missing."""
    preds = np.zeros(len(days))
    for i in range(len(days)):
        d = days[i]
        if d in coeffs_dict:
            preds[i] = max(0.0, round(float(X[i] @ coeffs_dict[d]), 2))
        else:
            preds[i] = 0.0  # should not happen with proper training
    return preds


def compute_mae(preds, targets):
    return np.mean(np.abs(preds - targets))


def compute_exact(preds, targets):
    return np.sum(np.abs(preds - targets) < 0.01)


def compute_score(preds, targets):
    mae = compute_mae(preds, targets)
    exact = compute_exact(preds, targets)
    return mae * 100 + (len(targets) - exact) * 0.1


print("=" * 90)
print(f"{'Level':<6} {'Features':<9} {'Train MAE':>10} {'Train Score':>12} {'Train Alpha':>12} "
      f"{'CV MAE':>10} {'CV Alpha':>10} {'Exact':>6}")
print("-" * 90)

results = []

for level_idx, (level_name, n_feat) in enumerate(LEVELS):
    X = X_full[:, :n_feat]

    # --- Training: find best alpha ---
    best_train_mae = float('inf')
    best_train_alpha = None
    best_train_exact = 0
    best_train_score = float('inf')

    for alpha in ALPHAS:
        coeffs = fit_perday_ridge(X, y, days_arr, alpha)
        preds = predict_perday(X, days_arr, coeffs)
        mae = compute_mae(preds, y)
        if mae < best_train_mae:
            best_train_mae = mae
            best_train_alpha = alpha
            best_train_exact = int(compute_exact(preds, y))
            best_train_score = compute_score(preds, y)

    # --- 5-fold CV: find best alpha ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_cv_mae = float('inf')
    best_cv_alpha = None

    for alpha in ALPHAS:
        fold_preds = np.zeros(len(y))
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            days_train, days_test = days_arr[train_idx], days_arr[test_idx]

            coeffs = fit_perday_ridge(X_train, y_train, days_train, alpha)
            fold_preds[test_idx] = predict_perday(X_test, days_test, coeffs)

        cv_mae = compute_mae(fold_preds, y)
        if cv_mae < best_cv_mae:
            best_cv_mae = cv_mae
            best_cv_alpha = alpha

    print(f"{level_idx:<6} {n_feat:<9} ${best_train_mae:>8.2f} {best_train_score:>12.2f} "
          f"{best_train_alpha:>12.0e} ${best_cv_mae:>8.2f} {best_cv_alpha:>10.0e} "
          f"{best_train_exact:>5}")

    results.append({
        'level': level_idx,
        'name': level_name,
        'n_feat': n_feat,
        'train_mae': best_train_mae,
        'train_score': best_train_score,
        'train_alpha': best_train_alpha,
        'train_exact': best_train_exact,
        'cv_mae': best_cv_mae,
        'cv_alpha': best_cv_alpha,
    })

print("=" * 90)

# Find sweet spot
best_cv = min(results, key=lambda r: r['cv_mae'])
print(f"\nBest generalizing level: {best_cv['level']} ({best_cv['name']})")
print(f"  Features: {best_cv['n_feat']}")
print(f"  CV MAE: ${best_cv['cv_mae']:.2f} (alpha={best_cv['cv_alpha']:.0e})")
print(f"  Train MAE: ${best_cv['train_mae']:.2f} (alpha={best_cv['train_alpha']:.0e})")

# --- Generate approach3_generalized.py ---
print("\n\nGenerating approach3_generalized.py...")

sweet_n_feat = best_cv['n_feat']
sweet_alpha = best_cv['cv_alpha']

# Refit with CV-optimal alpha on ALL data
X_sweet = X_full[:, :sweet_n_feat]
final_coeffs = fit_perday_ridge(X_sweet, y, days_arr, sweet_alpha)

# Also fit global fallback
scaler = StandardScaler()
Xs = scaler.fit_transform(X_sweet)
ridge = Ridge(alpha=sweet_alpha, fit_intercept=True)
ridge.fit(Xs, y)
raw_coef = ridge.coef_ / scaler.scale_
raw_intercept = ridge.intercept_ - np.sum(ridge.coef_ * scaler.mean_ / scaler.scale_)
global_coeffs = np.zeros(sweet_n_feat)
global_coeffs[0] = raw_intercept + raw_coef[0]
global_coeffs[1:] = raw_coef[1:]

# Verify final model
final_preds = predict_perday(X_sweet, days_arr, final_coeffs)
final_mae = compute_mae(final_preds, y)
final_exact = int(compute_exact(final_preds, y))
final_score = compute_score(final_preds, y)
print(f"Final model (all data, alpha={sweet_alpha:.0e}): MAE=${final_mae:.2f}, exact={final_exact}, score={final_score:.2f}")


# --- Build the output file ---
# We need to determine which feature groups to include based on sweet_n_feat
# Extended feature groups and their sizes:
EXT_GROUPS = [
    ('mile_tiers', 98),
    ('receipt_tiers', 112),
    ('perday_interactions', 42),
    ('grid_4x4_piecewise', 36),  # 16+20
    ('grid_6x6', 36),
    ('grid_8x8_nonlinear', 88),  # 64+24
]

# Figure out which extended groups are fully included
ext_needed = sweet_n_feat - N_BASE
cumsum = 0
included_groups = []
for name, size in EXT_GROUPS:
    if cumsum >= ext_needed:
        break
    included_groups.append(name)
    cumsum += size

print(f"Included extended feature groups: {included_groups}")
print(f"Extended features needed: {ext_needed}, cumulative from groups: {cumsum}")

# Now generate the Python file
lines = []
lines.append('#!/usr/bin/env python3')
lines.append('"""')
lines.append(f'Generalized Ridge model with {sweet_n_feat} features.')
lines.append(f'CV-optimal alpha={sweet_alpha:.0e}, per-day coefficients.')
lines.append('"""')
lines.append('')
lines.append('import sys')
lines.append('import math')
lines.append('')
lines.append('')
lines.append('def is_special_cents(receipts):')
lines.append('    cents = round(receipts * 100) % 100')
lines.append('    return cents == 49 or cents == 99')
lines.append('')
lines.append('')

# PER_DAY_COEFFS
lines.append('PER_DAY_COEFFS = {')
for d in range(1, 15):
    if d in final_coeffs:
        c = final_coeffs[d]
        vals = ", ".join(f"{v:.10f}" for v in c)
        lines.append(f'    {d}: [{vals}],')
lines.append('}')
lines.append('')

# GLOBAL_COEFFS
vals = ", ".join(f"{v:.10f}" for v in global_coeffs)
lines.append(f'GLOBAL_COEFFS = [{vals}]')
lines.append('')
lines.append('')

# _build_features function
# Always include base 108 features
lines.append('def _build_features(days, miles, receipts):')
lines.append(f'    """Build the {sweet_n_feat}-feature vector."""')
lines.append('    mi_pd = miles / days')
lines.append('    r_pd = receipts / days')
lines.append('    sp = 1.0 if is_special_cents(receipts) else 0.0')
lines.append('    feats = [')
lines.append('        1, days, miles, receipts,')
lines.append('        days*days, days*days*days, miles*miles/10000,')
lines.append('        math.log(days+1), math.log(miles+1), math.log(receipts+1),')
lines.append('        math.sqrt(miles), math.sqrt(receipts), math.sqrt(days),')
lines.append('        mi_pd, r_pd, r_pd*r_pd/10000,')
lines.append('        days*receipts/1000, miles*receipts/100000, days*miles/100,')
lines.append('        math.log(miles+1)*math.log(receipts+1),')
lines.append('        min(receipts, 300), min(max(0, receipts-300), 300),')
lines.append('        min(max(0, receipts-600), 600), max(0, receipts-1200), max(0, receipts-1800),')
lines.append('        min(miles, 100), min(max(0, miles-100), 200),')
lines.append('        max(0, miles-300), max(0, miles-800),')
lines.append('        min(mi_pd, 200), max(0, mi_pd-200),')
lines.append('        1.0 if mi_pd > 150 and days <= 3 else 0.0,')
lines.append('        1.0 if days >= 7 and miles > 800 and receipts > 800 else 0.0,')
lines.append('        1.0 if days >= 7 and miles > 600 else 0.0,')
lines.append('        1.0 if 4 <= days <= 6 else 0.0,')
lines.append('        1.0 if 7 <= days <= 9 else 0.0,')
lines.append('        1.0 if days >= 10 else 0.0,')
lines.append('        1.0 if days <= 2 else 0.0,')
lines.append('        1.0 if days == 1 else 0.0,')
lines.append('        days*math.log(miles+1), mi_pd*r_pd/10000,')
lines.append('        max(0, days-10), max(0, days-10)*receipts/1000,')
lines.append('        math.sqrt(days)*math.log(miles+1),')
lines.append('        days*math.sqrt(receipts), math.sqrt(miles)*math.sqrt(receipts),')
lines.append('        miles*math.log(receipts+1)/1000, receipts*math.log(miles+1)/1000,')
lines.append('        days*days*miles/10000, days*days*receipts/10000,')
lines.append('        min(days, 3), min(days, 3)*miles/100, min(days, 3)*receipts/1000,')
lines.append('        1.0 if 6 <= days <= 9 and miles > 900 else 0.0,')
lines.append('        (1.0 if 6 <= days <= 9 and miles > 900 else 0.0) * miles / 1000,')
lines.append('        (1.0 if 6 <= days <= 9 and miles > 900 else 0.0) * receipts / 1000,')
lines.append('        1.0 if days >= 12 and miles > 800 else 0.0,')
lines.append('        max(0, days-12)*miles/1000,')
lines.append('        1.0 if days >= 12 and miles < 200 else 0.0,')
lines.append('        miles**3/1e9, mi_pd*mi_pd/100000,')
lines.append('        days*mi_pd/100, receipts**2/1e7,')
lines.append('        max(0, receipts-2200), min(max(0, receipts-300), 300),')
lines.append('        1.0 if r_pd > 300 else 0.0,')
lines.append('        (1.0 if r_pd > 300 else 0.0) * receipts / 1000,')
lines.append('        1.0 if r_pd < 30 else 0.0,')
lines.append('        (1.0 if r_pd < 30 else 0.0) * days,')
lines.append('        1.0 if days <= 2 and receipts > 1500 else 0.0,')
lines.append('        (1.0 if days <= 2 and receipts > 1500 else 0.0) * receipts / 1000,')
lines.append('        mi_pd*r_pd/10000, r_pd*days/1000,')
lines.append('        sp, sp*receipts, sp*days, sp*miles, sp*receipts*receipts/10000, sp*r_pd,')
lines.append('        sp*days*receipts/1000,')
lines.append('        miles*math.log(days+1)/1000,')
lines.append('        1.0 if days >= 5 and miles > 700 and receipts < 1300 else 0.0,')
lines.append('        (1.0 if days >= 5 and miles > 700 else 0.0) * miles / 1000,')
lines.append('        days*miles*receipts/1e7,')
lines.append('        math.log(days+1)*math.log(miles+1)*math.log(receipts+1),')
lines.append('        min(mi_pd, 150)*days/100,')
lines.append('        max(0, miles-500)*days/1000,')
lines.append('        (1.0 if days >= 10 and miles > 600 else 0.0) * miles / 1000,')
lines.append('        mi_pd*days/100,')
lines.append('        1.0 if days >= 7 and miles > 700 and 600 < receipts < 1400 else 0.0,')
lines.append('        (1.0 if days >= 7 and miles > 700 and 600 < receipts < 1400 else 0.0) * miles / 1000,')
lines.append('        (1.0 if days >= 7 and miles > 700 and 600 < receipts < 1400 else 0.0) * receipts / 1000,')
lines.append('        days*miles*math.log(miles+1)/10000,')
lines.append('        days*receipts*math.log(receipts+1)/100000,')
lines.append('        miles*receipts*days/10000000,')
lines.append('        math.sqrt(days)*math.sqrt(receipts),')
lines.append('        math.sqrt(days)*math.sqrt(miles),')
lines.append('        math.log(days+1)*math.log(receipts+1),')
lines.append('        math.log(days+1)*math.log(miles+1),')
lines.append('        math.sqrt(miles)*math.log(receipts+1)/10,')
lines.append('        math.sqrt(receipts)*math.log(miles+1)/10,')
lines.append('        days*mi_pd*mi_pd/1000000,')
lines.append('        1.0 if 50 < r_pd < 100 else 0.0,')
lines.append('        1.0 if 100 < r_pd < 200 else 0.0,')
lines.append('        1.0 if 200 < r_pd < 300 else 0.0,')
lines.append('        (1.0 if days >= 5 and days <= 8 else 0.0) * miles / 1000,')
lines.append('        (1.0 if days >= 9 and days <= 12 else 0.0) * miles / 1000,')
lines.append('        (1.0 if days >= 13 else 0.0) * miles / 1000,')
lines.append('    ]')

# Conditionally include extended feature groups
if ext_needed > 0:
    lines.append('    # Extended features')

if ext_needed >= 98:  # mile tiers
    lines.append('    for d in range(1, 15):')
    lines.append('        is_day = 1.0 if days == d else 0.0')
    lines.append('        feats.extend([')
    lines.append('            is_day * min(miles, 50),')
    lines.append('            is_day * min(max(0, miles-50), 50),')
    lines.append('            is_day * min(max(0, miles-100), 200),')
    lines.append('            is_day * min(max(0, miles-300), 200),')
    lines.append('            is_day * min(max(0, miles-500), 300),')
    lines.append('            is_day * min(max(0, miles-800), 400),')
    lines.append('            is_day * max(0, miles-1200),')
    lines.append('        ])')

if ext_needed >= 210:  # + receipt tiers
    lines.append('    for d in range(1, 15):')
    lines.append('        is_day = 1.0 if days == d else 0.0')
    lines.append('        feats.extend([')
    lines.append('            is_day * min(receipts, 150),')
    lines.append('            is_day * min(max(0, receipts-150), 150),')
    lines.append('            is_day * min(max(0, receipts-300), 300),')
    lines.append('            is_day * min(max(0, receipts-600), 300),')
    lines.append('            is_day * min(max(0, receipts-900), 300),')
    lines.append('            is_day * min(max(0, receipts-1200), 400),')
    lines.append('            is_day * min(max(0, receipts-1600), 400),')
    lines.append('            is_day * max(0, receipts-2000),')
    lines.append('        ])')

if ext_needed >= 252:  # + per-day interactions
    lines.append('    for d in range(1, 15):')
    lines.append('        is_day = 1.0 if days == d else 0.0')
    lines.append('        feats.append(is_day * miles * receipts / 100000)')
    lines.append('    for d in range(1, 15):')
    lines.append('        is_day = 1.0 if days == d else 0.0')
    lines.append('        feats.append(is_day * mi_pd / 100)')
    lines.append('    for d in range(1, 15):')
    lines.append('        is_day = 1.0 if days == d else 0.0')
    lines.append('        feats.append(is_day * r_pd / 100)')

if ext_needed >= 288:  # + 4x4 grid + piecewise products
    lines.append('    mile_bins = [min(miles,300)/300, min(max(0,miles-300),500)/500, min(max(0,miles-800),500)/500, max(0,miles-1300)/300]')
    lines.append('    rcpt_bins = [min(receipts,500)/500, min(max(0,receipts-500),700)/700, min(max(0,receipts-1200),800)/800, max(0,receipts-2000)/500]')
    lines.append('    for mb in mile_bins:')
    lines.append('        for rb in rcpt_bins:')
    lines.append('            feats.append(mb * rb * 100)')
    lines.append('    feats.extend([')
    lines.append('        min(miles,300)*min(receipts,500)/10000, max(0,miles-300)*max(0,receipts-500)/100000,')
    lines.append('        max(0,miles-800)*max(0,receipts-1200)/100000, min(miles,100)*min(miles,100)/10000,')
    lines.append('        max(0,miles-800)*max(0,miles-800)/1000000, min(receipts,300)*min(receipts,300)/100000,')
    lines.append('        max(0,receipts-1200)*max(0,receipts-1200)/1000000, mi_pd*max(0,receipts-1000)/1000,')
    lines.append('        r_pd*max(0,miles-500)/1000, mi_pd*mi_pd*receipts/10000000,')
    lines.append('        r_pd*r_pd*miles/10000000, min(mi_pd,100)*min(r_pd,200)/10000,')
    lines.append('        max(0,mi_pd-100)*max(0,r_pd-100)/10000, (1.0 if mi_pd>200 else 0.0)*receipts/1000,')
    lines.append('        (1.0 if r_pd>200 else 0.0)*miles/1000, max(0,miles-400)*max(0,receipts-800)/100000,')
    lines.append('        min(miles,200)*max(0,receipts-1500)/100000, max(0,miles-600)*min(receipts,600)/100000,')
    lines.append('        (1.0 if miles>500 and receipts>800 else 0.0)*(miles+receipts)/1000,')
    lines.append('        (1.0 if miles<200 and receipts>1500 else 0.0)*receipts/1000,')
    lines.append('    ])')

if ext_needed >= 324:  # + 6x6 grid
    lines.append('    mile_bins2 = [min(miles,150)/150, min(max(0,miles-150),250)/250, min(max(0,miles-400),300)/300, min(max(0,miles-700),300)/300, min(max(0,miles-1000),300)/300, max(0,miles-1300)/200]')
    lines.append('    rcpt_bins2 = [min(receipts,250)/250, min(max(0,receipts-250),350)/350, min(max(0,receipts-600),400)/400, min(max(0,receipts-1000),400)/400, min(max(0,receipts-1400),500)/500, max(0,receipts-1900)/600]')
    lines.append('    for mb in mile_bins2:')
    lines.append('        for rb in rcpt_bins2:')
    lines.append('            feats.append(mb * rb * 100)')

if ext_needed >= 412:  # + 8x8 grid + non-linear
    lines.append('    mile_bins3 = [min(miles,80)/80, min(max(0,miles-80),120)/120, min(max(0,miles-200),150)/150, min(max(0,miles-350),200)/200, min(max(0,miles-550),250)/250, min(max(0,miles-800),250)/250, min(max(0,miles-1050),250)/250, max(0,miles-1300)/200]')
    lines.append('    rcpt_bins3 = [min(receipts,200)/200, min(max(0,receipts-200),200)/200, min(max(0,receipts-400),300)/300, min(max(0,receipts-700),300)/300, min(max(0,receipts-1000),300)/300, min(max(0,receipts-1300),400)/400, min(max(0,receipts-1700),400)/400, max(0,receipts-2100)/500]')
    lines.append('    for mb in mile_bins3:')
    lines.append('        for rb in rcpt_bins3:')
    lines.append('            feats.append(mb * rb * 100)')
    lines.append('    feats.extend([')
    lines.append('        days*miles*miles/1000000, days*receipts*receipts/10000000,')
    lines.append('        (mi_pd**2)*days/100000, (r_pd**2)*days/10000,')
    lines.append('        min(days,5)*miles/100, max(0,days-7)*receipts/1000,')
    lines.append('        max(0,days-7)*miles/1000, min(days,3)*receipts/1000,')
    lines.append('        (1.0 if days>=8 else 0.0)*miles*receipts/100000,')
    lines.append('        (1.0 if days<=3 else 0.0)*miles*receipts/100000,')
    lines.append('        max(0,miles-600)*max(0,days-5)/1000, max(0,receipts-1000)*max(0,days-5)/1000,')
    lines.append('        miles*miles*receipts/1e9, miles*receipts*receipts/1e9,')
    lines.append('        mi_pd*min(receipts,1000)/1000, r_pd*min(miles,800)/1000,')
    lines.append('        max(0,miles-500)*min(receipts,800)/100000, min(miles,500)*max(0,receipts-800)/100000,')
    lines.append('        max(0,miles-300)*max(0,receipts-300)*max(0,receipts-300)/1e8,')
    lines.append('        min(miles,300)*min(receipts,300)*min(receipts,300)/1e7,')
    lines.append('        miles*receipts*miles/1e8, miles*receipts*receipts/1e8,')
    lines.append('        mi_pd*r_pd*miles/1e6, mi_pd*r_pd*receipts/1e6,')
    lines.append('    ])')

lines.append('    return feats')
lines.append('')
lines.append('')
lines.append('def calculate_reimbursement(days, miles, receipts):')
lines.append('    days = max(1, min(days, 14))')
lines.append('    miles = max(0.0, miles)')
lines.append('    receipts = max(0.0, receipts)')
lines.append('')
lines.append('    features = _build_features(days, miles, receipts)')
lines.append('    coeffs = PER_DAY_COEFFS.get(days, GLOBAL_COEFFS)')
lines.append('    result = sum(c * f for c, f in zip(coeffs, features))')
lines.append('    return round(max(0.0, result), 2)')
lines.append('')
lines.append('')
lines.append("if __name__ == '__main__':")
lines.append('    if len(sys.argv) != 4:')
lines.append('        print("Usage: python3 approach3_generalized.py <days> <miles> <receipts>")')
lines.append('        sys.exit(1)')
lines.append('    days = int(float(sys.argv[1]))')
lines.append('    miles = float(sys.argv[2])')
lines.append('    receipts = float(sys.argv[3])')
lines.append('    print(calculate_reimbursement(days, miles, receipts))')

output_text = "\n".join(lines) + "\n"
output_path = os.path.join(BASE, 'approach3_generalized.py')
with open(output_path, 'w') as f:
    f.write(output_text)
print(f"\nWrote {output_path}")
print(f"Features: {sweet_n_feat}, Alpha: {sweet_alpha:.0e}")

# Verify the generated file works
print("\nVerifying generated model...")
import subprocess
test_cases = [(cases[0], 0), (cases[100], 100), (cases[500], 500)]
for c, idx in test_cases:
    d = c['input']['trip_duration_days']
    m = c['input']['miles_traveled']
    r = c['input']['total_receipts_amount']
    expected = c['expected_output']
    result = subprocess.run(
        ['python3', output_path, str(d), str(m), str(r)],
        capture_output=True, text=True
    )
    pred = float(result.stdout.strip())
    print(f"  Case {idx}: pred={pred}, expected={expected}, diff={abs(pred-expected):.2f}")
