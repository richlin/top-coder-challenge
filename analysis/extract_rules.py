#!/usr/bin/env python3
"""
Extract interpretable business rules from the reimbursement data.

Strategy: iterative decomposition.
1. Fit f(days) on median reimbursement per day count
2. Subtract f(days), fit g(miles) on residuals
3. Subtract g(miles), fit h(receipts) on residuals
4. Analyze remaining residuals for interaction rules
5. Analyze .49/.99 bug separately

Each step uses controlled subsets to isolate the component.
"""

import json
import os
import math
import numpy as np
from scipy.optimize import curve_fit, minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'public_cases.json')

def load_data():
    with open(DATA_PATH) as f:
        cases = json.load(f)
    days = np.array([c['input']['trip_duration_days'] for c in cases])
    miles = np.array([c['input']['miles_traveled'] for c in cases])
    receipts = np.array([c['input']['total_receipts_amount'] for c in cases])
    expected = np.array([c['expected_output'] for c in cases])
    return days, miles, receipts, expected

def is_special_cents(r):
    cents = round(r * 100) % 100
    return cents == 49 or cents == 99

# ─── Step 1: Per-diem f(days) ──────────────────────────────────
def extract_per_diem(days, miles, receipts, expected):
    """
    Extract f(days) by fitting against median reimbursement per day count,
    normalized by controlling for miles/receipts.
    """
    print("=" * 70)
    print("RULE 1: Per-Diem Component f(days)")
    print("=" * 70)

    non_bug = np.array([not is_special_cents(r) for r in receipts])

    # For each day count, compute median reimbursement
    # Use cases with "typical" miles (200-800) and receipts (400-1200) to control
    typical = non_bug & (miles >= 200) & (miles <= 800) & (receipts >= 400) & (receipts <= 1200)

    day_medians = {}
    for d in range(1, 15):
        mask = typical & (days == d)
        if mask.sum() >= 3:
            day_medians[d] = np.median(expected[mask])

    print("\n  Median reimbursement (typical trips: 200-800mi, $400-$1200):")
    for d, med in sorted(day_medians.items()):
        print(f"    {d:2d} days: ${med:7.2f} (per day: ${med/d:6.2f})")

    # Fit: f(days) = a * log(days + 1) + b * days + c
    ds = np.array(sorted(day_medians.keys()))
    ms = np.array([day_medians[d] for d in ds])

    def per_diem_func(d, a, b, c):
        return a * np.log(d + 1) + b * d + c

    popt, _ = curve_fit(per_diem_func, ds, ms, p0=[300, 20, -100])
    a, b, c = popt

    print(f"\n  Fitted formula: f(days) = {a:.2f} * log(days+1) + {b:.2f} * days + {c:.2f}")
    print(f"\n  Predictions vs actual:")
    for d in range(1, 15):
        pred = per_diem_func(d, a, b, c)
        actual = day_medians.get(d, float('nan'))
        print(f"    {d:2d} days: predicted=${pred:7.2f}, actual=${actual:7.2f}")

    return lambda d: per_diem_func(d, a, b, c), (a, b, c)


# ─── Step 2: Mileage g(miles) ──────────────────────────────────
def extract_mileage(days, miles, receipts, expected, f_days):
    """
    Extract g(miles) from residuals after removing f(days).
    """
    print("\n" + "=" * 70)
    print("RULE 2: Mileage Component g(miles)")
    print("=" * 70)

    non_bug = np.array([not is_special_cents(r) for r in receipts])

    # Remove per-diem component
    residuals = expected - np.array([f_days(d) for d in days])

    # Use cases with typical receipts (400-1200) to isolate mileage effect
    typical = non_bug & (receipts >= 400) & (receipts <= 1200)

    # Bin by miles and compute median residual
    bins = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400),
            (400, 500), (500, 600), (600, 700), (700, 800),
            (800, 900), (900, 1000), (1000, 1100), (1100, 1400)]

    print("\n  Median residual after removing f(days) [typical receipts $400-$1200]:")
    mile_points = []
    for lo, hi in bins:
        mask = typical & (miles >= lo) & (miles < hi)
        if mask.sum() >= 3:
            med_res = np.median(residuals[mask])
            avg_mi = np.mean(miles[mask])
            mile_points.append((avg_mi, med_res))
            print(f"    {lo:4d}-{hi:4d} mi (n={mask.sum():3d}): avg_mi={avg_mi:6.0f}, residual=${med_res:7.2f}")

    # Fit piecewise linear: different rate per tier
    mi_arr = np.array([p[0] for p in mile_points])
    res_arr = np.array([p[1] for p in mile_points])

    def mileage_piecewise(m, r1, r2, r3, r4, offset):
        return (r1 * np.minimum(m, 100)
                + r2 * np.minimum(np.maximum(0, m - 100), 200)
                + r3 * np.minimum(np.maximum(0, m - 300), 500)
                + r4 * np.maximum(0, m - 800)
                + offset)

    try:
        popt, _ = curve_fit(mileage_piecewise, mi_arr, res_arr, p0=[0.5, 0.4, 0.6, 0.3, 0])
        r1, r2, r3, r4, offset = popt
        print(f"\n  Fitted piecewise mileage rates:")
        print(f"    0-100 miles:   ${r1:.4f}/mile")
        print(f"    100-300 miles: ${r2:.4f}/mile")
        print(f"    300-800 miles: ${r3:.4f}/mile")
        print(f"    800+ miles:    ${r4:.4f}/mile")
        print(f"    Base offset:   ${offset:.2f}")

        mileage_func = lambda m: mileage_piecewise(m, r1, r2, r3, r4, offset)
        return mileage_func, (r1, r2, r3, r4, offset)
    except Exception as e:
        print(f"  Curve fit failed: {e}")
        # Fallback: simple linear
        from numpy.polynomial import polynomial as P
        c = P.polyfit(mi_arr, res_arr, 1)
        mileage_func = lambda m: c[0] + c[1] * m
        return mileage_func, c


# ─── Step 3: Receipt h(receipts) ──────────────────────────────
def extract_receipts(days, miles, receipts, expected, f_days, g_miles):
    """
    Extract h(receipts) from residuals after removing f(days) + g(miles).
    """
    print("\n" + "=" * 70)
    print("RULE 3: Receipt Component h(receipts)")
    print("=" * 70)

    non_bug = np.array([not is_special_cents(r) for r in receipts])

    # Remove per-diem and mileage
    residuals = expected - np.array([f_days(d) for d in days]) - np.array([g_miles(m) for m in miles])

    # Use cases with typical days (4-8) and miles (200-800)
    typical = non_bug & (days >= 4) & (days <= 8) & (miles >= 200) & (miles <= 800)

    bins = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 600),
            (600, 800), (800, 1000), (1000, 1200), (1200, 1400),
            (1400, 1600), (1600, 1800), (1800, 2000), (2000, 2200), (2200, 2600)]

    print("\n  Median residual after removing f(days)+g(miles) [typical: 4-8d, 200-800mi]:")
    rcpt_points = []
    for lo, hi in bins:
        mask = typical & (receipts >= lo) & (receipts < hi)
        if mask.sum() >= 2:
            med_res = np.median(residuals[mask])
            avg_r = np.mean(receipts[mask])
            rcpt_points.append((avg_r, med_res))
            print(f"    ${lo:5d}-${hi:5d} (n={mask.sum():3d}): avg_rcpt=${avg_r:7.0f}, residual=${med_res:7.2f}")

    r_arr = np.array([p[0] for p in rcpt_points])
    res_arr = np.array([p[1] for p in rcpt_points])

    def receipt_piecewise(r, r1, r2, r3, r4, r5, offset):
        return (r1 * np.minimum(r, 300)
                + r2 * np.minimum(np.maximum(0, r - 300), 300)
                + r3 * np.minimum(np.maximum(0, r - 600), 600)
                + r4 * np.minimum(np.maximum(0, r - 1200), 600)
                + r5 * np.maximum(0, r - 1800)
                + offset)

    try:
        popt, _ = curve_fit(receipt_piecewise, r_arr, res_arr, p0=[0.1, 0.5, 0.8, 0.3, -0.1, 0])
        r1, r2, r3, r4, r5, offset = popt
        print(f"\n  Fitted piecewise receipt rates:")
        print(f"    $0-$300:       ${r1:.4f} per dollar ({r1*100:.1f}% reimbursement)")
        print(f"    $300-$600:     ${r2:.4f} per dollar ({r2*100:.1f}%)")
        print(f"    $600-$1200:    ${r3:.4f} per dollar ({r3*100:.1f}%)")
        print(f"    $1200-$1800:   ${r4:.4f} per dollar ({r4*100:.1f}%)")
        print(f"    $1800+:        ${r5:.4f} per dollar ({r5*100:.1f}%)")
        print(f"    Base offset:   ${offset:.2f}")

        receipt_func = lambda r: receipt_piecewise(r, r1, r2, r3, r4, r5, offset)
        return receipt_func, (r1, r2, r3, r4, r5, offset)
    except Exception as e:
        print(f"  Curve fit failed: {e}")
        from numpy.polynomial import polynomial as P
        c = P.polyfit(r_arr, res_arr, 2)
        receipt_func = lambda r: c[0] + c[1] * r + c[2] * r**2
        return receipt_func, c


# ─── Step 4: Interaction rules ──────────────────────────────────
def extract_interactions(days, miles, receipts, expected, f_days, g_miles, h_receipts):
    """
    Analyze residuals after removing all three base components.
    These residuals reveal the interaction rules.
    """
    print("\n" + "=" * 70)
    print("RULE 4: Interaction Effects (residual analysis)")
    print("=" * 70)

    non_bug = np.array([not is_special_cents(r) for r in receipts])

    base_pred = (np.array([f_days(d) for d in days])
                 + np.array([g_miles(m) for m in miles])
                 + np.array([h_receipts(r) for r in receipts]))

    residuals = expected[non_bug] - base_pred[non_bug]
    d_nb, m_nb, r_nb = days[non_bug], miles[non_bug], receipts[non_bug]

    print(f"\n  Residual stats (after removing base components, non-bug cases):")
    print(f"    Mean:   ${residuals.mean():.2f}")
    print(f"    Median: ${np.median(residuals):.2f}")
    print(f"    Std:    ${residuals.std():.2f}")
    print(f"    Min:    ${residuals.min():.2f}")
    print(f"    Max:    ${residuals.max():.2f}")

    # Analyze by spending rate (receipts/day)
    r_pd = r_nb / d_nb
    print(f"\n  Residual by spending rate (receipts/day):")
    for lo, hi in [(0, 30), (30, 75), (75, 150), (150, 250), (250, 400), (400, 1000)]:
        mask = (r_pd >= lo) & (r_pd < hi)
        if mask.sum() >= 5:
            print(f"    ${lo:4d}-${hi:4d}/day (n={mask.sum():3d}): "
                  f"median_resid=${np.median(residuals[mask]):7.2f}, "
                  f"mean=${residuals[mask].mean():7.2f}")

    # Analyze by efficiency (miles/day)
    mi_pd = m_nb / d_nb
    print(f"\n  Residual by efficiency (miles/day):")
    for lo, hi in [(0, 25), (25, 50), (50, 100), (100, 175), (175, 300), (300, 600)]:
        mask = (mi_pd >= lo) & (mi_pd < hi)
        if mask.sum() >= 5:
            print(f"    {lo:4d}-{hi:4d} mi/day (n={mask.sum():3d}): "
                  f"median_resid=${np.median(residuals[mask]):7.2f}, "
                  f"mean=${residuals[mask].mean():7.2f}")

    # Analyze by days × miles combination
    print(f"\n  Residual by trip type (days × miles_category):")
    for d_lo, d_hi in [(1, 3), (4, 6), (7, 9), (10, 14)]:
        for m_lo, m_hi in [(0, 300), (300, 700), (700, 1400)]:
            mask = (d_nb >= d_lo) & (d_nb <= d_hi) & (m_nb >= m_lo) & (m_nb < m_hi)
            if mask.sum() >= 5:
                print(f"    {d_lo}-{d_hi}d, {m_lo}-{m_hi}mi (n={mask.sum():3d}): "
                      f"median=${np.median(residuals[mask]):7.2f}, "
                      f"mean=${residuals[mask].mean():7.2f}")

    # Fit a simple interaction model on residuals
    X = np.column_stack([
        d_nb * m_nb / 1000,        # days × miles
        d_nb * r_nb / 1000,        # days × receipts
        m_nb * r_nb / 100000,      # miles × receipts
        mi_pd,                      # efficiency
        r_pd,                       # spending rate
        np.ones(len(d_nb)),         # intercept
    ])
    interaction_labels = ['days×miles/1k', 'days×rcpt/1k', 'miles×rcpt/100k',
                          'mi/day', 'rcpt/day', 'intercept']

    coeffs, _, _, _ = np.linalg.lstsq(X, residuals, rcond=None)

    print(f"\n  Fitted interaction coefficients:")
    for label, coeff in zip(interaction_labels, coeffs):
        print(f"    {label:18s}: {coeff:10.4f}")

    pred_interaction = X @ coeffs
    final_residuals = residuals - pred_interaction

    print(f"\n  After interaction fit:")
    print(f"    Mean abs residual: ${np.abs(final_residuals).mean():.2f}")
    print(f"    Max abs residual:  ${np.abs(final_residuals).max():.2f}")

    return coeffs, interaction_labels


# ─── Step 5: Bug analysis ──────────────────────────────────────
def extract_bug_rules(days, miles, receipts, expected, f_days, g_miles, h_receipts):
    """
    Analyze the .49/.99 bug cases separately.
    """
    print("\n" + "=" * 70)
    print("RULE 5: The .49/.99 Rounding Bug")
    print("=" * 70)

    bug_mask = np.array([is_special_cents(r) for r in receipts])
    normal_mask = ~bug_mask

    base_pred = (np.array([f_days(d) for d in days])
                 + np.array([g_miles(m) for m in miles])
                 + np.array([h_receipts(r) for r in receipts]))

    bug_residuals = expected[bug_mask] - base_pred[bug_mask]
    normal_residuals = expected[normal_mask] - base_pred[normal_mask]

    print(f"\n  Bug cases (n={bug_mask.sum()}):")
    print(f"    Mean residual:   ${bug_residuals.mean():.2f}")
    print(f"    Normal mean:     ${normal_residuals.mean():.2f}")
    print(f"    Bug penalty:     ${bug_residuals.mean() - normal_residuals.mean():.2f}")

    # Bug effect by receipt amount
    bug_r = receipts[bug_mask]
    print(f"\n  Bug effect by receipt amount:")
    for lo, hi in [(0, 300), (300, 600), (600, 1000), (1000, 1500), (1500, 2500)]:
        mask = (bug_r >= lo) & (bug_r < hi)
        if mask.sum() > 0:
            print(f"    ${lo}-${hi} (n={mask.sum()}): avg residual=${bug_residuals[mask].mean():.2f}")

    # Fit bug effect: penalty = a + b * receipts
    bug_d = days[bug_mask]
    bug_m = miles[bug_mask]
    X = np.column_stack([np.ones(bug_mask.sum()), bug_r, bug_r**2 / 10000])
    coeffs, _, _, _ = np.linalg.lstsq(X, bug_residuals, rcond=None)
    print(f"\n  Bug effect formula: penalty = {coeffs[0]:.2f} + {coeffs[1]:.4f} * receipts + {coeffs[2]:.4f} * receipts²/10000")

    # Show each bug case
    print(f"\n  Individual bug cases:")
    for i in np.where(bug_mask)[0]:
        d, m, r, e = days[i], miles[i], receipts[i], expected[i]
        base = f_days(d) + g_miles(m) + h_receipts(r)
        print(f"    {d}d, {m:.0f}mi, ${r:.2f} -> ${e:.2f} (base=${base:.2f}, penalty=${e-base:.2f})")


# ─── Step 6: Summary ──────────────────────────────────────────
def generate_summary(f_params, g_params, h_params, interaction_coeffs, interaction_labels):
    """
    Generate a clean summary of all extracted rules.
    """
    print("\n" + "=" * 70)
    print("EXTRACTED RULES SUMMARY")
    print("=" * 70)

    a, b, c = f_params
    print(f"""
  RULE 1: Per-Diem
    formula: {a:.2f} * log(days + 1) + {b:.2f} * days + {c:.2f}
    meaning: ~${a*math.log(2):.0f} for day 1, diminishing ~${b:.0f}/day after that
    """)

    if len(g_params) == 5:
        r1, r2, r3, r4, offset = g_params
        print(f"""  RULE 2: Mileage (tiered)
    0-100 miles:    ${r1:.4f}/mile
    100-300 miles:  ${r2:.4f}/mile
    300-800 miles:  ${r3:.4f}/mile
    800+ miles:     ${r4:.4f}/mile
    base offset:    ${offset:.2f}
    """)

    if len(h_params) == 6:
        r1, r2, r3, r4, r5, offset = h_params
        print(f"""  RULE 3: Receipt Reimbursement (tiered)
    $0-$300:      {r1*100:.1f}% reimbursement (${r1:.4f}/dollar)
    $300-$600:    {r2*100:.1f}% reimbursement
    $600-$1200:   {r3*100:.1f}% reimbursement
    $1200-$1800:  {r4*100:.1f}% reimbursement
    $1800+:       {r5*100:.1f}% reimbursement
    base offset:  ${offset:.2f}
    """)

    print("  RULE 4: Interactions")
    for label, coeff in zip(interaction_labels, interaction_coeffs):
        print(f"    {label:18s}: {coeff:+.4f}")

    print(f"""
  RULE 5: .49/.99 Bug
    If receipt cents == .49 or .99:
      Large penalty that scales with receipt amount
      Low receipts ($0-300): ~$300-400 penalty
      High receipts ($1500+): ~$1000+ penalty

  OVERALL FORMULA:
    reimbursement = f(days) + g(miles) + h(receipts)
                  + interactions(days, miles, receipts)
                  + bug_penalty(receipts)  [if .49/.99]
    """)


if __name__ == '__main__':
    days, miles, receipts, expected = load_data()
    print(f"Loaded {len(days)} cases\n")

    # Step 1: Per-diem
    f_days, f_params = extract_per_diem(days, miles, receipts, expected)

    # Step 2: Mileage
    g_miles, g_params = extract_mileage(days, miles, receipts, expected, f_days)

    # Step 3: Receipts
    h_receipts, h_params = extract_receipts(days, miles, receipts, expected, f_days, g_miles)

    # Step 4: Interactions
    interaction_coeffs, interaction_labels = extract_interactions(
        days, miles, receipts, expected, f_days, g_miles, h_receipts)

    # Step 5: Bug
    extract_bug_rules(days, miles, receipts, expected, f_days, g_miles, h_receipts)

    # Step 6: Summary
    generate_summary(f_params, g_params, h_params, interaction_coeffs, interaction_labels)

    # Final: test the full additive model accuracy
    print("=" * 70)
    print("FULL ADDITIVE MODEL ACCURACY (no KNN, no interactions)")
    print("=" * 70)

    non_bug = np.array([not is_special_cents(r) for r in receipts])
    base_pred = (np.array([f_days(d) for d in days])
                 + np.array([g_miles(m) for m in miles])
                 + np.array([h_receipts(r) for r in receipts]))

    errors_nb = np.abs(expected[non_bug] - base_pred[non_bug])
    print(f"\n  Non-bug cases ({non_bug.sum()}):")
    print(f"    MAE:        ${errors_nb.mean():.2f}")
    print(f"    Median:     ${np.median(errors_nb):.2f}")
    print(f"    Max:        ${errors_nb.max():.2f}")
    print(f"    Within $50: {(errors_nb < 50).sum()}/{non_bug.sum()}")
    print(f"    Within $100: {(errors_nb < 100).sum()}/{non_bug.sum()}")
