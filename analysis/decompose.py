#!/usr/bin/env python3
"""
Decompose the reimbursement function into additive components.
Goal: Isolate f(days), g(miles), h(receipts), and interaction terms.
"""

import json
import os
import numpy as np
from collections import defaultdict

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

def analyze_per_diem(days, miles, receipts, expected):
    """Isolate per-diem effect by grouping by days, controlling for miles/receipts."""
    print("=" * 60)
    print("PER-DIEM ANALYSIS: f(days)")
    print("=" * 60)

    # Group by day count
    for d in sorted(set(days)):
        mask = (days == d) & np.array([not is_special_cents(r) for r in receipts])
        if mask.sum() < 5:
            continue
        avg_reimb = expected[mask].mean()
        avg_miles = miles[mask].mean()
        avg_rcpt = receipts[mask].mean()
        per_day = avg_reimb / d
        print(f"  {d:2d} days: n={mask.sum():3d}, avg_reimb=${avg_reimb:7.2f}, "
              f"per_day=${per_day:6.2f}, avg_miles={avg_miles:6.1f}, avg_rcpt=${avg_rcpt:7.2f}")

    # Control for similar miles and receipts ranges
    print("\n  Controlled analysis (miles 200-400, receipts 400-800):")
    for d in sorted(set(days)):
        mask = ((days == d) & (miles >= 200) & (miles <= 400) &
                (receipts >= 400) & (receipts <= 800) &
                np.array([not is_special_cents(r) for r in receipts]))
        if mask.sum() < 3:
            continue
        avg_reimb = expected[mask].mean()
        per_day = avg_reimb / d
        print(f"    {d:2d} days: n={mask.sum():3d}, avg_reimb=${avg_reimb:7.2f}, per_day=${per_day:6.2f}")

def analyze_mileage(days, miles, receipts, expected):
    """Isolate mileage effect by binning miles, controlling for days/receipts."""
    print("\n" + "=" * 60)
    print("MILEAGE ANALYSIS: g(miles)")
    print("=" * 60)

    # Bin miles
    bins = [0, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1400]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        mask = ((miles >= lo) & (miles < hi) &
                np.array([not is_special_cents(r) for r in receipts]))
        if mask.sum() < 3:
            continue
        avg_reimb = expected[mask].mean()
        avg_days = days[mask].mean()
        avg_rcpt = receipts[mask].mean()
        avg_miles = miles[mask].mean()
        print(f"  miles {lo:4d}-{hi:4d}: n={mask.sum():3d}, avg_reimb=${avg_reimb:7.2f}, "
              f"avg_days={avg_days:.1f}, avg_rcpt=${avg_rcpt:7.2f}")

    # Control for days=5 and receipts 400-800
    print("\n  Controlled analysis (days 4-6, receipts 400-800):")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        mask = ((miles >= lo) & (miles < hi) & (days >= 4) & (days <= 6) &
                (receipts >= 400) & (receipts <= 800) &
                np.array([not is_special_cents(r) for r in receipts]))
        if mask.sum() < 2:
            continue
        avg_reimb = expected[mask].mean()
        avg_miles = miles[mask].mean()
        # Estimate marginal mileage contribution
        print(f"    miles {lo:4d}-{hi:4d}: n={mask.sum():3d}, avg_reimb=${avg_reimb:7.2f}, avg_mi={avg_miles:.0f}")

def analyze_receipts(days, miles, receipts, expected):
    """Isolate receipt effect by binning receipts, controlling for days/miles."""
    print("\n" + "=" * 60)
    print("RECEIPT ANALYSIS: h(receipts)")
    print("=" * 60)

    # Bin receipts
    bins = [0, 100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2600]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        mask = ((receipts >= lo) & (receipts < hi) &
                np.array([not is_special_cents(r) for r in receipts]))
        if mask.sum() < 3:
            continue
        avg_reimb = expected[mask].mean()
        avg_days = days[mask].mean()
        avg_miles = miles[mask].mean()
        print(f"  rcpt ${lo:5d}-${hi:5d}: n={mask.sum():3d}, avg_reimb=${avg_reimb:7.2f}, "
              f"avg_days={avg_days:.1f}, avg_miles={avg_miles:.0f}")

    # Control for days=5, miles 300-500
    print("\n  Controlled analysis (days 4-6, miles 300-600):")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        mask = ((receipts >= lo) & (receipts < hi) & (days >= 4) & (days <= 6) &
                (miles >= 300) & (miles <= 600) &
                np.array([not is_special_cents(r) for r in receipts]))
        if mask.sum() < 2:
            continue
        avg_reimb = expected[mask].mean()
        avg_rcpt = receipts[mask].mean()
        print(f"    rcpt ${lo:5d}-${hi:5d}: n={mask.sum():3d}, avg_reimb=${avg_reimb:7.2f}, avg_rcpt=${avg_rcpt:.0f}")

def analyze_interactions(days, miles, receipts, expected):
    """Analyze interaction terms: spending rate, efficiency, cross-terms."""
    print("\n" + "=" * 60)
    print("INTERACTION ANALYSIS")
    print("=" * 60)

    non_bug = np.array([not is_special_cents(r) for r in receipts])

    # Spending rate (receipts/day)
    r_pd = receipts / days
    print("\n  Spending Rate (receipts/day):")
    bins = [0, 30, 50, 75, 100, 150, 200, 300, 500, 1000]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        mask = (r_pd >= lo) & (r_pd < hi) & non_bug
        if mask.sum() < 3:
            continue
        avg_reimb = expected[mask].mean()
        avg_days = days[mask].mean()
        avg_miles = miles[mask].mean()
        avg_rcpt = receipts[mask].mean()
        print(f"    ${lo:4d}-${hi:4d}/day: n={mask.sum():3d}, avg_reimb=${avg_reimb:7.2f}, "
              f"d={avg_days:.1f}, mi={avg_miles:.0f}, r=${avg_rcpt:.0f}")

    # Efficiency (miles/day)
    mi_pd = miles / days
    print("\n  Efficiency (miles/day):")
    bins = [0, 25, 50, 75, 100, 150, 200, 300, 500]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        mask = (mi_pd >= lo) & (mi_pd < hi) & non_bug
        if mask.sum() < 3:
            continue
        avg_reimb = expected[mask].mean()
        avg_days = days[mask].mean()
        avg_miles = miles[mask].mean()
        print(f"    {lo:4d}-{hi:4d} mi/d: n={mask.sum():3d}, avg_reimb=${avg_reimb:7.2f}, "
              f"d={avg_days:.1f}, mi={avg_miles:.0f}")

def analyze_bug(days, miles, receipts, expected):
    """Analyze the .49/.99 bug cases."""
    print("\n" + "=" * 60)
    print(".49/.99 BUG ANALYSIS")
    print("=" * 60)

    bug_mask = np.array([is_special_cents(r) for r in receipts])
    normal_mask = ~bug_mask

    print(f"\n  Bug cases: {bug_mask.sum()}")
    print(f"  Normal cases: {normal_mask.sum()}")

    # For each bug case, find the nearest normal case and compare
    print("\n  Bug case comparisons (bug vs nearest normal):")
    bug_indices = np.where(bug_mask)[0]

    diffs = []
    for idx in bug_indices:
        d, m, r, e = days[idx], miles[idx], receipts[idx], expected[idx]
        # Find nearest normal case
        normal_dists = np.abs(days[normal_mask] - d) / 14 * 2 + \
                       np.abs(miles[normal_mask] - m) / 1400 + \
                       np.abs(receipts[normal_mask] - r) / 2600
        nearest_idx = np.where(normal_mask)[0][np.argmin(normal_dists)]
        nd, nm, nr, ne = days[nearest_idx], miles[nearest_idx], receipts[nearest_idx], expected[nearest_idx]
        diff = e - ne
        diffs.append(diff)
        if abs(diff) > 200:
            print(f"    Bug: {d}d, {m:.0f}mi, ${r:.2f} -> ${e:.2f}")
            print(f"    Near: {nd}d, {nm:.0f}mi, ${nr:.2f} -> ${ne:.2f}  (diff=${diff:.2f})")

    diffs = np.array(diffs)
    print(f"\n  Bug effect stats: mean=${diffs.mean():.2f}, median=${np.median(diffs):.2f}, "
          f"min=${diffs.min():.2f}, max=${diffs.max():.2f}")

    # Bug effect vs receipt amount
    print("\n  Bug effect by receipt amount:")
    for lo, hi in [(0, 300), (300, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, 2600)]:
        mask = bug_mask & (receipts >= lo) & (receipts < hi)
        if mask.sum() == 0:
            continue
        avg_reimb = expected[mask].mean()
        # Average of nearest normal cases in same receipt range
        nmask = normal_mask & (receipts >= lo) & (receipts < hi)
        if nmask.sum() == 0:
            continue
        avg_normal = expected[nmask].mean()
        print(f"    rcpt ${lo}-${hi}: bug_avg=${avg_reimb:.2f}, normal_avg=${avg_normal:.2f}, "
              f"diff=${avg_reimb - avg_normal:.2f} (n_bug={mask.sum()}, n_normal={nmask.sum()})")

def fit_simple_additive(days, miles, receipts, expected):
    """Fit a simple additive model to get baseline parameters."""
    print("\n" + "=" * 60)
    print("SIMPLE ADDITIVE MODEL FIT")
    print("=" * 60)

    non_bug = np.array([not is_special_cents(r) for r in receipts])
    d, m, r, e = days[non_bug], miles[non_bug], receipts[non_bug], expected[non_bug]
    n = len(d)

    # Build feature matrix for piecewise model
    # f(days): per-diem with log decay
    # g(miles): tiered mileage
    # h(receipts): tiered receipts
    X = np.column_stack([
        # Per-diem
        d,
        np.log(d + 1),

        # Mileage tiers
        np.minimum(m, 100),
        np.minimum(np.maximum(0, m - 100), 200),
        np.maximum(0, m - 300),
        np.maximum(0, m - 800),

        # Receipt tiers
        np.minimum(r, 300),
        np.minimum(np.maximum(0, r - 300), 300),
        np.minimum(np.maximum(0, r - 600), 600),
        np.maximum(0, r - 1200),
        np.maximum(0, r - 1800),

        # Interactions
        m / d,  # efficiency
        r / d,  # spending rate
        d * m / 1000,
        d * r / 1000,
        m * r / 100000,

        # Intercept
        np.ones(n),
    ])

    # Least squares fit
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, e, rcond=None)

    pred = X @ coeffs
    errors = np.abs(pred - e)

    labels = [
        'days (linear)', 'log(days+1)',
        'miles [0-100]', 'miles [100-300]', 'miles [300-800]', 'miles [800+]',
        'rcpt [0-300]', 'rcpt [300-600]', 'rcpt [600-1200]', 'rcpt [1200-1800]', 'rcpt [1800+]',
        'miles/day', 'rcpt/day', 'days*miles/1k', 'days*rcpt/1k', 'miles*rcpt/100k',
        'intercept'
    ]

    print("\n  Fitted coefficients:")
    for label, coeff in zip(labels, coeffs):
        print(f"    {label:20s}: {coeff:10.4f}")

    print(f"\n  Non-bug cases (n={n}):")
    print(f"    Mean abs error:  ${errors.mean():.2f}")
    print(f"    Median abs error: ${np.median(errors):.2f}")
    print(f"    Max abs error:   ${errors.max():.2f}")
    print(f"    Exact (±$0.01):  {(errors < 0.01).sum()}")
    print(f"    Close (±$1.00):  {(errors < 1.0).sum()}")
    print(f"    Within $10:      {(errors < 10).sum()}")
    print(f"    Within $50:      {(errors < 50).sum()}")
    print(f"    Within $100:     {(errors < 100).sum()}")

    # Show worst cases
    worst = np.argsort(errors)[-10:]
    print("\n  Worst 10 cases:")
    for idx in worst:
        print(f"    {d[idx]}d, {m[idx]:.0f}mi, ${r[idx]:.2f} -> expected ${e[idx]:.2f}, got ${pred[idx]:.2f}, err=${errors[idx]:.2f}")

    return coeffs

if __name__ == '__main__':
    days, miles, receipts, expected = load_data()

    print(f"Loaded {len(days)} cases")
    print(f"Days: {days.min()}-{days.max()}, Miles: {miles.min():.0f}-{miles.max():.0f}, "
          f"Receipts: ${receipts.min():.2f}-${receipts.max():.2f}")
    print(f"Reimbursement: ${expected.min():.2f}-${expected.max():.2f}")
    print()

    analyze_per_diem(days, miles, receipts, expected)
    analyze_mileage(days, miles, receipts, expected)
    analyze_receipts(days, miles, receipts, expected)
    analyze_interactions(days, miles, receipts, expected)
    analyze_bug(days, miles, receipts, expected)
    coeffs = fit_simple_additive(days, miles, receipts, expected)
