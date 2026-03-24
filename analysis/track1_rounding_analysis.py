#!/usr/bin/env python3
"""Track 1: Rounding pattern analysis for the legacy reimbursement system."""

import json
import math
import sys
import os
from collections import Counter

# Add parent dir to path so we can import the model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_rules_model import calculate_reimbursement

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public_cases.json")

def is_special_cents(r):
    cents = round(r * 100) % 100
    return cents == 49 or cents == 99


def load_cases():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    cases = []
    for c in raw:
        inp = c["input"]
        cases.append((
            inp["trip_duration_days"],
            inp["miles_traveled"],
            inp["total_receipts_amount"],
            c["expected_output"],
        ))
    return cases


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def analyze_cent_distribution(cases):
    section("1. Output Cent Distribution")
    cents = [round(c[3] * 100) % 100 for c in cases]
    counter = Counter(cents)
    print(f"Total cases: {len(cases)}")
    print(f"Unique cent values: {len(counter)}")
    print(f"\nTop 20 most common cents:")
    for val, count in counter.most_common(20):
        print(f"  .{val:02d}  -> {count:4d} occurrences ({100*count/len(cases):.1f}%)")

    # Uniformity check
    expected = len(cases) / 100
    print(f"\nExpected per cent if uniform: {expected:.1f}")
    max_count = counter.most_common(1)[0][1]
    min_count = min(counter.values())
    print(f"Actual range: {min_count} - {max_count}")


def analyze_precision(cases):
    section("2. Output Precision")
    outputs = [c[3] for c in cases]
    cents = [round(o * 100) % 100 for o in outputs]

    end_00 = sum(1 for c in cents if c == 0)
    end_50 = sum(1 for c in cents if c == 50)
    end_25_75 = sum(1 for c in cents if c in (25, 75))
    print(f"Outputs ending in .00: {end_00}")
    print(f"Outputs ending in .50: {end_50}")
    print(f"Outputs ending in .25 or .75: {end_25_75}")

    # Check minimum resolution
    for res_name, divisor in [("$0.01", 1), ("$0.05", 5), ("$0.10", 10), ("$0.25", 25), ("$0.50", 50), ("$1.00", 100)]:
        count = sum(1 for c in cents if c % divisor == 0)
        print(f"  Multiples of {res_name}: {count}/{len(cases)} ({100*count/len(cases):.1f}%)")

    unique_cents = sorted(set(cents))
    print(f"\nUnique cent values ({len(unique_cents)}): {unique_cents[:30]}{'...' if len(unique_cents)>30 else ''}")


def analyze_bug_mechanism(cases):
    section("3. Bug Mechanism Analysis")

    TIERS = [300, 600, 1200, 1800]
    bug_cases = [(d, m, r, o) for d, m, r, o in cases if is_special_cents(r)]
    non_bug = [(d, m, r, o) for d, m, r, o in cases if not is_special_cents(r)]

    print(f"Bug cases (receipts ending .49/.99): {len(bug_cases)}")
    print(f"Non-bug cases: {len(non_bug)}\n")

    print(f"{'days':>4} {'miles':>7} {'receipts':>10} {'output':>10} | {'floor(r)':>8} {'ceil(r)':>8} {'round(r)':>8} | {'tier_cross':>10} | {'penalty':>10}")
    print("-" * 110)

    for d, m, r, o in sorted(bug_cases, key=lambda x: x[2]):
        fr = math.floor(r)
        cr = math.ceil(r)
        rr = round(r)
        rd50 = round(r * 2) / 2  # nearest 0.50

        # Check tier crossing
        tier_cross = ""
        for t in TIERS:
            if fr < t <= cr:
                tier_cross = f"<{t}>"
            elif fr < t <= rr:
                tier_cross = f"~{t}~"

        # Estimate penalty: find similar non-bug cases (same days, similar miles)
        neighbors = [
            (nd, nm, nr, no)
            for nd, nm, nr, no in non_bug
            if nd == d and abs(nm - m) < 50 and abs(nr - r) < 200
        ]
        if neighbors:
            avg_neighbor_output = sum(no for _, _, _, no in neighbors) / len(neighbors)
            penalty = o - avg_neighbor_output
        else:
            penalty = float('nan')

        print(f"{d:4d} {m:7.1f} {r:10.2f} {o:10.2f} | {fr:8d} {cr:8d} {rr:8d} | {tier_cross:>10} | {penalty:10.2f}")

    # Summary stats
    print(f"\nTier boundary analysis for bug cases:")
    for t in TIERS:
        crossing = [(d, m, r, o) for d, m, r, o in bug_cases if math.floor(r) < t <= math.ceil(r)]
        near = [(d, m, r, o) for d, m, r, o in bug_cases if abs(r - t) < 2]
        print(f"  Tier ${t}: {len(crossing)} cross boundary on floor/ceil, {len(near)} within $2 of boundary")


def analyze_factors(cases):
    section("4. Factor Analysis (non-bug cases)")
    non_bug = [(d, m, r, o) for d, m, r, o in cases if not is_special_cents(r)]

    # Per-day rates
    print("Output / days  (per-diem rate):")
    by_days = {}
    for d, m, r, o in non_bug:
        by_days.setdefault(d, []).append((m, r, o))

    for d in sorted(by_days.keys()):
        entries = by_days[d]
        rates = [o / d for _, _, o in entries]
        avg_rate = sum(rates) / len(rates)
        min_rate = min(rates)
        max_rate = max(rates)
        print(f"  days={d:2d}: n={len(entries):3d}  avg={avg_rate:8.2f}  min={min_rate:8.2f}  max={max_rate:8.2f}")

    # Mileage rate after removing per-diem estimate
    print("\nMileage rate = (output - per_diem_est) / miles:")
    print("  per_diem_est = 275.6 * log(days+1) + 13.7 * days - 82.9")

    mile_rates = []
    for d, m, r, o in non_bug:
        if m < 1:
            continue
        per_diem = 275.6 * math.log(d + 1) + 13.7 * d - 82.9
        residual = o - per_diem
        rate = residual / m
        mile_rates.append((d, m, r, o, rate))

    # Show distribution of mileage rates
    rates_only = [x[4] for x in mile_rates]
    rates_only.sort()
    n = len(rates_only)
    print(f"  n={n}")
    print(f"  median: {rates_only[n//2]:.4f}")
    print(f"  p10:    {rates_only[n//10]:.4f}")
    print(f"  p25:    {rates_only[n//4]:.4f}")
    print(f"  p75:    {rates_only[3*n//4]:.4f}")
    print(f"  p90:    {rates_only[9*n//10]:.4f}")

    # Check common rate values
    for target in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.585, 0.60, 0.65, 0.70]:
        close = sum(1 for r in rates_only if abs(r - target) < 0.05)
        print(f"  within 0.05 of {target:.3f}: {close}/{n}")


def analyze_residual_cents(cases):
    section("5. Residual Cent Patterns")
    non_bug = [(d, m, r, o) for d, m, r, o in cases if not is_special_cents(r)]

    residuals = []
    for d, m, r, o in non_bug:
        pred = calculate_reimbursement(d, m, r)
        resid = pred - o
        residuals.append((d, m, r, o, pred, resid))

    resid_vals = [x[5] for x in residuals]
    abs_resid = [abs(x) for x in resid_vals]
    print(f"Residual stats (predicted - actual), n={len(residuals)}:")
    print(f"  mean:   {sum(resid_vals)/len(resid_vals):8.2f}")
    print(f"  median: {sorted(resid_vals)[len(resid_vals)//2]:8.2f}")
    print(f"  MAE:    {sum(abs_resid)/len(abs_resid):8.2f}")
    print(f"  max:    {max(abs_resid):8.2f}")

    # Residual cent distribution
    resid_cents = [round(abs(r) * 100) % 100 for _, _, _, _, _, r in residuals]
    counter = Counter(resid_cents)
    print(f"\nResidual cent distribution (top 15):")
    for val, count in counter.most_common(15):
        print(f"  .{val:02d}  -> {count:4d}")

    # Check if residuals correlate with input cent values
    print(f"\nMean residual by receipt cent value:")
    by_receipt_cent = {}
    for d, m, r, o, pred, resid in residuals:
        rc = round(r * 100) % 100
        by_receipt_cent.setdefault(rc, []).append(resid)

    # Show cents with largest mean residual
    cent_means = [(cent, sum(vs)/len(vs), len(vs)) for cent, vs in by_receipt_cent.items() if len(vs) >= 3]
    cent_means.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top 15 by |mean residual| (min 3 samples):")
    for cent, mean_r, n in cent_means[:15]:
        print(f"    receipt cents .{cent:02d}: mean_resid={mean_r:8.2f}  n={n}")

    # Check if residuals correlate with output cent values
    print(f"\nMean residual by OUTPUT cent value:")
    by_output_cent = {}
    for d, m, r, o, pred, resid in residuals:
        oc = round(o * 100) % 100
        by_output_cent.setdefault(oc, []).append(resid)

    cent_means = [(cent, sum(vs)/len(vs), len(vs)) for cent, vs in by_output_cent.items() if len(vs) >= 3]
    cent_means.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top 15 by |mean residual| (min 3 samples):")
    for cent, mean_r, n in cent_means[:15]:
        print(f"    output cents .{cent:02d}: mean_resid={mean_r:8.2f}  n={n}")


def main():
    cases = load_cases()
    print(f"Loaded {len(cases)} cases from {DATA_PATH}")

    analyze_cent_distribution(cases)
    analyze_precision(cases)
    analyze_bug_mechanism(cases)
    analyze_factors(cases)
    analyze_residual_cents(cases)

    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
