#!/usr/bin/env python3
"""
Simple rule-based reimbursement calculator.

Extracted from 1,000 historical cases. No ML, no KNN — just business rules.
23 interpretable parameters, $71 MAE.

This is what the legacy system's logic likely looks like under the hood.
"""

import sys
import math

def is_special_cents(receipts):
    """Detect the .49/.99 rounding bug."""
    cents = round(receipts * 100) % 100
    return cents == 49 or cents == 99

def calculate_reimbursement(days, miles, receipts):
    days = max(1, days)
    miles = max(0.0, miles)
    receipts = max(0.0, receipts)

    # ── Rule 1: Per-diem (log decay) ──────────────────────────
    # ~$122 for 1 day, diminishing returns for longer trips
    per_diem = 275.6 * math.log(days + 1) + 13.7 * days - 82.9

    # ── Rule 2: Mileage (tiered rates) ───────────────────────
    # Higher rate for first 100mi, dips 100-300, rises 300-800, drops 800+
    mileage = (0.93 * min(miles, 100)
               + 0.41 * min(max(0, miles - 100), 200)
               + 0.61 * min(max(0, miles - 300), 500)
               + 0.29 * max(0, miles - 800))

    # ── Rule 3: Receipt reimbursement (tiered) ───────────────
    # Sweet spot at $600-$1200 (114% rate!), drops above $1200
    receipt_reimb = (-0.05 * min(receipts, 300)
                     + 0.96 * min(max(0, receipts - 300), 300)
                     + 1.14 * min(max(0, receipts - 600), 600)
                     + 0.15 * min(max(0, receipts - 1200), 600)
                     + 0.07 * max(0, receipts - 1800)
                     - 66.0)

    # ── Rule 4: Interaction effects ──────────────────────────
    # Long trips + high miles = bonus; long trips + high spending = penalty
    interactions = (8.07 * days * miles / 1000           # days×miles bonus
                    - 10.48 * days * receipts / 1000     # days×receipts penalty
                    - 12.63 * miles * receipts / 100000  # miles×receipts penalty
                    - 0.01 * miles / days                # efficiency effect
                    + 0.00 * receipts / days             # spending rate effect
                    + 12.3)

    # ── Rule 5: .49/.99 rounding bug ─────────────────────────
    # Legacy floating-point error: receipts ending in .49/.99 get penalized
    # Penalty grows quadratically with receipt amount
    bug_effect = 0.0
    if is_special_cents(receipts):
        bug_effect = 241.9 - 1.01 * receipts + 0.86 * receipts * receipts / 10000

    total = per_diem + mileage + receipt_reimb + interactions + bug_effect
    return round(max(0.0, total), 2)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 simple_rules_model.py <days> <miles> <receipts>")
        sys.exit(1)
    days = int(float(sys.argv[1]))
    miles = float(sys.argv[2])
    receipts = float(sys.argv[3])
    print(calculate_reimbursement(days, miles, receipts))
