#!/usr/bin/env python3
"""
Jointly optimize all business rule parameters.

Define the full rule-based formula with named, interpretable parameters,
then use scipy.optimize to find the best-fit values all at once.
"""

import json
import os
import math
import numpy as np
from scipy.optimize import minimize

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

# ─── Named parameter structure ─────────────────────────────────
PARAM_NAMES = [
    # Per-diem (3 params)
    'pd_log',       # coefficient for log(days+1)
    'pd_linear',    # coefficient for days
    'pd_base',      # constant

    # Mileage tiers (5 params)
    'mi_rate1',     # $/mile for 0-100 mi
    'mi_rate2',     # $/mile for 100-300 mi
    'mi_rate3',     # $/mile for 300-800 mi
    'mi_rate4',     # $/mile for 800+ mi
    'mi_base',      # mileage base offset

    # Receipt tiers (6 params)
    'rc_rate1',     # rate for $0-300
    'rc_rate2',     # rate for $300-600
    'rc_rate3',     # rate for $600-1200
    'rc_rate4',     # rate for $1200-1800
    'rc_rate5',     # rate for $1800+
    'rc_base',      # receipt base offset

    # Interactions (6 params)
    'ix_dm',        # days × miles / 1000
    'ix_dr',        # days × receipts / 1000
    'ix_mr',        # miles × receipts / 100000
    'ix_eff',       # miles/day
    'ix_spend',     # receipts/day
    'ix_base',      # interaction constant

    # Bug effect (3 params)
    'bug_base',     # constant penalty when bug triggered
    'bug_linear',   # penalty per dollar of receipts
    'bug_quad',     # penalty per (receipts²/10000)
]

# Initial guesses based on prior analysis
INIT_PARAMS = [
    # Per-diem
    330.0, 11.0, 480.0,
    # Mileage
    0.45, 0.35, 0.55, 0.30, 0.0,
    # Receipts
    0.10, 0.50, 0.80, 0.25, -0.10, -50.0,
    # Interactions
    3.0, -5.0, -8.0, -0.10, 0.05, -30.0,
    # Bug
    100.0, -0.80, 1.50,
]

def compute_reimbursement(d, m, r, p):
    """Compute reimbursement from named parameters."""
    # Per-diem
    per_diem = p[0] * math.log(d + 1) + p[1] * d + p[2]

    # Mileage (tiered)
    mileage = (p[3] * min(m, 100)
               + p[4] * min(max(0, m - 100), 200)
               + p[5] * min(max(0, m - 300), 500)
               + p[6] * max(0, m - 800)
               + p[7])

    # Receipts (tiered)
    receipt = (p[8] * min(r, 300)
               + p[9] * min(max(0, r - 300), 300)
               + p[10] * min(max(0, r - 600), 600)
               + p[11] * min(max(0, r - 1200), 600)
               + p[12] * max(0, r - 1800)
               + p[13])

    # Interactions
    mi_pd = m / d
    r_pd = r / d
    interact = (p[14] * d * m / 1000
                + p[15] * d * r / 1000
                + p[16] * m * r / 100000
                + p[17] * mi_pd
                + p[18] * r_pd
                + p[19])

    # Bug
    bug = 0.0
    if is_special_cents(r):
        bug = p[20] + p[21] * r + p[22] * r * r / 10000

    return max(0.0, per_diem + mileage + receipt + interact + bug)

def main():
    data = load_data()
    n = len(data)

    def objective(p):
        total = 0.0
        for d, m, r, e in data:
            pred = compute_reimbursement(d, m, r, p)
            total += abs(pred - e)
        return total / n

    x0 = np.array(INIT_PARAMS)
    print(f"Initial MAE: ${objective(x0):.2f}")
    print(f"Optimizing {len(x0)} parameters...")

    # Multiple rounds of Nelder-Mead for better convergence
    best_x = x0
    best_mae = objective(x0)

    for round_num in range(5):
        result = minimize(objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 0.0001,
                                   'fatol': 0.0001, 'adaptive': True})
        if result.fun < best_mae:
            best_mae = result.fun
            best_x = result.x
        print(f"  Round {round_num+1}: MAE=${result.fun:.2f}")

    p = best_x
    print(f"\nFinal MAE: ${best_mae:.2f}")

    # Print extracted rules
    print("\n" + "=" * 70)
    print("EXTRACTED BUSINESS RULES")
    print("=" * 70)

    print(f"""
RULE 1: Per-Diem Calculation
  formula: {p[0]:.1f} × log(days + 1) + {p[1]:.1f} × days + {p[2]:.1f}
  Example values:
    1 day:  ${p[0]*math.log(2) + p[1]*1 + p[2]:.0f}
    3 days: ${p[0]*math.log(4) + p[1]*3 + p[2]:.0f}
    5 days: ${p[0]*math.log(6) + p[1]*5 + p[2]:.0f}
    7 days: ${p[0]*math.log(8) + p[1]*7 + p[2]:.0f}
    10 days: ${p[0]*math.log(11) + p[1]*10 + p[2]:.0f}
    14 days: ${p[0]*math.log(15) + p[1]*14 + p[2]:.0f}

RULE 2: Mileage Reimbursement (tiered)
  0-100 miles:    ${p[3]:.4f}/mile
  100-300 miles:  ${p[4]:.4f}/mile
  300-800 miles:  ${p[5]:.4f}/mile
  800+ miles:     ${p[6]:.4f}/mile
  Base offset:    ${p[7]:.2f}
  Example: 500 miles = ${p[3]*100 + p[4]*200 + p[5]*200 + p[7]:.0f}

RULE 3: Receipt Reimbursement (tiered)
  $0-$300:      {p[8]*100:.1f}% (${p[8]:.4f}/dollar)
  $300-$600:    {p[9]*100:.1f}% (${p[9]:.4f}/dollar)
  $600-$1200:   {p[10]*100:.1f}% (${p[10]:.4f}/dollar)
  $1200-$1800:  {p[11]*100:.1f}% (${p[11]:.4f}/dollar)
  $1800+:       {p[12]*100:.1f}% (${p[12]:.4f}/dollar)
  Base offset:  ${p[13]:.2f}
  Example: $1000 receipts = ${p[8]*300 + p[9]*300 + p[10]*400 + p[13]:.0f}

RULE 4: Interaction Effects
  days × miles / 1000:    {p[14]:+.4f}  (long trips + high miles = bonus)
  days × receipts / 1000: {p[15]:+.4f}  (long trips + high spending = penalty)
  miles × receipts / 100k:{p[16]:+.4f}  (high miles + high spending = penalty)
  miles/day efficiency:    {p[17]:+.4f}  (higher efficiency = slight penalty/bonus)
  receipts/day spending:   {p[18]:+.4f}  (higher daily spend = slight effect)
  Base:                    {p[19]:+.2f}

RULE 5: .49/.99 Bug
  Triggered when receipt cents = .49 or .99
  penalty = {p[20]:.1f} + {p[21]:.4f} × receipts + {p[22]:.4f} × receipts²/10000
  Example penalties:
    $100 receipts:  ${p[20] + p[21]*100 + p[22]*100*100/10000:.0f}
    $500 receipts:  ${p[20] + p[21]*500 + p[22]*500*500/10000:.0f}
    $1000 receipts: ${p[20] + p[21]*1000 + p[22]*1000*1000/10000:.0f}
    $2000 receipts: ${p[20] + p[21]*2000 + p[22]*2000*2000/10000:.0f}
""")

    # Detailed accuracy
    errors = []
    bug_errors = []
    normal_errors = []
    for d, m, r, e in data:
        pred = compute_reimbursement(d, m, r, p)
        err = abs(pred - e)
        errors.append(err)
        if is_special_cents(r):
            bug_errors.append(err)
        else:
            normal_errors.append(err)

    errors = np.array(errors)
    normal_errors = np.array(normal_errors)
    bug_errors = np.array(bug_errors)

    print(f"ACCURACY:")
    print(f"  Overall (n={n}):     MAE=${errors.mean():.2f}, median=${np.median(errors):.2f}, max=${errors.max():.2f}")
    print(f"  Normal (n={len(normal_errors)}):  MAE=${normal_errors.mean():.2f}, median=${np.median(normal_errors):.2f}")
    print(f"  Bug (n={len(bug_errors)}):     MAE=${bug_errors.mean():.2f}, median=${np.median(bug_errors):.2f}")
    print(f"  Within $10:  {(errors < 10).sum()}/{n}")
    print(f"  Within $50:  {(errors < 50).sum()}/{n}")
    print(f"  Within $100: {(errors < 100).sum()}/{n}")

    # Print the parameter vector for use in code
    print(f"\n# For use in code:")
    print(f"PARAMS = {[round(float(x), 6) for x in p]}")

    # Worst 10
    worst = np.argsort(errors)[-10:]
    print(f"\nWorst 10 cases:")
    for idx in worst:
        d, m, r, e = data[idx]
        pred = compute_reimbursement(d, m, r, p)
        sp = " *BUG*" if is_special_cents(r) else ""
        print(f"  {d}d, {m:.0f}mi, ${r:.2f} -> expected ${e:.2f}, got ${pred:.2f}, err=${abs(pred-e):.2f}{sp}")


if __name__ == '__main__':
    main()
