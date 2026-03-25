#!/usr/bin/env python3
"""
Test suite for the reimbursement model.
- Quick regression test: 10 known cases (< 1 second)
- Full eval: all 1000 public cases with scoring (< 2 seconds)
"""

import json
import sys
import time

from approach3_ridge_features import calculate_reimbursement, is_special_cents

# 10 known cases for quick regression testing
# Includes: normal cases, .49/.99 bug cases, edge cases
REGRESSION_CASES = [
    # (days, miles, receipts, expected, description)
    (3, 93, 1.42, 364.51, "low receipts, short trip"),
    (1, 55, 3.60, 126.06, "1-day minimal"),
    (5, 250, 150.75, None, "normal 5-day trip (no exact expected, just smoke test)"),
    (4, 69, 2321.49, 322.00, ".49 bug - high receipts severely penalized"),
    (8, 795, 1645.99, 644.69, ".99 bug - high receipts severely penalized"),
    (1, 1082, 1809.49, 446.94, ".49 bug - 1-day high miles"),
    (14, 481, 939.99, 877.17, ".99 bug - long trip"),
    (1, 1002, 2320.13, 1475.40, "1-day high miles high receipts (NOT .49/.99)"),
    (7, 1006, 1181.33, 2279.82, "7-day high miles moderate receipts"),
    (14, 94, 105.94, 1180.63, "14-day low miles low receipts"),
]

def test_special_cents():
    """Test the .49/.99 detection function."""
    assert is_special_cents(2321.49) == True, "2321.49 should be special"
    assert is_special_cents(1645.99) == True, "1645.99 should be special"
    assert is_special_cents(1809.49) == True, "1809.49 should be special"
    assert is_special_cents(100.00) == False, "100.00 should NOT be special"
    assert is_special_cents(100.50) == False, "100.50 should NOT be special"
    assert is_special_cents(100.48) == False, "100.48 should NOT be special"
    assert is_special_cents(0.49) == True, "0.49 should be special"
    assert is_special_cents(0.99) == True, "0.99 should be special"
    print("  PASS: is_special_cents()")

def test_no_crash():
    """Test that edge inputs don't crash."""
    # These should all return a non-negative number without crashing
    cases = [
        (1, 0, 0),       # zero miles and receipts
        (1, 5, 1.42),    # minimum-like values
        (14, 1317, 2503), # maximum-like values
        (1, 1, 0.01),    # tiny receipts
        (1, 1000, 0.01), # high miles tiny receipts
        (14, 5, 2500),   # long trip high receipts low miles
    ]
    for days, miles, receipts in cases:
        result = calculate_reimbursement(days, miles, receipts)
        assert isinstance(result, (int, float)), f"Expected number, got {type(result)} for ({days}, {miles}, {receipts})"
        assert result >= 0, f"Got negative result {result} for ({days}, {miles}, {receipts})"
    print("  PASS: no_crash edge cases")

def test_regression():
    """Test known cases within tolerance."""
    tolerance = 500  # generous tolerance for now - tighten as model improves
    failures = 0
    for days, miles, receipts, expected, desc in REGRESSION_CASES:
        result = calculate_reimbursement(days, miles, receipts)
        if expected is not None:
            error = abs(result - expected)
            status = "OK" if error < tolerance else "FAIL"
            if status == "FAIL":
                failures += 1
            print(f"  {status}: {desc} — expected ${expected:.2f}, got ${result:.2f}, err ${error:.2f}")
        else:
            print(f"  SMOKE: {desc} — got ${result:.2f}")
    return failures

def full_eval():
    """Run against all 1000 public cases. Much faster than eval.sh."""
    with open('public_cases.json') as f:
        cases = json.load(f)

    total_error = 0
    exact = 0
    close = 0
    max_error = 0
    max_error_case = None
    errors_list = []

    for i, case in enumerate(cases):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']

        result = calculate_reimbursement(days, miles, receipts)
        error = abs(result - expected)
        total_error += error

        if error < 0.01:
            exact += 1
        if error < 1.0:
            close += 1
        if error > max_error:
            max_error = error
            max_error_case = (i + 1, days, miles, receipts, expected, result)

        errors_list.append((error, i + 1, days, miles, receipts, expected, result))

    n = len(cases)
    avg_error = total_error / n
    score = avg_error * 100 + (n - exact) * 0.1

    print(f"\n  Total cases: {n}")
    print(f"  Exact matches (±$0.01): {exact} ({exact*100/n:.1f}%)")
    print(f"  Close matches (±$1.00): {close} ({close*100/n:.1f}%)")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Maximum error: ${max_error:.2f}")
    print(f"  Score: {score:.2f}")

    if max_error_case:
        c = max_error_case
        print(f"\n  Worst case: #{c[0]} — {c[1]}d {c[2]}mi ${c[3]:.2f}rcpt → exp ${c[4]:.2f} got ${c[5]:.2f}")

    # Show top 5 worst
    errors_list.sort(key=lambda x: x[0], reverse=True)
    print(f"\n  Top 5 worst:")
    for error, idx, d, mi, r, exp, got in errors_list[:5]:
        sp = " [.49/.99]" if is_special_cents(r) else ""
        print(f"    #{idx}: {d}d {mi}mi ${r:.2f}{sp} → exp ${exp:.2f} got ${got:.2f} err ${error:.2f}")

    return avg_error, exact, score

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if mode in ('quick', 'all'):
        print("=== Quick Tests ===")
        test_special_cents()
        test_no_crash()
        failures = test_regression()
        if failures:
            print(f"\n  {failures} regression failures!")

    if mode in ('full', 'all'):
        print("\n=== Full Evaluation (1000 cases) ===")
        start = time.time()
        avg_err, exact, score = full_eval()
        elapsed = time.time() - start
        print(f"\n  Completed in {elapsed:.2f}s")
