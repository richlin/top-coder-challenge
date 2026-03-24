import pytest
import sys
import os

# Add parent dir to path so we can import simple_rules_model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from simple_rules_model import calculate_reimbursement, is_special_cents


class TestSpecialCents:
    def test_49_cents(self):
        assert is_special_cents(100.49) == True

    def test_99_cents(self):
        assert is_special_cents(100.99) == True

    def test_50_cents(self):
        assert is_special_cents(100.50) == False

    def test_00_cents(self):
        assert is_special_cents(100.00) == False

    def test_48_cents(self):
        assert is_special_cents(100.48) == False

    def test_01_cents(self):
        assert is_special_cents(100.01) == False


class TestNonNegative:
    """All outputs should be non-negative"""
    def test_typical_inputs(self):
        for d in [1, 3, 7, 14]:
            for m in [10, 200, 800, 1300]:
                for r in [5, 500, 1500, 2500]:
                    result = calculate_reimbursement(d, m, r)
                    assert result >= 0, f"Negative output for d={d}, m={m}, r={r}: {result}"

    def test_edge_inputs(self):
        assert calculate_reimbursement(1, 0, 0) >= 0
        assert calculate_reimbursement(1, 5, 1.42) >= 0
        assert calculate_reimbursement(14, 1317, 2503.46) >= 0


class TestBugBehavior:
    """The .49/.99 bug should produce lower outputs for high receipts"""
    def test_high_receipt_bug_is_penalized(self):
        # Bug case: high receipts with .49 should be much lower than similar non-bug
        bug_result = calculate_reimbursement(4, 69, 2321.49)
        normal_result = calculate_reimbursement(4, 69, 2321.50)
        assert bug_result < normal_result, "Bug case should produce lower output"

    def test_low_receipt_bug_may_bonus(self):
        # Low receipt bug cases may get a slight bonus
        bug_result = calculate_reimbursement(3, 117, 21.99)
        # Just verify it's a reasonable number (not negative or extreme)
        assert 0 < bug_result < 2000


class TestKnownCases:
    """Test against a few known public cases (within reasonable tolerance)"""
    def test_case_short_trip_low_receipts(self):
        # 3 days, 93 miles, $1.42 receipts -> expected ~$364.51
        result = calculate_reimbursement(3, 93, 1.42)
        assert abs(result - 364.51) < 100, f"Expected ~364.51, got {result}"

    def test_case_medium_trip(self):
        # 5 days, 500 miles, $1000 receipts -> should be in reasonable range
        result = calculate_reimbursement(5, 500, 1000)
        assert 800 < result < 2000, f"Expected 800-2000, got {result}"

    def test_case_long_trip(self):
        # 14 days, 1000 miles, $500 receipts -> should be reasonable
        result = calculate_reimbursement(14, 1000, 500)
        assert 800 < result < 2500, f"Expected 800-2500, got {result}"


class TestNoTrainingData:
    """Model should not load training data at runtime"""
    def test_no_json_import_in_calculate(self):
        import inspect
        source = inspect.getsource(calculate_reimbursement)
        assert 'json' not in source, "calculate_reimbursement should not reference json"
        assert 'open(' not in source, "calculate_reimbursement should not open files"

    def test_no_public_cases_reference(self):
        import inspect
        source = inspect.getsource(calculate_reimbursement)
        assert 'public_cases' not in source


class TestReturnType:
    """Output should be a properly rounded float"""
    def test_returns_float(self):
        result = calculate_reimbursement(5, 300, 800)
        assert isinstance(result, float)

    def test_two_decimal_places(self):
        result = calculate_reimbursement(5, 300, 800)
        assert result == round(result, 2)
