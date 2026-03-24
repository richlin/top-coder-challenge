# Exact Rules Extraction: Pure Rule-Based Model from $71 MAE toward $40

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the pure rule-based reimbursement model (no KNN, no training data at runtime) from $71 MAE / score 7223 to under $40 MAE / score 4000. Aspirational: discover the exact legacy formula for score 0.

**Architecture:** Sequential investigation: start with residual diagnostics to identify what's missing, then apply targeted fixes. The legacy system is a 60-year-old deterministic calculator with tiered rates, intermediate rounding, and a known floating-point bug. It does NOT use ML.

**Tech Stack:** Python 3 stdlib, scipy, numpy. PySR (optional, requires Julia) for symbolic regression.

**Current State:**
- `simple_rules_model.py` — 23 parameters, $71 MAE, score 7223
- `calculate_reimbursement.py` — KNN+ridge hybrid, score 0 (memorizes training data)
- Existing analysis in `analysis/` — per-day fits, joint optimization, tree analysis already done

---

## File Structure

```
simple_rules_model.py                    # TARGET: the model to improve
run.sh                                   # Entry point (calls simple_rules_model.py)
eval_fast.py                             # Fast scorer (FIX: update import)

analysis/
  fit_per_day.py                         # Existing: per-day-count models
  fit_rules_jointly.py                   # Existing: Nelder-Mead joint optimization
  decompose.py                           # Existing: component isolation
  track0_residual_diagnostics.py         # Create: systematic residual analysis
  track1_rounding_analysis.py            # Create: intermediate rounding detection
  track2_expanded_rules.py               # Create: 108-feature rules model (no KNN)
  track3_symbolic_regression.py          # Create: PySR formula search (optional)
  synthesize_results.py                  # Create: combine findings into final model

tests/
  test_simple_rules.py                   # Create: regression tests
```

---

## Task 0: Fix eval_fast.py and establish baseline

**Files:**
- Modify: `eval_fast.py` (line 10: change import to `simple_rules_model`)

- [ ] **Step 1: Fix the import**

Change `eval_fast.py` line 10 from:
```python
from calculate_reimbursement_rules import calculate_reimbursement
```
to:
```python
from simple_rules_model import calculate_reimbursement
```

- [ ] **Step 2: Run baseline**

Run: `python3 eval_fast.py`
Expected: score ~7223, avg_error ~$71, exact matches ~1

- [ ] **Step 3: Commit**

```bash
git add eval_fast.py
git commit -m "Fix eval_fast.py to import simple_rules_model"
```

---

## Track 0: Residual Diagnostics (run FIRST)

**Hypothesis:** The $71 MAE has systematic patterns — specific trip types, input ranges, or derived features where the model consistently over/under-predicts. Finding these patterns tells us exactly which new features or rules to add.

### Task 0.1: Systematic residual analysis

**Files:**
- Create: `analysis/track0_residual_diagnostics.py`
- Reference: `simple_rules_model.py`, `public_cases.json`

- [ ] **Step 1: Write the residual analysis script**

For each of the 1,000 cases, compute predicted vs actual and the signed residual (predicted - actual). Then analyze:

1. **Residuals vs each raw input**: Plot/print residual means binned by days (1-14), miles (8 bins), receipts (8 bins)
2. **Residuals vs derived features**: miles/day, receipts/day, days×miles, days×receipts
3. **Residuals vs trip type**: 2D grid of (day bucket × mile bucket × receipt bucket) — which combinations have the worst errors?
4. **Residual distribution**: Are residuals normally distributed or do they cluster? Clustering suggests missing categorical rules.
5. **Worst 20 cases**: Print full details. Look for patterns — are they all high-mileage? All long trips? All bug-adjacent?
6. **Per-day-count MAE**: Which day counts are the model worst at? (Already know from `fit_per_day.py` that days 7-8 and 13-14 are worst)

- [ ] **Step 2: Run**

Run: `python3 analysis/track0_residual_diagnostics.py`

- [ ] **Step 3: Analyze output and document findings**

Key questions to answer:
- Is the error dominated by a few bad trip types or spread evenly?
- Are there sign patterns (always over-predicting for long trips, under-predicting for high miles)?
- Do the worst cases share a common feature (e.g., all have miles > 800 AND days > 7)?

These findings determine which subsequent tracks to prioritize.

- [ ] **Step 4: Commit**

```bash
git add analysis/track0_residual_diagnostics.py
git commit -m "Track 0: residual diagnostics for simple rules model"
```

---

## Track 1: Rounding Analysis

**Hypothesis:** The .49/.99 bug proves the legacy system rounds intermediate results. Discovering the rounding operations constrains the formula and could explain micro-errors in our predictions.

**Dependency:** Run after Track 0 (diagnostics may reveal rounding patterns).

### Task 1.1: Analyze output precision and intermediate rounding

**Files:**
- Create: `analysis/track1_rounding_analysis.py`
- Reference: `public_cases.json`

- [ ] **Step 1: Write the rounding analysis script**

Analyze the 1,000 outputs for precision patterns:

1. **Cent distribution**: Count frequency of each cent value (00-99). Uniform = no output rounding. Clustered = output is rounded.
2. **Intermediate rounding test**: For non-bug cases, compute `(output - per_diem_estimate)`. Is the remainder consistent with rounded mileage + rounded receipts?
3. **Bug mechanism test**: For each of the 30 bug cases:
   - Compute what output would be if `receipts = round(receipts)` (nearest integer)
   - Compute what output would be if `receipts = floor(receipts * 100) / 100` vs `round(receipts * 100) / 100`
   - Check if `int(receipts)` vs `float(receipts)` crossing a tier boundary explains the penalty
4. **Multiplicative factor test**: Check if `output / (days * some_rate)` yields a recognizable pattern
5. **Common factor test**: Are outputs always divisible by some value (e.g., $0.01, $0.05)?

- [ ] **Step 2: Run and report**

Run: `python3 analysis/track1_rounding_analysis.py`

- [ ] **Step 3: Commit**

```bash
git add analysis/track1_rounding_analysis.py
git commit -m "Track 1: rounding pattern analysis"
```

---

## Track 2: Expanded Rules Model (no KNN)

**Hypothesis:** The gap from $71 to $57 MAE is because our 23-parameter model is too coarse. The original 108-feature ridge regression achieves $57 MAE. Porting those 108 features into `simple_rules_model.py` (with hardcoded coefficients, no training data at runtime) should close this gap. Then combine with per-day-count adjustments from `analysis/fit_per_day.py` to push further.

**Dependency:** Run after Track 0 (diagnostics tell us which features matter most).

### Task 2.1: Port 108 features and fit standalone coefficients

**Files:**
- Create: `analysis/track2_expanded_rules.py`
- Reference: `calculate_reimbursement.py` (lines 59-133 for feature list)
- Reference: `analysis/fit_per_day.py` (existing per-day analysis)

- [ ] **Step 1: Write the expanded model script**

1. Copy the 108-feature builder from `calculate_reimbursement.py:_regression_predict()` lines 59-133
2. Build feature matrix X (1000 × 108) for all public cases
3. Fit via ridge regression with CV alpha selection (test alpha = 0, 0.01, 0.1, 1, 10)
4. Also test per-day-count models: for each day 1-14, fit separate ridge on the 108 features using only that day's cases. Use a larger alpha (10-100) since n~70 cases per day.
5. Compare: global 108-feature model vs 14 per-day models
6. Hardcode the winning coefficients

- [ ] **Step 2: Evaluate**

Run: `python3 analysis/track2_expanded_rules.py`

Report: MAE for global model, MAE for per-day models, improvement over 23-param baseline.

Expected: Global should match ~$57. Per-day models may reach ~$45-50.

- [ ] **Step 3: Update simple_rules_model.py**

Replace the 23-parameter formula with the best approach (either global 108-feature or 14 per-day models). Hardcode all coefficients — no training data loaded at runtime.

- [ ] **Step 4: Verify**

Run: `python3 eval_fast.py`
Run: `./run.sh 3 93 1.42` — should output close to $364.51

- [ ] **Step 5: Commit**

```bash
git add analysis/track2_expanded_rules.py simple_rules_model.py
git commit -m "Track 2: expanded 108-feature rules model, hardcoded coefficients"
```

---

## Track 3: Symbolic Regression (optional, high-reward/high-risk)

**Hypothesis:** The legacy formula uses specific mathematical operations that symbolic regression can literally discover. This is the only realistic path to score 0 without KNN.

**Dependency:** Independent. Can run in parallel with Track 2.

**Prerequisite:** PySR requires Julia. Check with `julia --version`. If not installed, use `gplearn` as a lighter alternative.

### Task 3.1: Run formula search

**Files:**
- Create: `analysis/track3_symbolic_regression.py`

- [ ] **Step 1: Check Julia and install PySR**

Run: `julia --version` — if fails, install Julia first or fall back to `gplearn`.
Run: `pip3 install pysr` — if PySR install fails (common), use `pip3 install gplearn` instead.

- [ ] **Step 2: Write the symbolic regression script**

Split into two searches:

**Search A: Non-bug cases (n=970)**
- Variables: days, miles, receipts
- Operators: +, -, ×, ÷, log, sqrt, pow, min, max
- Do NOT include floor/round (they make the search degenerate)
- Target: the reimbursement output
- Complexity: 20-40 nodes
- PySR: `niterations=100, populations=50, maxsize=40`

**Search B: Bug penalty (n=30)**
- Variables: receipts (the dominant factor)
- Target: (actual output) - (non-bug model prediction for same inputs)
- Operators: +, -, ×, ÷, pow, floor, round (include floor/round here since the bug IS about rounding)
- Complexity: 10-20 nodes

- [ ] **Step 3: Run (allow 1-4 hours)**

Run: `python3 analysis/track3_symbolic_regression.py`

Examine the Pareto front. Look for:
- Formulas with recognizable coefficients (0.585, 100, 0.75)
- Specific tier boundaries appearing naturally
- Any floor/round operations in the bug formula

- [ ] **Step 4: Commit**

```bash
git add analysis/track3_symbolic_regression.py
git commit -m "Track 3: symbolic regression formula search"
```

---

## Task 4: Synthesize and build final model

**Files:**
- Create: `analysis/synthesize_results.py`
- Modify: `simple_rules_model.py`

- [ ] **Step 1: Compare track results**

Collect MAE from each track:
- Track 0: identified which features/trip types cause most error
- Track 1: any rounding operations discovered?
- Track 2: expanded model MAE (expected ~$50-57)
- Track 3: any discovered formula? (bonus)

- [ ] **Step 2: Build final model**

Combine the best elements into `simple_rules_model.py`:
- Best coefficient set (from Track 2)
- Any rounding operations (from Track 1)
- Any discovered formula elements (from Track 3)

Ensure the model:
- Does NOT load `public_cases.json` at runtime
- Does NOT use KNN or any training data lookup
- Uses only Python stdlib (math module)

- [ ] **Step 3: Final evaluation**

Run: `./eval.sh`
Report: score, MAE, exact matches, max error, worst 5 cases.

- [ ] **Step 4: Commit**

```bash
git add simple_rules_model.py analysis/synthesize_results.py
git commit -m "Final synthesized rules model"
```

---

## Task 5: Regression test suite

**Files:**
- Create: `tests/test_simple_rules.py`

- [ ] **Step 1: Write tests**

```python
import pytest
from simple_rules_model import calculate_reimbursement, is_special_cents

def test_known_case_1():
    """Case #1 from public_cases.json"""
    assert calculate_reimbursement(3, 93, 1.42) == pytest.approx(364.51, abs=50)

def test_bug_case():
    """The .49/.99 bug should produce low output for high receipts"""
    result = calculate_reimbursement(4, 69, 2321.49)
    assert result < 500  # normal case would be ~$1400

def test_non_negative():
    """All outputs should be non-negative"""
    for d in [1, 3, 7, 14]:
        for m in [10, 200, 800, 1300]:
            for r in [5, 500, 1500, 2500]:
                assert calculate_reimbursement(d, m, r) >= 0

def test_no_training_data():
    """Model should not import json or load files"""
    import inspect
    source = inspect.getsource(calculate_reimbursement)
    assert 'json' not in source
    assert 'open(' not in source

def test_special_cents():
    assert is_special_cents(100.49) == True
    assert is_special_cents(100.99) == True
    assert is_special_cents(100.50) == False
    assert is_special_cents(100.00) == False
```

- [ ] **Step 2: Run**

Run: `python3 -m pytest tests/test_simple_rules.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_simple_rules.py
git commit -m "Add regression tests for simple rules model"
```

---

## Execution Order

```
Task 0 (fix eval) ──> Track 0 (diagnostics) ──> Track 1 (rounding) ──┐
                                                                       ├──> Task 4 (synthesize)
                                                Track 2 (expanded)  ──┤
                                                                       │
                                                Track 3 (symbolic)  ──┘

                                                Task 5 (tests) ── anytime
```

Track 0 runs first — its output determines which subsequent tracks to prioritize. Tracks 1, 2, 3 can run in parallel after Track 0.

---

## Success Criteria

| Milestone | Score | MAE | Approach |
|-----------|-------|-----|----------|
| Current baseline | 7224 | $71 | 23-param simple rules |
| Expanded features | <5800 | <$57 | 108-feature ridge, no KNN |
| Per-day + expanded | <5000 | <$50 | 14 per-day models |
| With rounding fixes | <4000 | <$40 | If rounding ops discovered |
| Exact formula | 0 | $0 | If symbolic regression succeeds |

**Primary target:** Score < 5000 (MAE < $50). This is achievable with Track 2.
**Stretch target:** Score < 4000 (MAE < $40). Requires rounding or formula discoveries.
**Aspirational:** Score 0. Only if symbolic regression finds the exact formula.
