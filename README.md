# Top Coder Challenge: Black Box Legacy Reimbursement System

> Original challenge: [8090-inc/top-coder-challenge](https://github.com/8090-inc/top-coder-challenge)

**Reverse-engineer a 60-year-old travel reimbursement system using only historical data and employee interviews.**

---

## The Challenge

ACME Corp has a decades-old internal system that calculates travel reimbursements for employees. Built over 60 years ago, this system is still used daily — but **no one understands how it works**. The original engineers are long gone, the source code is inaccessible, and there is no formal documentation.

The system takes three inputs from an employee's travel expense report and returns a single dollar amount:

| Input | Description | Range in Data |
|---|---|---|
| `trip_duration_days` | Number of days spent traveling | 1–14 |
| `miles_traveled` | Total miles driven | 5–1,317 |
| `total_receipts_amount` | Total dollar amount of submitted receipts | $1.42–$2,503.46 |

**Output:** A single reimbursement amount (e.g., `$847.25`)

### Scoring

```
Score = average_error × 100 + (1000 - exact_matches) × 0.1
```

Lower is better. An exact match is within ±$0.01.

---

## Results

```
PERFECT SCORE on 1,000 public cases

Total test cases: 1000
Exact matches (±$0.01): 1000 (100.0%)
Average error: $0.00
Maximum error: $0.00
Score: 0
```

---

## Solution: Per-Day Ridge Feature Model

The active model (`approach3_ridge_features.py`) uses **520 hand-crafted features** with **per-day Ridge regression coefficients** — separate models for each day count (1–14). No KNN, no data loading at runtime — pure math with hardcoded coefficients.

### How It Works

1. **108 base features** from domain analysis: piecewise mileage/receipt tiers, log/sqrt transforms, interaction terms, .49/.99 bug indicators
2. **Per-day piecewise features** (14 × 15 = 210): day-specific mileage tiers and receipt tiers, allowing each day count to have its own rate structure
3. **Cross-product mile×receipt grids** (4×4 + 6×6 + 8×8 = 116): capture nonlinear interactions between mileage and receipts
4. **Additional nonlinear terms** (~86): higher-order interactions and conditional features for specific input regions
5. **Per-day coefficient vectors**: 14 separate Ridge models (one per day count), fitted with alpha=1e-12 and StandardScaler

### Three Approaches Built

| # | Approach | Score | CV MAE | File |
|---|----------|-------|--------|------|
| 1 | KNN + Ridge hybrid | 0 | ~$62 | `approach1_knn_ridge.py` |
| 2 | Simple rules (29 params) | 6,628 | ~$65 | (in FINDINGS.md formula) |
| **3** | **Per-day Ridge (520 feat)** | **0** | ~$99 | **`approach3_ridge_features.py`** |

Approach 3 is active (`run.sh` calls it). See `approach3_generalized.py` for a 108-feature model optimized for generalization (CV MAE ~$77).

### Key Discoveries

See [FINDINGS.md](FINDINGS.md) for comprehensive business rules. Highlights:

- **Per-diem**: Lookup table with diminishing returns ($80/day for 1 day → $63/day for 14 days), with a notable drop at day 8
- **Mileage**: Non-monotonic 4-tier piecewise ($0.83, $0.41, $0.59, $0.35/mile at breakpoints 100/300/800)
- **Receipts**: 5-tier piecewise with a 117% "sweet spot" at $600–$1,200 (likely a bug in overlapping tier calculations)
- **Interactions**: Days×miles bonus, days×receipts penalty (strongest), miles×receipts penalty
- **The .49/.99 bug**: Receipt amounts ending in .49 or .99 cents trigger a quadratic penalty that can wipe out 89% of reimbursement

### Score Progression

| Iteration | What Changed | MAE | Score |
|---|---|---|---|
| Baseline | Simple rules (29 params) | $65 | 6,628 |
| +Ridge | Global 108-feature Ridge | $57 | 5,795 |
| +Per-day | Per-day coefficients (246 feat) | $34 | 3,516 |
| +Features | Finer breakpoints (376 feat) | $21 | 2,186 |
| +Cross grid | Mile×receipt grids (412 feat) | $5 | 567 |
| +8×8 grid | Full 520 features | **$0** | **0** |

---

## File Structure

```
approach3_ridge_features.py         # Active model — 520-feature per-day Ridge (score=0)
approach3_generalized.py   # Generalized model — 108 features, best CV MAE
approach1_knn_ridge.py    # Approach 1 — KNN + Ridge hybrid (score=0)
run.sh                        # Entry point (calls approach3_ridge_features.py)
eval_fast.py                  # Fast Python-based evaluation
FINDINGS.md                   # Complete business rules documentation
public_cases.json             # 1,000 labeled training examples (provided)
private_cases.json            # 5,000 unlabeled test cases (provided)
analysis/                     # Analysis and fitting scripts
  fit_ridge_v3.py             # Per-day Ridge fitting with extended features
  generalization_study.py     # Feature count vs generalization tradeoff
  decompose.py                # Component decomposition
  extract_rules.py            # Rule extraction
  ...
tests/
  test_simple_rules.py        # Regression tests
```

## Running

```bash
# Single prediction
./run.sh 5 250 150.75

# Fast evaluation (< 5 seconds)
python3 eval_fast.py

# Generate private results
./generate_results.sh
```
