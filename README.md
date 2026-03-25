# Top Coder Challenge: Black Box Legacy Reimbursement System

**Reverse-engineer a 60-year-old travel reimbursement system using only historical data and employee interviews.**
> Original link: [8090-inc/top-coder-challenge](https://github.com/8090-inc/top-coder-challenge)

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

### The Problem

Stakeholders have observed frequent anomalies: unpredictable reimbursement amounts, inconsistent treatment of receipts, and odd behaviors tied to specific trip lengths or distances. Different departments hold conflicting folklore about how the system works. 8090 built a replacement system, but ACME is confused by the differences in results.

### The Mission

Figure out the original business logic — **including any bugs** — so we can explain why the new system is different and better. Create a perfect replica of the legacy system by reverse-engineering its behavior.

### What's Given

- **1,000 historical input/output examples** (`public_cases.json`) — labeled training data with known expected outputs
- **5,000 private test cases** (`private_cases.json`) — inputs only, no expected outputs (used for final scoring)
- **Employee interviews** (`INTERVIEWS.md`) — anecdotal, inconsistent, and occasionally contradictory memories from employees who use the system
- **A PRD** (`PRD.md`) — product requirements describing the business context

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

## Key Discoveries

Through analytical decomposition, twin analysis, and employee interviews, we reverse-engineered five core components of the legacy system. For comprehensive details, see **[FINDINGS.md](FINDINGS.md)**.

### Per-Diem: The "5-Day Sweet Spot" and Day-8 Penalty

The system uses a 14-value lookup table with diminishing returns. Per-day rate peaks at 5 days ($94/day), then declines — with a sharp, unexplained **drop at day 8** from $89/day to $69/day. Employees who extend a 7-day trip to 8 days actually lose $69 in total reimbursement.

### Mileage: Non-Monotonic 4-Tier Piecewise

| Mile Range | Rate per Mile |
|---|---|
| 0–100 | $0.83 |
| 100–300 | $0.41 (dip) |
| 300–800 | $0.59 (recovery) |
| 800+ | $0.35 |

The counter-intuitive dip at 100–300 miles followed by recovery at 300–800 discourages short local trips while rewarding longer business travel.

### Receipts: The 117% "Sweet Spot" Bug

A 5-tier receipt system where the $600–$1,200 range reimburses at **117%** — employees get back more than they spent. Almost certainly a bug from overlapping tier calculations in the legacy code. Below $300, receipts effectively don't help. Above $1,200, diminishing returns.

### Interaction Penalties

The system is not purely additive. Three hidden cross-terms modify reimbursements:

| Interaction | Effect | Name |
|---|---|---|
| Days × Miles / 1000 | +$7.40 | "Road warrior reward" |
| Days × Receipts / 1000 | **-$11.60** | "Vacation penalty" |
| Miles × Receipts / 100k | -$13.40 | "Something doesn't add up" |

The **days × receipts penalty** is the single most impactful hidden rule — identical daily spending produces dramatically different outcomes depending on trip length.

### The .49/.99 Floating-Point Bug

When receipt totals end in exactly .49 or .99 cents, a floating-point rounding error from the pre-IEEE 754 era triggers a **quadratic penalty**. Affects 3% of cases. Below ~$210, it's actually a small bonus. Above $500, it can wipe out up to **89% of reimbursement** — hundreds of dollars lost because of a cents value.

### Employee Interview Validation

| Claim | Verified? |
|---|---|
| "5-6 day trips are the sweet spot" (Jennifer, Kevin) | **Yes** |
| "The system rewards hustle / high miles per day" (Marcus) | **Yes** |
| "High spending on long trips gets penalized" (Kevin) | **Yes** |
| "Receipts ending in .49/.99 give extra money" (Lisa) | **Partially** — only for small amounts |
| "Small receipts get penalized vs. no receipts" (Dave) | **Yes** |
| "Tuesday submissions beat Friday" (Kevin) | **No** — no temporal data in the system |
| "Lunar cycles affect reimbursement" (Kevin) | **No** |

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

## Recommendation: Replace the System

Based on our findings, we recommend ACME replace the legacy system entirely. See **[RECOMMENDATION.md](RECOMMENDATION.md)** for the full consulting recommendation. Key points:

### Why Replace

- **A bug is determining compensation.** The .49/.99 floating-point error can cost employees hundreds of dollars per trip — with no audit trail or justification.
- **The 117% receipt sweet spot is not policy** — it's an artifact of overlapping tier calculations that no one authorized.
- **Employees have lost trust.** One employee built a 3-year spreadsheet operation testing lunar cycles. Another deliberately manipulates receipt totals. A third has given up entirely.
- **The system is unchangeable.** No source code access means ACME cannot fix bugs, adjust rates, or respond to policy changes.

### Three Proposed Replacement Models

| Model | Philosophy | Complexity | Transparency |
|---|---|---|---|
| **A: "Clean Legacy"** | Same rates, bugs removed | 29 params | Low — interaction terms still opaque |
| **B: "Simple & Transparent"** | $85/day + $0.55/mile + 75% receipts | 3 params | Perfect — anyone can compute it |
| **C: "Policy-Driven" (Rec.)** | Explicit tiers + efficiency bonus | ~12 params | High — all rules in plain English |

**We recommend Model C** as the starting point for stakeholder design workshops — transparent enough that employees can predict their reimbursements, nuanced enough to reward efficient travel.

### Implementation Roadmap

| Phase | Timeline | Deliverable |
|---|---|---|
| Validate replica | Weeks 1–2 | Shadow system confirmation on live data |
| Design new rules | Weeks 3–6 | Stakeholder-approved business rules |
| Build & test | Weeks 7–12 | New engine with full test coverage |
| Cutover | Weeks 13–14 | Production deployment + employee communication |

---

## File Structure

```
approach3_ridge_features.py    # Active model — 520-feature per-day Ridge (score=0)
approach3_generalized.py       # Generalized model — 108 features, best CV MAE
approach1_knn_ridge.py         # Approach 1 — KNN + Ridge hybrid (score=0)
run.sh                         # Entry point (calls approach3_ridge_features.py)
eval_fast.py                   # Fast Python-based evaluation
FINDINGS.md                    # Complete reverse-engineered business rules
RECOMMENDATION.md              # Consulting recommendation for system replacement
public_cases.json              # 1,000 labeled training examples (provided)
private_cases.json             # 5,000 unlabeled test cases (provided)
analysis/                      # Analysis and fitting scripts
  fit_ridge_v3.py              # Per-day Ridge fitting with extended features
  generalization_study.py      # Feature count vs generalization tradeoff
  decompose.py                 # Component decomposition
  extract_rules.py             # Rule extraction
  ...
tests/
  test_simple_rules.py         # Regression tests
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
