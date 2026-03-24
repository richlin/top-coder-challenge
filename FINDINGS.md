# Reverse-Engineered Business Rules: ACME Corp Reimbursement System

## Overview

These rules were extracted from 1,000 historical input/output examples and employee interviews. The system takes three inputs (trip duration, miles traveled, total receipts) and produces a single reimbursement amount. The formula is **not purely additive** — it contains interaction effects, piecewise thresholds, and at least one confirmed bug.

The complete formula uses **23 interpretable parameters** and achieves $71 MAE (mean absolute error) on held-out data — the same generalization ceiling as any machine learning model we tested, including XGBoost and 108-feature ridge regression.

---

## 1. Per Diem (Daily Allowance)

**Formula**: `275.6 × log(days + 1) + 13.7 × days - 82.9`

The per-diem follows a logarithmic curve — each additional day adds less than the previous one.

| Trip Length | Total Per-Diem | Per-Day Rate |
|---|---|---|
| 1 day | $122 | $122/day |
| 3 days | $340 | $113/day |
| 5 days | $479 | $96/day |
| 7 days | $586 | $84/day |
| 10 days | $715 | $72/day |
| 14 days | $856 | $61/day |

The company is generous for the first few days but expects efficiency on longer trips. The per-day rate drops by half from 1-day to 14-day trips.

## 2. Mileage Reimbursement

**Four-tier piecewise rate** — not a flat per-mile rate, and notably non-monotonic:

| Mile Range | Rate | Cumulative at Top |
|---|---|---|
| 0–100 miles | **$0.93/mile** | $93 |
| 100–300 miles | **$0.41/mile** | $175 |
| 300–800 miles | **$0.61/mile** | $480 |
| 800+ miles | **$0.29/mile** | varies |

The dip at 100–300 miles followed by a recovery at 300–800 is unexpected. It suggests the system rewards "real road warriors" covering serious distance (300–800 mi) over moderate local driving (100–300 mi). Above 800 miles, diminishing returns kick in.

**Example**: 500 miles = $0.93×100 + $0.41×200 + $0.61×200 = **$297**

## 3. Receipt Processing

**Five-tier piecewise reimbursement** — the most complex component:

| Receipt Range | Reimbursement Rate | Interpretation |
|---|---|---|
| $0–$300 | **~0%** (-5%) | Per-diem already covers basic expenses |
| $300–$600 | **96%** | Nearly full reimbursement |
| $600–$1,200 | **114%** | Sweet spot — more than you submitted! |
| $1,200–$1,800 | **15%** | Sharp dropoff — diminishing returns |
| $1,800+ | **7%** | Near-cap — additional receipts barely matter |

The $600–$1,200 sweet spot at 114% means the system effectively *rewards* spending in this range — likely an artifact of overlapping reimbursement tiers in the legacy code rather than intentional policy.

Base offset: **-$66** (built-in reduction applied to all receipt calculations).

**Example**: $1,000 in receipts = -$15 + $288 + $456 - $66 = **$663**

## 4. Interaction Effects

The system is **NOT** simply `per_diem + mileage + receipts`. Three interaction terms significantly affect the output:

| Interaction | Coefficient | Effect |
|---|---|---|
| **Days × Miles / 1000** | **+8.07** | Long trips with high mileage get a bonus ("covering ground") |
| **Days × Receipts / 1000** | **-10.48** | Long trips with high spending get a penalty ("vacation penalty") |
| **Miles × Receipts / 100k** | **-12.63** | High mileage + high spending get penalized ("something doesn't add up") |

The days × receipts penalty (-10.48) is the **single most impactful interaction** — a 10-day trip with $2,000 in receipts gets a -$210 adjustment. This is the "hidden rule" that frustrates employees: spending the same total amount over more days is punished more.

The miles/day and receipts/day ratios have near-zero independent coefficients (-0.01 and +0.00), meaning the efficiency and spending rate effects are fully captured by the interaction terms above rather than being separate rules.

## 5. The .49/.99 Bug

**This is the confirmed bug referenced in the PRD.**

If total receipts end in exactly **.49 or .99 cents**, a floating-point rounding error in the legacy code produces dramatically different results.

**Bug penalty formula**: `241.9 - 1.01 × receipts + 0.86 × receipts²/10000`

| Receipt Amount | Bug Penalty | Normal Reimbursement | Bug Reimbursement |
|---|---|---|---|
| $100 | +$142 (bonus!) | ~$750 | ~$892 |
| ~$240 | $0 (breakeven) | — | — |
| $500 | -$241 | ~$900 | ~$659 |
| $1,000 | -$681 | ~$1,400 | ~$719 |
| $1,500 | -$1,057 | ~$1,700 | ~$643 |
| $2,000 | -$1,432 | ~$1,600 | ~$168 |
| $2,321 | -$1,713 | ~$1,392 | ~$0 |

The penalty is **quadratic** — it accelerates with receipt amount. Below ~$240, it's actually a slight bonus (the rounding goes in the employee's favor). Above $240, the penalty grows rapidly and can wipe out most of the reimbursement.

### Likely Cause

A floating-point rounding error in the original legacy code that pushes the receipt amount across a tier boundary during an intermediate calculation step. The quadratic shape suggests the error cascades through multiple calculation stages. This is consistent with a 60-year-old system built before IEEE 754 floating-point standards.

---

## Complete Formula

```
reimbursement = max(0,
    # Per-diem (log decay)
    275.6 × log(days + 1) + 13.7 × days - 82.9

    # Mileage (4-tier)
  + 0.93 × min(miles, 100)
  + 0.41 × min(max(0, miles - 100), 200)
  + 0.61 × min(max(0, miles - 300), 500)
  + 0.29 × max(0, miles - 800)

    # Receipts (5-tier)
  - 0.05 × min(receipts, 300)
  + 0.96 × min(max(0, receipts - 300), 300)
  + 1.14 × min(max(0, receipts - 600), 600)
  + 0.15 × min(max(0, receipts - 1200), 600)
  + 0.07 × max(0, receipts - 1800)
  - 66.0

    # Interactions
  + 8.07 × days × miles / 1000
  - 10.48 × days × receipts / 1000
  - 12.63 × miles × receipts / 100000
  + 12.3

    # Bug (only if receipts end in .49 or .99)
  + 241.9 - 1.01 × receipts + 0.86 × receipts² / 10000
)
```

**23 parameters. $71 mean absolute error. No machine learning required.**

---

## Practical Advice for Employees

1. **Submit receipts in the $600–$1,200 range** — this is the 114% sweet spot
2. **Keep daily spending reasonable** — the days×receipts penalty hits hard on long trips
3. **Never submit receipts ending in .49 or .99** — round up one cent to avoid the bug
4. **High-mileage long trips are rewarded** — the days×miles bonus offsets per-diem decay
5. **Don't bother submitting receipts under $300** — you get essentially 0% on the first $300
6. **4–6 day trips** get the best per-day treatment ($96–$80/day)

---

## Model Comparison

We tested multiple approaches to understand which generalizes best to unseen cases:

| Approach | Training MAE | CV MAE (unseen) | Parameters |
|---|---|---|---|
| Simple rules (this doc) | $71 | $66 | 23 |
| Ridge regression (108 features) | $57 | $63 | 108 |
| KNN + Ridge hybrid | $0 (memorized) | $62 | 108 + training data |
| XGBoost (depth=5) | $14 | $65 | ~thousands |

All approaches converge to **~$62–66 MAE on unseen data**. The simple 23-parameter model is within $4 of the best approach. The KNN hybrid achieves $0 training error by memorizing the 1,000 known cases, but on new inputs it performs comparably to the simple rules.

---

## Methodology

- Analyzed 1,000 historical input/output cases from `public_cases.json`
- Cross-referenced with employee interview signals (taken directionally, not literally)
- Used iterative decomposition to isolate per-diem, mileage, and receipt components
- Jointly optimized all 23 parameters via Nelder-Mead minimization of mean absolute error
- Validated via 5-fold cross-validation to confirm generalization performance
- The .49/.99 bug was discovered via twin analysis: comparing cases with near-identical inputs but dramatically different outputs
- XGBoost feature importance analysis confirmed: special_cents (31%), receipts (38%), interactions (19%), miles alone (<2%)
- Tree split point extraction validated the tier boundaries (100/300/800 for miles, 300/600/1200/1800 for receipts)
- Implementation: `simple_rules_model.py` (60 lines, zero dependencies beyond Python stdlib)
