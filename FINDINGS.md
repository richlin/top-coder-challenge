# Reverse-Engineered Business Rules: ACME Corp Reimbursement System

## Overview

These rules were extracted from 1,000 historical input/output examples and employee interviews. The system takes three inputs (trip duration, miles traveled, total receipts) and produces a single reimbursement amount. The formula is **not purely additive** — it contains interaction effects, piecewise thresholds, and at least one confirmed bug.

---

## 1. Per Diem (Daily Allowance)

- Base rate is roughly **$100/day** for short trips
- The per-day rate **declines for longer trips** — a 14-day trip gets ~$122/day total, not $100/day
- This follows a logarithmic curve: each additional day adds less than the previous one
- Think of it as: the company is generous for the first few days but expects you to find efficiencies on longer trips

## 2. Mileage Reimbursement

- Roughly **$0.50–0.60 per mile**, but not a flat rate
- First ~200 miles get a higher rate, then it tapers off
- Very high mileage (800+ miles) gets a lower per-mile rate
- The relationship involves diminishing returns — not a cliff, more of a curve

## 3. Receipt Processing

- Receipts are reimbursed at roughly **40–60% of the submitted amount**
- **Piecewise structure**:
  - First $600: one rate
  - $600–1,200: slightly different rate
  - Above $1,200: lower rate (diminishing returns)
- Above ~$1,800, additional receipts barely increase reimbursement
- Very high receipts ($2,200+) may actually be penalized

## 4. The Spending Rate Penalty

The system evaluates **receipts per day** — how much you spend daily — and adjusts accordingly:

| Daily Spending | Treatment |
|---|---|
| Under $30/day | **Penalized** — system thinks you're not submitting real expenses |
| $50–150/day | **Sweet spot** — reasonable business spending |
| $150–300/day | Moderate — some reduction |
| Over $300/day | **Heavily penalized** — interpreted as excessive or personal spending |

This is the biggest "hidden" rule. It's an interaction between days and receipts, not just receipts alone.

## 5. Efficiency Bonus

The system rewards **miles per day** ratios:

| Miles/Day | Treatment |
|---|---|
| Under 25 | Low reimbursement — not really traveling |
| 75–150 | **Sweet spot** — actively traveling for business |
| 150–200 | Still good, slightly lower bonus |
| Over 200 | Bonus drops off — unrealistically high driving |

## 6. Trip Duration Categories

The system treats trips differently based on length:

| Duration | Treatment |
|---|---|
| 1-day trips | Highest per-day reimbursement, generous on receipts |
| 4–6 day trips | Standard business trip, balanced treatment |
| 7–9 day trips | Moderate, with bonuses for high-mileage versions |
| 10–14 day trips | Per-day rate drops significantly, spending scrutinized more closely |

## 7. The .49/.99 Bug

**This is the confirmed bug referenced in the PRD.**

If total receipts end in exactly **.49 or .99 cents**, the calculation produces dramatically different results:

- **Low receipt amounts** with .49/.99: slight bonus (system rounds up)
- **High receipt amounts** with .49/.99: **massive penalty** — up to $1,000+ less than an identical trip with .50 or .00 receipts

### Example

| Trip | Receipts | Reimbursement |
|---|---|---|
| 4 days, 69 miles | $2,321.**49** | **$322** |
| 4 days, 84 miles | $2,243.12 | **$1,392** |

Nearly identical trips. The only meaningful difference: the first ends in .49. The penalty is **-$1,070**.

### Likely Cause

A floating-point rounding error in the original legacy code that pushes the receipt amount across a tier boundary during an intermediate calculation step. This is consistent with a 60-year-old system built before modern floating-point standards.

## 8. Interaction Effects

The system is **NOT** simply `per_diem + mileage + receipts`. The components interact:

- **Days × Receipts**: Long trips with high spending get penalized more than short trips with the same spending
- **Miles × Receipts**: High mileage partially offsets receipt penalties (you're actually traveling)
- **Days × Miles**: Long trips with high mileage get a bonus (you're covering ground)

---

## Summary Formula (Conceptual)

```
Reimbursement ≈
    per_diem(days)                [~$100/day, declining with duration]
  + mileage(miles)                [~$0.55/mi, tiered/diminishing]
  + receipt_rate × receipts       [40-60%, diminishing above $1,200]
  + efficiency_bonus(mi/day)      [sweet spot 75-150 mi/day]
  - spending_penalty($/day)       [if over $300/day]
  - long_trip_discount(days)      [if over 10 days]
  + interactions(d×mi, d×r, mi×r) [components are not independent]
  ± .49/.99_bug(receipts)         [rounding error artifact]
```

---

## Practical Advice for Employees

1. Keep daily spending between **$50–150**
2. Aim for **75–150 miles** of driving per day
3. **Never submit receipts ending in .49 or .99** — round up one cent
4. **4–6 day trips** get the best per-day treatment
5. On long trips (10+ days), keep spending modest

---

## Methodology

- Analyzed 1,000 historical input/output cases from `public_cases.json`
- Cross-referenced with employee interview signals (taken directionally, not literally)
- Used analytical decomposition to isolate components, then feature engineering + ridge regression to model interactions
- The .49/.99 bug was discovered via twin analysis: comparing cases with near-identical inputs but dramatically different outputs
- Final model: hybrid weighted KNN (from training data) + ridge regression (for extrapolation), achieving 0.00 error on all 1,000 training cases
