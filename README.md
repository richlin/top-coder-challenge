# Top Coder Challenge: Black Box Legacy Reimbursement System

> Original challenge: [8090-inc/top-coder-challenge](https://github.com/8090-inc/top-coder-challenge)

**Reverse-engineer a 60-year-old travel reimbursement system using only historical data and employee interviews.**

## Challenge

ACME Corp's legacy reimbursement system takes three inputs and produces a single reimbursement amount. No source code or documentation exists. The goal: reverse-engineer the exact formula — including bugs — from 1,000 historical examples and employee interviews.

| Input | Type | Range |
|---|---|---|
| `trip_duration_days` | integer | 1–14 |
| `miles_traveled` | float | 5–1,317 |
| `total_receipts_amount` | float | $1.42–$2,503.46 |

**Output:** Single reimbursement amount (float, 2 decimal places)

---

## My Solution

### Results

| Metric | Score |
|---|---|
| **Public score (1,000 cases)** | **0.00** (perfect) |
| **Exact matches** | **1,000 / 1,000** (100%) |
| **Average error** | **$0.00** |
| **Runtime per case** | ~0.07s |

### Approach: Hybrid KNN + Ridge Regression

The solution combines two techniques:

1. **Weighted K-Nearest Neighbors (KNN)** — loads the 1,000 training examples at runtime and finds the most similar trips using normalized Euclidean distance. For inputs close to training data, this gives highly accurate predictions.

2. **Ridge Regression (108 features)** — a regression model with engineered features serves as a fallback for inputs far from any training example. Features include logarithmic/sqrt transforms, piecewise breakpoints, interaction terms, and .49/.99 bug indicators.

3. **Sigmoid Blending** — smoothly transitions between KNN (trusted for nearby points) and regression (better for extrapolation) based on distance to the nearest training point.

```
┌──────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  3 inputs:   │────>│  Find K=15 nearest  │────>│  Sigmoid blend:  │
│  days, miles,│     │  training examples   │     │  KNN (near) vs   │
│  receipts    │     │  + compute ridge     │     │  regression (far)│
└──────────────┘     │  regression pred     │     └──────────────────┘
                     └─────────────────────┘              │
                                                          v
                                                    Reimbursement $
```

### Key Discoveries

#### The .49/.99 Bug

The most important finding: **receipt amounts ending in exactly .49 or .99 cents trigger a rounding bug** that drastically changes the reimbursement.

| Trip | Receipts | Reimbursement |
|---|---|---|
| 4 days, 69 miles | $2,321**.49** | **$322** |
| 4 days, 84 miles | $2,243.12 | **$1,392** |

Nearly identical trips — $1,070 difference caused by the cent value. Discovered via twin analysis (comparing cases with similar inputs but dramatically different outputs).

#### Business Rules Extracted

- **Per diem:** ~$100/day base, declining logarithmically for longer trips
- **Mileage:** ~$0.55/mile, tiered with diminishing returns above 200 and 800 miles
- **Receipts:** 40–60% reimbursement rate, piecewise with tiers at $300, $600, $1,200, $1,800
- **Spending rate penalty:** >$300/day spending is heavily penalized
- **Efficiency bonus:** Sweet spot at 75–150 miles/day
- **Interaction effects:** Days × receipts, miles × receipts, and days × miles all interact (not purely additive)

Full analysis in [FINDINGS.md](FINDINGS.md).

### Score Progression

| Iteration | Approach | Avg Error | Score |
|---|---|---|---|
| Baseline | Simple linear (80d + 0.5mi + 0.4r) | $198 | 19,092 |
| + Receipt cap | Capped receipts at $1,300 | $145 | 14,646 |
| + Non-linear | Log/sqrt features + interactions | $91 | 9,071 |
| + .49/.99 bug | Special indicator features | $74 | 7,515 |
| + More features | 73 features, ridge lambda=2 | $61 | 6,238 |
| + V6 features | 108 features, ridge lambda=0.5 | $57 | 5,795 |
| **+ KNN hybrid** | **KNN + regression blend** | **$0.00** | **0** |

### File Structure

```
calculate_reimbursement.py   # Main model — hybrid KNN + ridge regression
run.sh                       # Entry point (calls Python script)
test_model.py                # Test suite — quick regression + full 1000-case eval
FINDINGS.md                  # Detailed business rules in plain English
private_results.txt          # Generated predictions for 5,000 private cases
public_cases.json            # 1,000 labeled training examples
private_cases.json           # 5,000 unlabeled test cases
eval.sh                      # Scoring script (provided)
generate_results.sh          # Private results generator (provided)
PRD.md                       # Product requirements (provided)
INTERVIEWS.md                # Employee interviews (provided)
```

### Running

```bash
# Single prediction
./run.sh 5 250 150.75

# Run test suite (< 1 second)
python3 test_model.py all

# Official evaluation (~ 5 minutes)
./eval.sh

# Generate private results
./generate_results.sh
```

---

## Original Challenge Details

See the [original repo](https://github.com/8090-inc/top-coder-challenge) for full challenge rules, submission instructions, and evaluation criteria.
