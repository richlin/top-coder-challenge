import numpy as np
from typing import List, Tuple, Optional
from sklearn.linear_model import LinearRegression
import json
import pandas as pd
from statsmodels.formula.api import ols

from rulefit import RuleFit


def load_cases_from_json(filename='public_cases.json'):
    """Load test cases from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    cases = []
    for case in data:
        input_data = case['input']
        expected_output = case['expected_output']
        cases.append((
            input_data['trip_duration_days'],
            input_data['miles_traveled'],
            input_data['total_receipts_amount'],
            expected_output
        ))
    return cases

# Load data
cases = load_cases_from_json()
df = pd.DataFrame(cases, columns=['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output'])

# Prepare feature data
X = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
y = df['expected_output']

# Train RuleFit with faster parameters
print(f"Training RuleFit on {len(X)} samples with {X.shape[1]} features...")
rf = RuleFit(
    tree_size=10,        # Smaller trees for faster training
    sample_fract=0.8,   # Use only 50% of samples for each tree
    max_rules=100,       # Limit number of rules to speed up training
    memory_par=0.01,    # Lower memory parameter
    rfmode='regress',   # Regression mode
    random_state=42     # For reproducibility
)
rf.fit(X.values, y.values, feature_names=['trip_duration_days', 'miles_traveled', 'total_receipts_amount'])
print("RuleFit training completed!")

# Extract rules
print("Rules extracted:")
rules = rf.get_rules()
print(rules)

# Save rules to text file
output_file = "rulefit_rules.txt"
with open(output_file, 'w') as f:
    f.write("RuleFit Rules for Reimbursement System\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total number of rules: {len(rules)}\n\n")
    
    # Set pandas options to display all rows and columns
    import pandas as pd
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.max_colwidth', None):
        f.write(str(rules))
    
    f.write("\n\n" + "=" * 50 + "\n")
    f.write("Detailed Rule Breakdown:\n")
    f.write("=" * 50 + "\n\n")
    
    # Write each rule individually for better readability
    for idx, row in rules.iterrows():
        f.write(f"Rule {idx}:\n")
        f.write(f"  Type: {row['type']}\n")
        f.write(f"  Rule: {row['rule']}\n")
        f.write(f"  Coefficient: {row['coef']:.6f}\n")
        f.write(f"  Support: {row['support']:.6f}\n")
        f.write(f"  Importance: {row['importance']:.6f}\n")
        f.write("-" * 30 + "\n")

print(f"\nRules saved to {output_file}") 