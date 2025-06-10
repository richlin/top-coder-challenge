# pip install orange3

import Orange
import numpy as np
from typing import List, Tuple, Optional
from sklearn.linear_model import LinearRegression
import json
import pandas as pd


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

# Create categorical bins for the target variable to enable rule learning
df['output_category'] = pd.cut(df['expected_output'], bins=10, labels=False)  # Use numeric labels
category_ranges = pd.cut(df['expected_output'], bins=10, retbins=True)[1]

# Create Orange data table properly
X = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']].values
y = df['output_category'].values.astype(float)

# Create domain with proper category names
category_names = [f'Range_{i}_({category_ranges[i]:.0f}-{category_ranges[i+1]:.0f})' 
                  for i in range(len(category_ranges)-1)]

domain = Orange.data.Domain(
    [Orange.data.ContinuousVariable('trip_duration_days'),
     Orange.data.ContinuousVariable('miles_traveled'), 
     Orange.data.ContinuousVariable('total_receipts_amount')],
    Orange.data.DiscreteVariable('output_category', values=category_names)
)

# Create data table with proper domain
data = Orange.data.Table.from_numpy(domain, X, y)

# Train CN2 rule learner
learner = Orange.classification.rules.CN2Learner()
classifier = learner(data)

# Print rules
print("Discovered Rules:")
print("=" * 50)
for i, rule in enumerate(classifier.rule_list):
    print(f"Rule {i+1}: {rule}")
    print()

# Also print some statistics about the data
print("\nData Statistics:")
print("=" * 50)
print(f"Total cases: {len(df)}")
print(f"Output range: ${df['expected_output'].min():.2f} - ${df['expected_output'].max():.2f}")
print(f"Average output: ${df['expected_output'].mean():.2f}")
print("\nOutput distribution by category:")
for i, count in enumerate(np.bincount(df['output_category'].dropna().astype(int))):
    if count > 0:
        range_start = category_ranges[i]
        range_end = category_ranges[i+1]
        print(f"Range {i} (${range_start:.0f}-${range_end:.0f}): {count} cases")

