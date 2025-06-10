from calculate_reimbursement import calculate_reimbursement
import numpy as np
import json

# Load test cases from public_cases.json
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

def calculate_mse(predictions, actuals):
    """Calculate Mean Squared Error between predictions and actual values."""
    return np.mean((np.array(predictions) - np.array(actuals)) ** 2)

def main():
    """Test each case individually and print the results."""
    predictions = []
    actuals = []
    
    print(f"\nTesting {len(CASES)} cases from public_cases.json:")
    print("=" * 80)
    print(f"{'Case':<5} {'Duration':<10} {'Miles':<10} {'Receipts':<12} {'Expected':<12} {'Calculated':<12} {'Difference':<12}")
    print("-" * 80)
    
    for i, (days, miles, receipts, expected) in enumerate(CASES, 1):
        calculated = calculate_reimbursement(days, miles, receipts)
        predictions.append(calculated)
        actuals.append(expected)
        
        difference = calculated - expected
        print(f"{i:<5} {days:<10} {miles:<10.1f} {receipts:<12.2f} {expected:<12.2f} {calculated:<12.2f} {difference:<12.2f}")
        
        # Print only first 20 cases to avoid overwhelming output
        if i >= 5:
            print(f"... (showing first 5 of {len(CASES)} cases)")
            break
    
    # Calculate and print MSE for all cases
    mse = calculate_mse(predictions, actuals)
    print(f"\nResults for all {len(CASES)} cases:")
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", np.sqrt(mse))
    
    # Calculate additional error metrics
    errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    average_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"Average Error (MAE): {average_error:.2f}")
    print(f"Maximum Error: {max_error:.2f}")
    
    # Calculate accuracy metrics
    exact_matches = sum(1 for p, a in zip(predictions, actuals) if abs(p - a) < 0.01)
    close_matches = sum(1 for p, a in zip(predictions, actuals) if abs(p - a) < 1.00)
    
    print(f"Exact matches (within $0.01): {exact_matches}/{len(CASES)} ({100*exact_matches/len(CASES):.1f}%)")
    print(f"Close matches (within $1.00): {close_matches}/{len(CASES)} ({100*close_matches/len(CASES):.1f}%)")

if __name__ == '__main__':
    # Load cases from JSON file
    CASES = load_cases_from_json()
    main() 