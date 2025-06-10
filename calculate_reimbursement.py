import sys
import math

"""
📈 Results Summary:
  Total test cases: 1000
  Successful runs: 1000
  Exact matches (±$0.01): 0 (0%)
  Close matches (±$1.00): 2 (.2%)
  Average error: $315.34
  Maximum error: $1843.53

🎯 Your Score: 31634.00 (lower is better)

"""


def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Calculate reimbursement using CN2 rule-based insights from Orange3 analysis.
    
    This implementation follows the discovered rule patterns more closely,
    using the specific thresholds and ranges identified by the CN2 algorithm.
    
    Key insights from CN2 analysis:
    - 10 distinct reimbursement ranges: $115-$2338
    - Rules primarily use receipt amount, mileage, and duration thresholds
    - Complex multi-condition logic with specific breakpoints
    """
    
    # Determine the reimbursement range based on CN2 rule patterns
    # The ranges discovered were:
    # Range 0: $115-$339, Range 1: $339-$561, Range 2: $561-$783
    # Range 3: $783-$1005, Range 4: $1005-$1227, Range 5: $1227-$1450
    # Range 6: $1450-$1672, Range 7: $1672-$1894, Range 8: $1894-$2116, Range 9: $2116-$2338
    
    # Start with base calculation using patterns from successful rules
    base_amount = 115.0  # Minimum from CN2 analysis
    
    # Apply the most frequent and successful rule patterns from CN2 output
    
    # Very high mileage rules (from CN2 Rule 409, 217, 218, etc.)
    if miles_traveled >= 997.0:
        return 2200.0 + (total_receipts_amount * 0.05)
    
    # High mileage with long duration (from multiple CN2 rules)
    if miles_traveled >= 1062.0 and trip_duration_days >= 5.0:
        if trip_duration_days >= 11.0:
            return 1950.0 + (total_receipts_amount * 0.08)
        else:
            return 1750.0 + (total_receipts_amount * 0.06)
    
    # High receipts with decent conditions (from CN2 Rule 393 and similar)
    if total_receipts_amount >= 1007.48 and miles_traveled >= 437.0:
        base_amount = 1450.0 + (trip_duration_days * 25.0) + (miles_traveled * 0.15)
    
    # Medium-high conditions (from frequent CN2 patterns)
    elif miles_traveled >= 948.0:
        base_amount = 1450.0 + (trip_duration_days * 20.0) + (total_receipts_amount * 0.1)
    
    elif miles_traveled >= 886.0:
        base_amount = 1350.0 + (trip_duration_days * 18.0) + (total_receipts_amount * 0.08)
    
    elif total_receipts_amount >= 1127.87:
        # High receipts threshold (very common in CN2 rules)
        if miles_traveled >= 756.0 and trip_duration_days >= 11.0:
            base_amount = 1600.0 + (miles_traveled * 0.1)
        elif miles_traveled >= 576.0:
            base_amount = 1200.0 + (trip_duration_days * 15.0) + (miles_traveled * 0.12)
        else:
            base_amount = 1000.0 + (trip_duration_days * 12.0) + (total_receipts_amount * 0.06)
    
    # Medium range conditions (from CN2 Range 4-5 rules)
    elif miles_traveled >= 576.0 and total_receipts_amount >= 500.0:
        if trip_duration_days >= 5.0:
            base_amount = 800.0 + (trip_duration_days * 35.0) + (miles_traveled * 0.3) + (total_receipts_amount * 0.15)
        else:
            base_amount = 600.0 + (trip_duration_days * 40.0) + (miles_traveled * 0.25) + (total_receipts_amount * 0.12)
    
    # Lower-medium range (from CN2 Range 2-3 rules)
    elif trip_duration_days >= 5.0:
        if total_receipts_amount >= 300.0:
            base_amount = 500.0 + (trip_duration_days * 45.0) + (miles_traveled * 0.4) + (total_receipts_amount * 0.2)
        else:
            base_amount = 400.0 + (trip_duration_days * 50.0) + (miles_traveled * 0.35) + (total_receipts_amount * 0.25)
    
    # Low range conditions (from CN2 Range 0-1 rules)
    else:
        if trip_duration_days <= 2.0:
            base_amount = 115.0 + (trip_duration_days * 30.0) + (miles_traveled * 0.5) + (total_receipts_amount * 0.3)
        else:
            base_amount = 200.0 + (trip_duration_days * 60.0) + (miles_traveled * 0.6) + (total_receipts_amount * 0.4)
    
    # Apply specific adjustments based on CN2 rule patterns
    
    # Duration-based adjustments (from rule frequency analysis)
    if trip_duration_days >= 14.0:
        base_amount *= 1.2
    elif trip_duration_days >= 12.0:
        base_amount *= 1.15
    elif trip_duration_days >= 9.0:
        base_amount *= 1.1
    elif trip_duration_days >= 6.0:
        base_amount *= 1.05
    elif trip_duration_days <= 1.0:
        base_amount *= 0.8
    
    # Special penalty cases (from specific CN2 rules)
    if (miles_traveled >= 576.0 and 
        total_receipts_amount <= 651.64 and 
        trip_duration_days <= 2.0):
        # High mileage but very short trip with low receipts
        base_amount *= 0.6
    
    if (total_receipts_amount <= 130.07 and 
        trip_duration_days <= 3.0):
        # Very low receipts and short trips
        base_amount *= 0.7
    
    # Receipt-based fine-tuning (from CN2 patterns)
    if total_receipts_amount >= 2000.0:
        base_amount += total_receipts_amount * 0.08
    elif total_receipts_amount >= 1500.0:
        base_amount += total_receipts_amount * 0.06
    elif total_receipts_amount >= 1000.0:
        base_amount += total_receipts_amount * 0.04
    elif total_receipts_amount <= 50.0:
        base_amount -= 30.0
    
    # Mileage fine-tuning (from CN2 patterns)
    if miles_traveled >= 1200.0:
        base_amount += miles_traveled * 0.05
    elif miles_traveled <= 100.0:
        base_amount -= 20.0
    
    # Apply range constraints from CN2 analysis
    if base_amount < 115.0:
        base_amount = 115.0
    elif base_amount > 2338.0:
        base_amount = 2338.0
    
    # Final smoothing based on the complexity of conditions
    # This accounts for the interaction effects seen in CN2 rules
    interaction_bonus = 0
    
    if (trip_duration_days >= 5 and 
        miles_traveled >= 500 and 
        total_receipts_amount >= 500):
        interaction_bonus = 50.0
    
    if (trip_duration_days >= 10 and 
        miles_traveled >= 800 and 
        total_receipts_amount >= 1000):
        interaction_bonus = 100.0
    
    base_amount += interaction_bonus
    
    return round(base_amount, 2)


def main():
    if len(sys.argv) != 4:
        print("Usage: python calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        trip_duration_days = int(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
        
        reimbursement = calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount)
        print(f"{reimbursement:.2f}")
        
    except ValueError as e:
        print(f"Error: Invalid input - {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 