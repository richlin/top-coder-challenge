#!/usr/bin/env python3

import json
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def load_training_data():
    """Load the training data from public_cases.json"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    return data

def prepare_data(data):
    """Convert the data into a format suitable for training"""
    X = []
    y = []
    for case in data:
        X.append([
            float(case['input']['trip_duration_days']),
            float(case['input']['miles_traveled']),
            float(case['input']['total_receipts_amount'])
        ])
        y.append(float(case['output']))
    return np.array(X), np.array(y)

def train_model():
    """Train a model on the public cases"""
    data = load_training_data()
    X, y = prepare_data(data)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

def calculate_reimbursement(days, miles, receipts):
    """Calculate reimbursement using the trained model"""
    model, scaler = train_model()
    
    # Prepare input
    X = np.array([[float(days), float(miles), float(receipts)]])
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    return round(prediction, 2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    days = sys.argv[1]
    miles = sys.argv[2]
    receipts = sys.argv[3]
    
    result = calculate_reimbursement(days, miles, receipts)
    print(result) 