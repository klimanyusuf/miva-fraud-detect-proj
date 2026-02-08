"""
Data generation module for Nigerian banking fraud detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

def generate_nigerian_dataset(n_transactions=1000):
    """
    Generate simulated Nigerian banking dataset with fraud patterns
    
    Args:
        n_transactions: Number of transactions to generate
        
    Returns:
        pandas.DataFrame: Generated dataset
    """
    np.random.seed(42)
    
    # Nigerian bank names
    nigerian_banks = [
        "First Bank of Nigeria", "Zenith Bank", "United Bank for Africa (UBA)",
        "Guaranty Trust Bank (GTBank)", "Access Bank", "EcoBank Nigeria",
        "Fidelity Bank", "Union Bank of Nigeria", "Stanbic IBTC Bank",
        "First City Monument Bank (FCMB)", "Sterling Bank", "Wema Bank",
        "Unity Bank", "Polaris Bank", "Keystone Bank"
    ]
    
    # Base transaction data
    data = {
        'transaction_id': range(1000, 1000 + n_transactions),
        'timestamp': pd.date_range('2024-01-01', periods=n_transactions, freq='H'),
        'bank': np.random.choice(nigerian_banks, n_transactions),
        'amount': np.random.exponential(50000, n_transactions),
        'customer_id': np.random.randint(100, 500, n_transactions),
        'account_age_days': np.random.randint(1, 3650, n_transactions),
        'location': np.random.choice(
            ['Lagos', 'Abuja', 'Kano', 'Port Harcourt', 'Ibadan', 
             'Kaduna', 'Benin City', 'Aba', 'Jos', 'Enugu'], 
            n_transactions
        ),
        'device_type': np.random.choice(
            ['Mobile', 'Web', 'USSD', 'ATM'], 
            n_transactions,
            p=[0.5, 0.3, 0.15, 0.05]
        ),
        'transaction_type': np.random.choice(
            ['Transfer', 'Bill Payment', 'Airtime', 'Withdrawal', 'Deposit'], 
            n_transactions
        ),
        'beneficiary_history': np.random.randint(0, 50, n_transactions),
        'time_since_last_transaction': np.random.exponential(24, n_transactions),
        'is_foreign': np.random.choice([0, 1], n_transactions, p=[0.93, 0.07]),
        'transaction_hour': np.random.randint(0, 24, n_transactions),
        'bvn_verified': np.random.choice([0, 1], n_transactions, p=[0.12, 0.88]),
        'weekday': np.random.choice([0, 1], n_transactions, p=[0.72, 0.28]),
    }
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['month'] = df['timestamp'].dt.month
    df['hour_category'] = pd.cut(
        df['transaction_hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening']
    )
    
    # Generate fraud labels based on Nigerian patterns
    fraud_conditions = generate_fraud_conditions(df)
    df['is_fraud'] = np.where(fraud_conditions, 1, 0)
    
    # Adjust fraud rate to ~2.5%
    fraud_indices = df[df['is_fraud'] == 1].index
    target_fraud = int(n_transactions * 0.025)
    
    if len(fraud_indices) > target_fraud:
        keep = np.random.choice(fraud_indices, target_fraud, replace=False)
        df.loc[~df.index.isin(keep), 'is_fraud'] = 0
    elif len(fraud_indices) < target_fraud:
        additional = target_fraud - len(fraud_indices)
        # Add more fraud cases from high-risk transactions
        high_risk = df[df['is_fraud'] == 0].copy()
        high_risk['risk_score'] = (
            (high_risk['amount'] > 300000).astype(int) * 3 +
            (high_risk['bvn_verified'] == 0).astype(int) * 2 +
            (high_risk['is_weekend'] == 1).astype(int) +
            (high_risk['device_type'] == 'USSD').astype(int)
        )
        additional_fraud = high_risk.nlargest(additional, 'risk_score').index
        df.loc[additional_fraud, 'is_fraud'] = 1
    
    return df

def generate_fraud_conditions(df):
    """
    Generate fraud conditions based on Nigerian patterns
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        Boolean array indicating fraud conditions
    """
    conditions = (
        # Pattern 1: Large amount + new account + quick succession
        (df['amount'] > 300000) & 
        (df['time_since_last_transaction'] < 0.5) &
        (df['account_age_days'] < 30)
    ) | (
        # Pattern 2: Foreign + USSD + moderate amount
        (df['is_foreign'] == 1) & 
        (df['device_type'] == 'USSD') &
        (df['amount'] > 100000)
    ) | (
        # Pattern 3: Late night + large amount
        (df['transaction_hour'].between(1, 5)) & 
        (df['amount'] > 200000)
    ) | (
        # Pattern 4: Unverified BVN + suspicious behavior
        (df['bvn_verified'] == 0) &
        (df['amount'] > 150000) &
        (df['account_age_days'] < 90)
    ) | (
        # Pattern 5: Weekend + unusual activity
        (df['is_weekend'] == 1) &
        (df['transaction_hour'].between(0, 6)) &
        (df['amount'] > 100000) &
        (df['beneficiary_history'] < 3)
    ) | (
        # Pattern 6: Unverified BVN + weekend + USSD
        (df['bvn_verified'] == 0) &
        (df['is_weekend'] == 1) &
        (df['device_type'] == 'USSD') &
        (df['amount'] > 50000)
    ) | (
        # Pattern 7: Multiple transactions in short time
        (df['time_since_last_transaction'] < 0.1) &
        (df['amount'] > 50000) &
        (df['beneficiary_history'] == 0)
    )
    
    return conditions

def prepare_features(df):
    """
    Prepare features for model training
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        tuple: (X_scaled, y, feature_names, scaler)
    """
    # Select features
    feature_names = [
        'amount', 
        'account_age_days', 
        'beneficiary_history',
        'time_since_last_transaction', 
        'is_foreign', 
        'transaction_hour',
        'bvn_verified', 
        'is_weekend'
    ]
    
    X = df[feature_names].copy()
    y = df['is_fraud'].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_names, scaler