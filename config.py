"""
Configuration file for fraud detection system
"""

# Nigerian banking parameters
NIGERIAN_PARAMS = {
    'fraud_rate': 0.025,          # 2.5% fraud rate (NIBSS 2024)
    'bvn_unverified_rate': 0.12,  # 12% unverified BVN (CBN 2024)
    'weekend_rate': 0.28,         # 28% weekend transactions
    'ussd_rate': 0.15,            # 15% USSD transactions
    'foreign_rate': 0.07,         # 7% foreign transactions
}

# Model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 150,
        'max_depth': 12,
        'class_weight': 'balanced'
    },
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'probability': True,
        'class_weight': 'balanced'
    },
    'isolation_forest': {
        'contamination': 0.03,
        'n_estimators': 100
    }
}

# Risk scoring thresholds
RISK_THRESHOLDS = {
    'low': 30,
    'medium': 60,
    'high': 80
}

# Nigerian locations
NIGERIAN_LOCATIONS = [
    'Lagos', 'Abuja', 'Kano', 'Port Harcourt', 'Ibadan',
    'Kaduna', 'Benin City', 'Aba', 'Jos', 'Enugu'
]

# Transaction types
TRANSACTION_TYPES = [
    'Transfer', 'Bill Payment', 'Airtime', 'Withdrawal', 'Deposit'
]

# Device types
DEVICE_TYPES = ['Mobile', 'Web', 'USSD', 'ATM']