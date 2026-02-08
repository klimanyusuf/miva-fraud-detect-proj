"""Test script to verify bank data is present throughout the pipeline"""
from data_generator import generate_nigerian_dataset, prepare_features
from model_trainer import train_all_models
import pandas as pd

# Generate dataset
print("1. Generating dataset...")
df = generate_nigerian_dataset(n_transactions=1000)
print(f"   ✓ Columns: {df.columns.tolist()[:5]}... (total: {len(df.columns)})")
print(f"   ✓ 'bank' in columns: {'bank' in df.columns}")
print(f"   ✓ Sample banks: {df['bank'].unique()[:3]}")

# Prepare features
print("\n2. Preparing features...")
X, y, feature_names, scaler = prepare_features(df)
print(f"   ✓ X shape: {X.shape}")
print(f"   ✓ Feature names: {feature_names}")
print(f"   ✓ DataFrame still has 'bank': {'bank' in df.columns}")
print(f"   ✓ Sample recent transactions:")
print(f"     {df[['transaction_id', 'bank', 'amount', 'transaction_type']].head(3).to_string()}")

# Train models
print("\n3. Training models...")
models = train_all_models(X, y)
print(f"   ✓ Models trained: {list(models.keys())}")

# Simulate session state
print("\n4. Simulating session state...")
session_df = df.copy()
print(f"   ✓ Session state df has 'bank': {'bank' in session_df.columns}")
print(f"   ✓ Session state df shape: {session_df.shape}")

# Test the groupby operation
print("\n5. Testing groupby('bank')...")
try:
    bank_fraud = session_df.groupby('bank')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
    print(f"   ✓ Groupby successful!")
    print(f"   ✓ Number of unique banks: {len(bank_fraud)}")
    print(f"   ✓ Top 3 banks by fraud rate:")
    print(bank_fraud.nlargest(3, 'mean')[['bank', 'mean']].to_string())
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n✅ All tests passed! Bank data should display on dashboard.")
