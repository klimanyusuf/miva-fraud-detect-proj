from app import generate_nigerian_banking_data, calculate_nigerian_risk_score

if __name__ == '__main__':
    print('Starting smoke test...')
    df = generate_nigerian_banking_data(300)
    print('Generated dataframe shape:', df.shape)
    print('Fraud cases (sum):', int(df['is_fraud'].sum()))
    print('Risk score stats (min/mean/max):', df['risk_score'].min(), df['risk_score'].mean(), df['risk_score'].max())
    # sample row
    sample = df.iloc[0]
    print('Sample transaction id:', sample['transaction_id'])
    print('Sample amount:', sample['amount'])
    print('Sample risk score:', sample['risk_score'])
    print('Smoke test completed successfully.')
