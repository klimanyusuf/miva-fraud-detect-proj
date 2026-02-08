from data_generator import generate_nigerian_dataset, prepare_features
from model_trainer import train_all_models

print('Generating data...')
df = generate_nigerian_dataset(n_transactions=1000)
print(f'Fraud rate in data: {df["is_fraud"].mean():.2%}')
print(f'Total frauds: {df["is_fraud"].sum()} out of {len(df)}')

X, y, feature_names, scaler = prepare_features(df)

print(f'\nTraining models on {X.shape[0]} samples...')
models = train_all_models(X, y)

print('\n' + '='*60)
print('FINAL METRICS:')
print('='*60)
for model_name, metrics in models.items():
    print(f'\n{model_name}:')
    print(f'  Accuracy:  {metrics["accuracy"]:.2%}')
    print(f'  Precision: {metrics["precision"]:.2%}')
    print(f'  Recall:    {metrics["recall"]:.2%}')
    print(f'  F1-Score:  {metrics["f1"]:.2%}')
