# Nigerian Bank Fraud Detection System - Technical Documentation

## Table of Contents
1. [System Architecture](#system-architecture)
2. [CRISP-DM Methodology](#crisp-dm-methodology)
3. [Data Generation](#data-generation)
4. [Feature Engineering](#feature-engineering)
5. [Model Training](#model-training)
6. [Model Tuning & Thresholds](#model-tuning--thresholds)
7. [Dashboard Guide](#dashboard-guide)
8. [Fraud Detection Patterns](#fraud-detection-patterns)

---

## System Architecture

### Components

```
┌─────────────────────────────────────────────────────┐
│  Streamlit Dashboard (app.py)                       │
│  ├─ Dashboard: Overview & Analytics                │
│  ├─ Fraud Detection: Real-time predictions         │
│  ├─ Model Comparison: Performance metrics          │
│  ├─ Analytics: Trends & patterns                   │
│  ├─ Model Settings: Configuration                  │
│  └─ Case Management: Flagged transactions          │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│  Model Training Pipeline (model_trainer.py)        │
│  ├─ Random Forest Classifier                       │
│  ├─ Support Vector Machine (SVM)                   │
│  ├─ Isolation Forest (Outlier detection)           │
│  └─ Ensemble (Voting combination)                  │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│  Data Generation (data_generator.py)               │
│  ├─ 1000 Nigerian banking transactions             │
│  ├─ 2.5% fraud rate                                │
│  ├─ 15 Nigerian banks                              │
│  └─ 10 Nigerian cities                             │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│  Utilities (utils.py)                              │
│  ├─ Risk factor calculation                        │
│  └─ Recommendation generation                      │
└─────────────────────────────────────────────────────┘
```

---

## CRISP-DM Methodology

The system follows the six-phase CRISP-DM (Cross Industry Standard Process for Data Mining) approach:

### 1. **Business Understanding**
- **Problem**: Detect fraudulent transactions in Nigerian digital banking
- **Goal**: Minimize fraud loss while reducing false positives
- **Stakeholders**: Banks, customers, compliance teams
- **Success Metrics**: Recall ≥ 40%, Precision ≥ 8%

### 2. **Data Understanding**
- **Dataset Size**: 1,000 transactions
- **Fraud Rate**: 2.5% (realistic for banking)
- **Time Period**: 1 month of synthetic data
- **Data Quality**: No missing values, stratified split

### 3. **Data Preparation**
- **Features**: 8 key features (see Feature Engineering section)
- **Scaling**: StandardScaler normalization
- **Train/Test Split**: 80/20 with stratification
- **Class Balance**: Applied class weights to handle imbalance

### 4. **Modeling**
- **Models Trained**: 4 different algorithms
- **Training Approach**: Scikit-learn with optimized hyperparameters
- **Ensemble Method**: Average probability voting

### 5. **Evaluation**
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Best Model**: SVM (90% accuracy, 16.67% F1-score)
- **Validation**: K-fold cross-validation on training set

### 6. **Deployment**
- **Platform**: Streamlit (real-time interactive dashboard)
- **Model Persistence**: joblib serialization
- **Inference**: Sub-second prediction on new transactions

---

## Data Generation

### Transaction Features

```python
# Base Transaction Data (data_generator.py)
- transaction_id: Unique identifier (1000-1999)
- timestamp: Transaction datetime (hourly resolution)
- bank: One of 15 Nigerian banks
- amount: Exponential distribution (mean ~50,000 ₦)
- customer_id: Customer identifier (100-500)
- account_age_days: Days since account creation (1-3650 days)
- location: One of 10 Nigerian cities
- device_type: Mobile (50%), Web (30%), USSD (15%), ATM (5%)
- transaction_type: Transfer, Bill Payment, Airtime, Withdrawal, Deposit
- beneficiary_history: Number of previous transactions to beneficiary
- time_since_last_transaction: Hours since last transaction
- is_foreign: Binary flag for international transfers
- transaction_hour: Hour of transaction (0-23)
- bvn_verified: Binary flag for BVN verification status
- weekday: Binary flag for weekday (1) vs weekend (0)
```

### Derived Features

- `day_of_week`: Calculated from timestamp
- `is_weekend`: Boolean flag for Saturday/Sunday
- `month`: Extracted from timestamp
- `hour_category`: Binned into Night/Morning/Afternoon/Evening

---

## Feature Engineering

### Selected Features for Training

```python
feature_names = [
    'amount',                          # Transaction size
    'account_age_days',               # Account maturity
    'beneficiary_history',            # Trust score
    'time_since_last_transaction',    # Activity pattern
    'is_foreign',                     # Geographic risk
    'transaction_hour',               # Temporal pattern
    'bvn_verified',                   # Identity verification
    'is_weekend'                      # Behavioral pattern
]
```

### Feature Importance Ranking (Random Forest)

Based on model training results:
1. **amount** - Transaction size is strongest predictor
2. **account_age_days** - New accounts have higher fraud risk
3. **bvn_verified** - Unverified accounts are risky
4. **is_weekend** - Weekend transactions have different patterns
5. **transaction_hour** - Late-night transactions are suspicious
6. **is_foreign** - International transfers require scrutiny
7. **time_since_last_transaction** - Rapid succession is suspicious
8. **beneficiary_history** - New beneficiaries are riskier

### Scaling

All features normalized using StandardScaler:
```
X_scaled = (X - mean) / std_dev
```
This ensures features with different ranges (e.g., amount vs. day_of_week) are comparable.

---

## Model Training

### 1. Random Forest Classifier

**Configuration:**
```python
RandomForestClassifier(
    n_estimators=150,      # 150 trees in ensemble
    max_depth=12,          # Limit tree depth to prevent overfitting
    random_state=42,       # Reproducibility
    class_weight='balanced',# Handle class imbalance
    n_jobs=-1              # Use all CPU cores
)
```

**Performance:**
- Accuracy: 87.50%
- Precision: 8.33%
- Recall: 40.00%
- F1-Score: 13.79%

**Threshold:** 0.05 (probability above 5% flagged as fraud)

---

### 2. Support Vector Machine (SVM) ⭐ **BEST MODEL**

**Configuration:**
```python
SVC(
    kernel='rbf',           # Radial Basis Function
    C=1.0,                  # Regularization strength
    probability=True,       # Enable probability calibration
    random_state=42,
    class_weight='balanced' # Handle imbalance
)
```

**Performance:**
- Accuracy: 90.00% ⭐
- Precision: 10.53% ⭐
- Recall: 40.00%
- F1-Score: 16.67% ⭐

**Threshold:** 0.10 (probability above 10% flagged as fraud)

**Why SVM Wins:**
- Best precision-recall balance
- Highest F1-score (most reliable)
- Highest accuracy overall
- Fewer false alarms than RF
- Strong generalization

---

### 3. Isolation Forest

**Configuration:**
```python
IsolationForest(
    contamination=0.10,    # Expect 10% anomalies
    random_state=42,
    n_estimators=100,
    n_jobs=-1
)
```

**Performance:**
- Accuracy: 89.50%
- Precision: 10.00%
- Recall: 40.00%
- F1-Score: 16.00%

**Threshold:** 10th percentile of anomaly scores

**Why Use It:**
- Unsupervised outlier detection
- Catches unusual patterns
- Complements supervised models
- Fast and efficient

---

### 4. Ensemble Model

**Method:** Average probability voting
```python
ensemble_proba = (rf_proba + svm_proba) / 2
ensemble_pred = (ensemble_proba > 0.10).astype(int)
```

**Performance:**
- Accuracy: 89.50%
- Precision: 10.00%
- Recall: 40.00%
- F1-Score: 16.00%

**Benefit:**
- Combines strengths of RF and SVM
- Reduces individual model biases
- More robust predictions

---

## Model Tuning & Thresholds

### Why Low Thresholds?

With imbalanced data (2.5% fraud), default 0.5 probability threshold leaves fraud undetected. Solution: Lower thresholds to catch more fraud.

### Current Thresholds

| Model | Threshold | Rationale |
|-------|-----------|-----------|
| **SVM** | 0.10 | Best precision, catches fraud efficiently |
| **Random Forest** | 0.05 | More aggressive, higher recall |
| **Isolation Forest** | 10th percentile | Anomaly-based cutoff |
| **Ensemble** | 0.10 | Balanced combination |

### Precision-Recall Trade-off

```
Higher Threshold → Higher Precision, Lower Recall
   (Fewer false alarms, miss more fraud)

Lower Threshold → Lower Precision, Higher Recall
   (More false alarms, catch more fraud)
```

**Our Choice:** Lower precision acceptable to maximize fraud detection. Better to investigate 10 safe transactions than miss 1 fraud.

---

## Fraud Detection Patterns

The system flags transactions based on 7 key patterns:

### Pattern 1: Large Amount + New Account + Quick Succession
```
Condition: amount > 300,000 AND time_since_last < 0.5 hrs AND account_age < 30 days
Risk: New account, rapid large transfer = account takeover
```

### Pattern 2: Foreign + USSD + Moderate Amount
```
Condition: is_foreign=1 AND device_type='USSD' AND amount > 100,000
Risk: USSD is high-risk channel, international transfer
```

### Pattern 3: Late Night + Large Amount
```
Condition: transaction_hour IN [1-5] AND amount > 200,000
Risk: Late-night large transactions are suspicious
```

### Pattern 4: Unverified BVN + Suspicious Behavior
```
Condition: bvn_verified=0 AND amount > 150,000 AND account_age < 90 days
Risk: Unverified identity + new account + large transfer
```

### Pattern 5: Weekend + Unusual Activity
```
Condition: is_weekend=1 AND hour IN [0-6] AND amount > 100,000 AND beneficiary_history < 3
Risk: Weekend early-morning transfers to new beneficiary
```

### Pattern 6: Unverified BVN + Weekend + USSD
```
Condition: bvn_verified=0 AND is_weekend=1 AND device_type='USSD' AND amount > 50,000
Risk: Multiple high-risk factors combined
```

### Pattern 7: Rapid Successive Transactions
```
Condition: time_since_last < 0.1 hrs AND amount > 50,000 AND beneficiary_history=0
Risk: Multiple rapid transfers to new beneficiary
```

---

## Dashboard Guide

### Page 1: Dashboard
**Displays:**
- 6 Key Metrics: Transaction count, fraud count, and accuracy for each model
- Recent Transactions: Last 5 transactions with bank names
- Fraud Distribution: 3-column view (Location, Device Type, Bank)

**Use for:** Quick system health check and overview

### Page 2: Fraud Detection
**Displays:**
- Transaction analysis form
- Predictions from all 4 models
- Risk factors and recommendations
- Flagged transaction history

**Use for:** Investigate specific transactions

### Page 3: Model Comparison
**Displays:**
- Performance metrics table
- Feature importance tabs (RF, IF)
- Model comparison bar chart

**Use for:** Model evaluation and selection

### Page 4: Analytics
**Displays:**
- Temporal trends
- Geographic patterns by city
- Customer segmentation
- Device type analysis

**Use for:** Understand fraud patterns

### Page 5: Model Settings
**Displays:**
- Dataset configuration
- Model parameters
- Model selection toggles

**Use for:** Customize system behavior

### Page 6: Case Management
**Displays:**
- Flagged transaction review
- Case details with risk factors
- Action buttons: Approve/Reject/Escalate

**Use for:** Manage and review flagged cases

---

## Deployment Notes

### Streamlit Cloud Requirements
- `requirements.txt`: Flexible versions (>= instead of ==)
- `packages.txt`: Empty (avoid system-level dependencies)
- `setup.sh`: Documentation only (no executable commands)

### Local Development
- Python 3.11+
- Virtual environment recommended
- `pip install -r requirements.txt`

### Performance
- Model training: ~2-3 seconds (1000 transactions)
- Prediction: <100ms per transaction
- Dashboard refresh: Real-time

---

## Metrics Interpretation

### Why 40% Recall?
With 25 frauds in 200 test samples, 40% recall = catching 10 frauds
- Acceptable for fraud detection (better safe than sorry)
- Remaining 15 frauds caught by other systems/manual review

### Why 8-10% Precision?
10% precision means 9 false alarms per 1 true fraud
- Standard for fraud detection industry
- Better than missing fraud entirely
- Manageable alert volume for human review

### Why Lower Accuracy with Aggressive Thresholds?
Lower thresholds catch more fraud but also flag more safe transactions
- Accuracy drops but Recall increases
- Trade-off is intentional and beneficial
- Precision-Recall curve optimization

---

## Future Improvements

1. **SHAP Values**: Explainability for SVM & Ensemble
2. **Real-time Feedback**: Learn from human reviewer decisions
3. **Deep Learning**: LSTM for temporal patterns
4. **API Integration**: Connect to actual banking systems
5. **Geographic Clustering**: ML-based location risk scores
6. **Customer Profiling**: Behavioral baseline detection

---

## Contact & Support

**Developer:** Khalid Yusuf Liman  
**Course:** MIT 8212 Seminar  
**Repository:** https://github.com/klimanyusuf/miva-fraud-detect-proj

