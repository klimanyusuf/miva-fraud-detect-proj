# Nigerian Bank Fraud Detection System

## 🏦 Overview
A comprehensive fraud detection system for Nigerian digital banking using CRISP-DM methodology with multiple machine learning models. Detects fraudulent transactions in real-time with advanced analytics by Nigerian banks and transaction patterns.

## ✨ Features
- **4 ML Models**: Random Forest (87.5%), SVM (90%), Isolation Forest (89.5%), Ensemble (89.5%)
- **Nigerian Context**: 15 major Nigerian banks, BVN verification, USSD patterns, local city analysis
- **Real-time Dashboard**: Streamlit interface with live fraud detection and analytics
- **Bank-level Analytics**: Fraud rate breakdown by Nigerian bank
- **Feature Analysis**: Feature importance for Random Forest & Isolation Forest models
- **Case Management**: Review and escalate flagged transactions
- **CRISP-DM Framework**: Structured six-phase methodology

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/klimanyusuf/miva-fraud-detect-proj.git
cd miva-fraud-detect-proj

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the App
```bash
streamlit run app.py
```
Visit `http://localhost:8501` in your browser.

## 📊 Dashboard Pages

### 1. **Dashboard** 
- System overview with all 4 model accuracies
- Recent transactions with bank names
- Fraud distribution by Location, Device Type, and Nigerian Bank

### 2. **Fraud Detection**
- Real-time transaction analysis
- Predictions from all 4 models
- Risk factor breakdown and recommendations

### 3. **Model Comparison**
- Performance metrics table (Accuracy, Precision, Recall, F1-Score)
- Feature Importance charts for Random Forest and Isolation Forest
- Model comparison visualization

### 4. **Analytics**
- Temporal patterns and trends
- Geographic analysis by Nigerian city
- Customer segmentation and device type analysis

### 5. **Model Settings** ⚙️
- Configure dataset size
- Adjust model parameters
- Toggle individual models

### 6. **Case Management**
- Review flagged fraud cases
- Approve/Reject/Escalate transactions
- View case details and risk factors

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Best For |
|-------|----------|-----------|--------|----------|----------|
| **SVM** ⭐ | 90.0% | 10.53% | 40.0% | 16.67% | Best overall performance |
| Isolation Forest | 89.5% | 10.0% | 40.0% | 16.0% | Outlier detection |
| Ensemble | 89.5% | 10.0% | 40.0% | 16.0% | Voting combination |
| Random Forest | 87.5% | 8.33% | 40.0% | 13.79% | Feature importance |

**Recommendation:** SVM is the best model with highest precision and F1-score, balancing fraud detection and false alarm rates.

## 🏦 Supported Nigerian Banks

1. First Bank of Nigeria
2. Zenith Bank
3. United Bank for Africa (UBA)
4. Guaranty Trust Bank (GTBank)
5. Access Bank
6. EcoBank Nigeria
7. Fidelity Bank
8. Union Bank of Nigeria
9. Stanbic IBTC Bank
10. FCMB
11. Sterling Bank
12. Wema Bank
13. Unity Bank
14. Polaris Bank
15. Keystone Bank

## 📍 Covered Nigerian Cities

Lagos, Abuja, Kano, Port Harcourt, Ibadan, Kaduna, Benin City, Aba, Jos, Enugu

## 🔍 Fraud Detection Patterns

The system detects fraud based on:
- Large amounts + new accounts + quick transactions
- Foreign transfers via USSD
- Late-night high-value transactions
- Unverified BVN with suspicious behavior
- Weekend unusual activity
- Multiple rapid transactions

See [DOCUMENTATION.md](DOCUMENTATION.md) for detailed technical architecture.

## 📚 Project Structure

```
.
├── app.py                 # Streamlit dashboard application
├── data_generator.py      # Nigerian dataset generation
├── model_trainer.py       # ML model training (RF, SVM, IF, Ensemble)
├── utils.py              # Risk calculation and recommendations
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── DOCUMENTATION.md     # Detailed technical documentation
```

## 🛠 Technologies

- **Framework**: Streamlit (UI/Dashboard)
- **ML**: scikit-learn (Random Forest, SVM, Isolation Forest)
- **Data**: pandas, numpy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Preprocessing**: StandardScaler, feature engineering

## 📝 Notes

- Fraud rate in dataset: 2.5% (realistic for banking)
- Test/Train split: 80/20 with stratification
- Feature scaling: StandardScaler normalization
- Class weight: Balanced to handle imbalanced data

## 👤 Developer

**Khalid Yusuf Liman**  
MIT 8212 Seminar: Industry Applications & Management in IT

## 📄 License

MIT License

---

For detailed technical documentation, see [DOCUMENTATION.md](DOCUMENTATION.md)
