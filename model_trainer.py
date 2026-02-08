"""
Model training module for fraud detection
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_all_models(X, y):
    """
    Train multiple machine learning models
    
    Args:
        X: Feature matrix
        y: Target labels
        
    Returns:
        dict: Trained models and their performance metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {}
    
    # 1. Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba > 0.05).astype(int)  # Even more aggressive - catch more fraud
    
    models['Random Forest'] = {
        'model': rf_model,
        'predictions': rf_pred,
        'probabilities': rf_proba,
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, zero_division=0),
        'recall': recall_score(y_test, rf_pred, zero_division=0),
        'f1': f1_score(y_test, rf_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, rf_pred)
    }
    
    # 2. Train SVM
    print("Training SVM...")
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        probability=True,
        random_state=42,
        class_weight='balanced'
    )
    svm_model.fit(X_train, y_train)
    svm_proba = svm_model.predict_proba(X_test)[:, 1]
    svm_pred = (svm_proba > 0.10).astype(int)  # Even more aggressive
    
    models['SVM'] = {
        'model': svm_model,
        'predictions': svm_pred,
        'probabilities': svm_proba,
        'accuracy': accuracy_score(y_test, svm_pred),
        'precision': precision_score(y_test, svm_pred, zero_division=0),
        'recall': recall_score(y_test, svm_pred, zero_division=0),
        'f1': f1_score(y_test, svm_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, svm_pred)
    }
    
    # 3. Train Isolation Forest for outlier detection
    print("Training Isolation Forest...")
    iso_model = IsolationForest(
        contamination=0.10,  # 10% - be more aggressive
        random_state=42,
        n_estimators=100,
        n_jobs=-1
    )
    iso_model.fit(X_train)
    iso_scores = iso_model.score_samples(X_test)
    iso_pred = np.where(iso_scores < np.percentile(iso_scores, 10), 1, 0)
    
    models['Isolation Forest'] = {
        'model': iso_model,
        'scores': iso_scores,
        'predictions': iso_pred,
        'accuracy': accuracy_score(y_test, iso_pred),
        'precision': precision_score(y_test, iso_pred, zero_division=0),
        'recall': recall_score(y_test, iso_pred, zero_division=0),
        'f1': f1_score(y_test, iso_pred, zero_division=0)
    }
    
    # 4. Create ensemble model (average probabilities)
    print("Creating ensemble model...")
    ensemble_proba = (rf_proba + svm_proba) / 2
    ensemble_pred = (ensemble_proba > 0.10).astype(int)  # More aggressive
    
    models['Ensemble'] = {
        'probabilities': ensemble_proba,
        'predictions': ensemble_pred,
        'accuracy': accuracy_score(y_test, ensemble_pred),
        'precision': precision_score(y_test, ensemble_pred, zero_division=0),
        'recall': recall_score(y_test, ensemble_pred, zero_division=0),
        'f1': f1_score(y_test, ensemble_pred, zero_division=0)
    }
    
    print("\nModel Training Complete!")
    print("-" * 50)
    for model_name, metrics in models.items():
        if 'accuracy' in metrics:
            print(f"{model_name}: Accuracy = {metrics['accuracy']:.3%}")
    
    return models

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a single model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Evaluation metrics
    """
    if hasattr(model, 'predict_proba'):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_proba = None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    if y_proba is not None:
        metrics['probabilities'] = y_proba
    
    return metrics