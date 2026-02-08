"""
Utility functions for fraud detection system
"""

def calculate_risk_factors(transaction_data):
    """
    Calculate risk factors for a transaction
    
    Args:
        transaction_data: Dictionary with transaction features
        
    Returns:
        list: List of risk factors with scores and descriptions
    """
    risk_factors = []
    
    # Amount risk
    if transaction_data['amount'] > 300000:
        risk_factors.append({
            'name': 'High Amount',
            'score': 25,
            'description': f"Transaction amount (₦{transaction_data['amount']:,}) exceeds normal threshold"
        })
    elif transaction_data['amount'] > 100000:
        risk_factors.append({
            'name': 'Moderate Amount',
            'score': 10,
            'description': f"Transaction amount (₦{transaction_data['amount']:,}) is above average"
        })
    
    # Account age risk
    if transaction_data['account_age_days'] < 30:
        risk_factors.append({
            'name': 'New Account',
            'score': 20,
            'description': f"Account is only {transaction_data['account_age_days']} days old"
        })
    elif transaction_data['account_age_days'] < 90:
        risk_factors.append({
            'name': 'Recent Account',
            'score': 10,
            'description': f"Account is {transaction_data['account_age_days']} days old"
        })
    
    # BVN verification risk
    if transaction_data.get('bvn_verified', 1) == 0:
        risk_factors.append({
            'name': 'Unverified BVN',
            'score': 30,
            'description': "Customer BVN is not verified"
        })
    
    # Time risk
    if transaction_data.get('is_weekend', 0) == 1:
        risk_factors.append({
            'name': 'Weekend Transaction',
            'score': 10,
            'description': "Transaction occurred on weekend"
        })
    
    if transaction_data['transaction_hour'] in range(1, 6):
        risk_factors.append({
            'name': 'Late Night Transaction',
            'score': 15,
            'description': f"Transaction at {transaction_data['transaction_hour']}:00 (unusual time)"
        })
    
    # Foreign transaction risk
    if transaction_data.get('is_foreign', 0) == 1:
        risk_factors.append({
            'name': 'Foreign Transaction',
            'score': 15,
            'description': "Transaction involves foreign currency/entity"
        })
    
    # Device risk
    if transaction_data.get('device_type', '') == 'USSD':
        risk_factors.append({
            'name': 'USSD Channel',
            'score': 10,
            'description': "Transaction via USSD (higher risk channel)"
        })
    
    # Quick succession risk
    if transaction_data.get('time_since_last_transaction', 24) < 0.5:
        risk_factors.append({
            'name': 'Quick Succession',
            'score': 15,
            'description': "Transaction soon after previous one"
        })
    
    return risk_factors

def generate_recommendations(transaction_data, model_results, risk_score):
    """
    Generate recommendations based on transaction analysis
    
    Args:
        transaction_data: Transaction features
        model_results: Model predictions
        risk_score: Calculated risk score
        
    Returns:
        list: List of recommendations
    """
    recommendations = []
    
    # Base recommendations
    if risk_score > 70:
        recommendations.append("🛑 **Block transaction immediately** and contact fraud team")
        recommendations.append("📞 **Call customer** using registered phone number for verification")
        recommendations.append("🔒 **Freeze account** temporarily pending investigation")
    elif risk_score > 40:
        recommendations.append("⏸️ **Hold transaction** for manual review")
        recommendations.append("📱 **Send OTP** for additional authentication")
        recommendations.append("📋 **Flag account** for enhanced monitoring")
    else:
        recommendations.append("✅ **Proceed with normal processing**")
        recommendations.append("📊 **Monitor account** for unusual patterns")
    
    # Model-specific recommendations
    fraud_votes = sum(1 for r in model_results.values() if r.get('prediction', 0) == 1)
    
    if fraud_votes >= len(model_results) / 2:
        recommendations.append("🤖 **Multiple models indicate fraud** - escalate immediately")
    
    # Transaction-specific recommendations
    if transaction_data.get('bvn_verified', 1) == 0:
        recommendations.append("🆔 **Require BVN verification** before proceeding")
    
    if transaction_data.get('device_type') == 'USSD':
        recommendations.append("📲 **Suggest mobile app** for higher security")
    
    if transaction_data.get('is_weekend', 0) == 1:
        recommendations.append("📅 **Weekend protocol**: Enhanced verification required")
    
    return recommendations

def calculate_risk_score(risk_factors):
    """
    Calculate total risk score from risk factors
    
    Args:
        risk_factors: List of risk factor dictionaries
        
    Returns:
        int: Total risk score (0-100)
    """
    if not risk_factors:
        return 0
    
    total_score = sum(factor['score'] for factor in risk_factors)
    return min(total_score, 100)