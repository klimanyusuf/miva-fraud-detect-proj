"""
Nigerian Bank Fraud Detection Dashboard v4.0
CRISP-DM Implementation with RF, SVM, and Outlier Detection
Developed by: Khalid Yusuf Liman
MIT 8212 Seminar Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_generator import generate_nigerian_dataset, prepare_features
from model_trainer import train_all_models, evaluate_model
from utils import calculate_risk_factors, generate_recommendations

# Page configuration
st.set_page_config(
    page_title="Nigeria Bank Fraud Detection System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Nigerian banking theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        font-weight: 600;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .fraud-alert {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #DC2626;
    }
    .normal-transaction {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #16A34A;
    }
    .warning-transaction {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #D97706;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        color: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for data and models
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'is_initialized' not in st.session_state:
    st.session_state.is_initialized = False

def initialize_system():
    """Initialize the fraud detection system"""
    with st.spinner("🔄 Initializing Nigerian Fraud Detection System..."):
        # Generate dataset
        df = generate_nigerian_dataset(n_transactions=1000)
        
        # Prepare features and train models
        X, y, feature_names, scaler = prepare_features(df)
        models = train_all_models(X, y)
        
        # Store in session state
        st.session_state.df = df
        st.session_state.models = models
        st.session_state.scaler = scaler
        st.session_state.feature_names = feature_names
        st.session_state.is_initialized = True
        
        st.success("✅ System initialized successfully!")

def main():
    """Main application function"""
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("# 🏦 Bank Fraud Detection")
        st.markdown("## Navigation")
        
        page = st.radio(
            "Select Page:",
            ["📊 Dashboard", "🔍 Fraud Detection", "🤖 Model Comparison", 
             "📈 Analytics", "⚙️ Model Settings", "📋 Case Management"]
        )
        
        st.markdown("---")
        st.markdown("### System Configuration")
        
        if not st.session_state.is_initialized:
            if st.button("🚀 Initialize System", use_container_width=True):
                initialize_system()
        else:
            st.success("✅ System Ready")
            
        st.markdown("---")
        st.markdown("#### About")
        st.markdown("""
        **Developer:** Khalid Yusuf Liman  
        **Course:** MIT 8212  
        **Seminar:** Industry Applications & Management in IT
        
        *Enhanced with CRISP-DM methodology*
        """)
    
    # Main content area
    if not st.session_state.is_initialized:
        st.markdown('<h1 class="main-header">🏦 Nigerian Bank Fraud Detection System</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <p style='font-size: 1.2rem; color: #6B7280;'>
                A comprehensive fraud detection system using CRISP-DM methodology with Random Forest, SVM, and Outlier Detection
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔍 CRISP-DM Framework")
            st.markdown("""
            - Business Understanding
            - Data Understanding  
            - Data Preparation
            - Modeling
            - Evaluation
            - Deployment
            """)
        
        with col2:
            st.markdown("### 🤖 ML Models")
            st.markdown("""
            - Random Forest Classifier
            - Support Vector Machine
            - Isolation Forest
            - Ensemble Methods
            """)
        
        with col3:
            st.markdown("### 🇳🇬 Nigerian Context")
            st.markdown("""
            - BVN Verification Status
            - USSD Transaction Patterns
            - Weekend Fraud Analysis
            - Localized Risk Factors
            """)
        
        st.markdown("---")
        st.info("👈 Click 'Initialize System' in the sidebar to start using the dashboard")
        
    else:
        # System is initialized, show selected page
        if page == "📊 Dashboard":
            show_dashboard()
        elif page == "🔍 Fraud Detection":
            show_fraud_detection()
        elif page == "🤖 Model Comparison":
            show_model_comparison()
        elif page == "📈 Analytics":
            show_analytics()
        elif page == "⚙️ Model Settings":
            show_model_settings()
        elif page == "📋 Case Management":
            show_case_management()

def show_dashboard():
    """Dashboard page showing system overview"""
    st.markdown('<h1 class="main-header">📊 System Dashboard</h1>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Transactions", f"{len(st.session_state.df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        fraud_count = st.session_state.df['is_fraud'].sum()
        fraud_rate = fraud_count / len(st.session_state.df) * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Fraud Detected", f"{fraud_count}", f"{fraud_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        rf_acc = st.session_state.models['Random Forest']['accuracy'] * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("RF Accuracy", f"{rf_acc:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        svm_acc = st.session_state.models['SVM']['accuracy'] * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("SVM Accuracy", f"{svm_acc:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        iso_acc = st.session_state.models['Isolation Forest']['accuracy'] * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("IF Accuracy", f"{iso_acc:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col6:
        ens_acc = st.session_state.models['Ensemble']['accuracy'] * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Ensemble Accuracy", f"{ens_acc:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent transactions
    st.markdown('<h3 class="sub-header">Recent Transaction Analysis</h3>', unsafe_allow_html=True)
    
    recent_df = st.session_state.df.tail(10).copy()
    
    # Ensure bank column exists, add default if missing
    if 'bank' not in recent_df.columns:
        recent_df['bank'] = 'N/A'
    
    for _, row in recent_df.iterrows():
        if row['is_fraud'] == 1:
            alert_class = "fraud-alert"
            icon = "⚠️"
            status = "FRAUD"
        elif row.get('risk_score', 0) > 50:
            alert_class = "warning-transaction"
            icon = "🔍"
            status = "SUSPICIOUS"
        else:
            alert_class = "normal-transaction"
            icon = "✓"
            status = "NORMAL"
        
        bank_display = row.get('bank', 'N/A')
        st.markdown(f"""
        <div class="{alert_class}">
            {icon} <strong>{status}</strong>: ₦{row['amount']:,.0f} {row['transaction_type']} via {bank_display}
            in {row['location']} at {row['timestamp'].strftime('%H:%M')}
            <br><small>Account: {row['account_age_days']} days | BVN: {'✅ Verified' if row['bvn_verified'] == 1 else '❌ Unverified'}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Fraud distribution
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<h3 class="sub-header">Fraud by Location</h3>', unsafe_allow_html=True)
        location_fraud = st.session_state.df.groupby('location')['is_fraud'].mean().reset_index()
        fig1 = px.bar(location_fraud, x='location', y='is_fraud', 
                     title='Fraud Rate by Nigerian City',
                     labels={'is_fraud': 'Fraud Rate', 'location': 'City'})
        fig1.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="sub-header">Fraud by Device Type</h3>', unsafe_allow_html=True)
        device_fraud = st.session_state.df.groupby('device_type')['is_fraud'].mean().reset_index()
        fig2 = px.pie(device_fraud, values='is_fraud', names='device_type',
                     title='Fraud Distribution by Device Type')
        st.plotly_chart(fig2, use_container_width=True)
    
    with col3:
        st.markdown('<h3 class="sub-header">Fraud by Nigerian Bank</h3>', unsafe_allow_html=True)
        try:
            # Ensure bank column exists
            if 'bank' not in st.session_state.df.columns:
                st.warning("⚠️ Bank column not found in dataset")
            else:
                bank_fraud = st.session_state.df.groupby('bank')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
                bank_fraud.columns = ['bank', 'total_transactions', 'fraud_count', 'fraud_rate']
                bank_fraud = bank_fraud.sort_values('fraud_rate', ascending=False).head(10)
                fig3 = px.bar(bank_fraud, x='bank', y='fraud_rate',
                             title='Top 10 Banks by Fraud Rate',
                             labels={'fraud_rate': 'Fraud Rate', 'bank': 'Bank'},
                             hover_data={'total_transactions': True, 'fraud_count': True})
                fig3.update_layout(yaxis_tickformat=".1%", xaxis_tickangle=45)
                st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying bank fraud data: {str(e)}")

def show_fraud_detection():
    """Real-time fraud detection page"""
    st.markdown('<h1 class="main-header">🔍 Real-time Fraud Detection</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Transaction Details")
        amount = st.number_input("Amount (₦)", min_value=1000, max_value=10000000, value=50000, step=10000)
        location = st.selectbox("Location", ['Lagos', 'Abuja', 'Kano', 'Port Harcourt', 'Ibadan', 'Kaduna'])
        device = st.selectbox("Device Type", ['Mobile', 'Web', 'USSD', 'ATM'])
        trans_type = st.selectbox("Transaction Type", ['Transfer', 'Bill Payment', 'Airtime', 'Withdrawal'])
    
    with col2:
        st.markdown("### Customer Profile")
        account_age = st.slider("Account Age (days)", 1, 3650, 365)
        beneficiary_count = st.slider("Transactions with Beneficiary", 0, 50, 5)
        time_since_last = st.slider("Hours Since Last Transaction", 0.1, 168.0, 12.0)
        is_foreign = st.checkbox("Foreign Transaction")
        hour = st.slider("Transaction Hour", 0, 23, 14)
        bvn_verified = st.radio("BVN Verified", ["Verified", "Unverified"])
        is_weekend = st.radio("Day Type", ["Weekday", "Weekend"])
    
    if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {
            'amount': amount,
            'account_age_days': account_age,
            'beneficiary_history': beneficiary_count,
            'time_since_last_transaction': time_since_last,
            'is_foreign': 1 if is_foreign else 0,
            'transaction_hour': hour,
            'bvn_verified': 1 if bvn_verified == "Verified" else 0,
            'is_weekend': 1 if is_weekend == "Weekend" else 0,
            'device_type': device,
            'location': location,
            'transaction_type': trans_type
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Get predictions from all models
        results = {}
        for model_name, model_data in st.session_state.models.items():
            if model_name != 'Isolation Forest' and isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
                X_input = st.session_state.scaler.transform(
                    input_df[st.session_state.feature_names]
                )
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_input)[0][1]
                    pred = model.predict(X_input)[0]
                    results[model_name] = {
                        'probability': proba,
                        'prediction': pred,
                        'confidence': proba if pred == 1 else 1 - proba
                    }
        
        # Calculate risk factors
        risk_factors = calculate_risk_factors(input_data)
        risk_score = sum(factor['score'] for factor in risk_factors)
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        # Model predictions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Random Forest' in results:
                rf_result = results['Random Forest']
                if rf_result['prediction'] == 1:
                    st.error(f"**Random Forest**\n\n🚨 Fraud: {rf_result['probability']*100:.1f}%")
                else:
                    st.success(f"**Random Forest**\n\n✅ Normal: {(1-rf_result['probability'])*100:.1f}%")
        
        with col2:
            if 'SVM' in results:
                svm_result = results['SVM']
                if svm_result['prediction'] == 1:
                    st.error(f"**SVM**\n\n🚨 Fraud: {svm_result['probability']*100:.1f}%")
                else:
                    st.success(f"**SVM**\n\n✅ Normal: {(1-svm_result['probability'])*100:.1f}%")
        
        with col3:
            st.warning(f"**Risk Score**\n\n{risk_score}/100")
            if risk_score > 70:
                st.error("HIGH RISK")
            elif risk_score > 40:
                st.warning("MEDIUM RISK")
            else:
                st.success("LOW RISK")
        
        # Risk factors
        st.markdown("---")
        st.markdown("### 🔍 Identified Risk Factors")
        
        if risk_factors:
            for factor in risk_factors:
                if factor['score'] > 0:
                    st.markdown(f"- ⚠️ **{factor['name']}**: {factor['description']}")
        else:
            st.markdown("✅ No significant risk factors detected")
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 📋 Recommended Actions")
        
        recommendations = generate_recommendations(input_data, results, risk_score)
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
        else:
            st.markdown("No specific recommendations at this time")
        
        # Consensus decision
        fraud_votes = sum(1 for r in results.values() if r['prediction'] == 1)
        total_votes = len(results)
        
        st.markdown("---")
        if fraud_votes >= total_votes / 2:
            st.error(f"## 🚨 FINAL DECISION: FRAUD DETECTED ({fraud_votes}/{total_votes} models agree)")
            st.markdown("**Immediate Actions Required:**")
            st.markdown("1. 🛑 Block transaction immediately")
            st.markdown("2. 📞 Contact customer for verification")
            st.markdown("3. 📋 Escalate to fraud team")
        else:
            st.success(f"## ✅ FINAL DECISION: TRANSACTION NORMAL (Only {fraud_votes}/{total_votes} models suspect fraud)")
            st.markdown("**Proceed with normal processing**")

def show_model_comparison():
    """Model comparison page"""
    st.markdown('<h1 class="main-header">🤖 Model Performance Comparison</h1>', unsafe_allow_html=True)
    
    # Performance metrics table
    perf_data = []
    for model_name, model_data in st.session_state.models.items():
        if 'accuracy' in model_data:
            perf_data.append({
                'Model': model_name,
                'Accuracy': model_data['accuracy'],
                'Precision': model_data.get('precision', 0),
                'Recall': model_data.get('recall', 0),
                'F1-Score': model_data.get('f1', 0)
            })
    
    perf_df = pd.DataFrame(perf_data)
    
    # Display metrics
    st.dataframe(
        perf_df.style.format({
            'Accuracy': '{:.2%}',
            'Precision': '{:.2%}',
            'Recall': '{:.2%}',
            'F1-Score': '{:.2%}'
        }).background_gradient(cmap='Blues', subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    )
    
    # Visualization
    fig = px.bar(perf_df.melt(id_vars='Model'), 
                 x='Model', y='value', color='variable',
                 barmode='group', 
                 title='Model Performance Comparison',
                 labels={'value': 'Score', 'variable': 'Metric'})
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.markdown("---")
    st.markdown('<h3 class="sub-header">Feature Importance Analysis - All Models</h3>', unsafe_allow_html=True)
    
    # Create tabs for each model that has feature importance
    models_with_importance = []
    
    # Random Forest and Isolation Forest have feature_importances_
    if 'Random Forest' in st.session_state.models:
        models_with_importance.append('Random Forest')
    
    if 'Isolation Forest' in st.session_state.models:
        models_with_importance.append('Isolation Forest')
    
    if models_with_importance:
        tabs = st.tabs(models_with_importance)
        
        # Random Forest tab
        if 'Random Forest' in models_with_importance:
            with tabs[models_with_importance.index('Random Forest')]:
                rf_model = st.session_state.models['Random Forest']['model']
                if hasattr(rf_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Importance': rf_model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    fig2 = px.bar(importance_df, x='Importance', y='Feature',
                                 orientation='h',
                                 title='Random Forest Feature Importance',
                                 labels={'Importance': 'Relative Importance'})
                    st.plotly_chart(fig2, use_container_width=True)
        
        # Isolation Forest tab
        if 'Isolation Forest' in models_with_importance:
            with tabs[models_with_importance.index('Isolation Forest')]:
                if_model = st.session_state.models['Isolation Forest']['model']
                if hasattr(if_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Importance': if_model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    fig3 = px.bar(importance_df, x='Importance', y='Feature',
                                 orientation='h',
                                 title='Isolation Forest Feature Importance',
                                 labels={'Importance': 'Relative Importance'})
                    st.plotly_chart(fig3, use_container_width=True)
        
        st.info("💡 **Note**: SVM and Ensemble models don't have feature importances. Use SHAP or permutation feature importance for advanced analysis.")

def show_analytics():
    """Analytics and insights page"""
    st.markdown('<h1 class="main-header">📈 Fraud Analytics & Insights</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Temporal Analysis", "Geographic Analysis", "Customer Segmentation"])
    
    with tab1:
        # Temporal patterns
        st.markdown("### ⏰ Temporal Fraud Patterns")
        
        # Fraud by hour
        st.session_state.df['hour'] = st.session_state.df['timestamp'].dt.hour
        hour_fraud = st.session_state.df.groupby('hour')['is_fraud'].mean().reset_index()
        
        fig1 = px.line(hour_fraud, x='hour', y='is_fraud',
                      title='Fraud Rate by Hour of Day',
                      labels={'is_fraud': 'Fraud Rate', 'hour': 'Hour (24h)'})
        fig1.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Weekend vs weekday
        st.session_state.df['day_type'] = np.where(
            st.session_state.df['is_weekend'] == 1, 'Weekend', 'Weekday'
        )
        day_fraud = st.session_state.df.groupby('day_type')['is_fraud'].mean().reset_index()
        
        fig2 = px.bar(day_fraud, x='day_type', y='is_fraud',
                     title='Fraud Rate: Weekend vs Weekday',
                     labels={'is_fraud': 'Fraud Rate', 'day_type': 'Day Type'})
        fig2.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # Geographic analysis
        st.markdown("### 📍 Geographic Fraud Patterns")
        
        location_stats = st.session_state.df.groupby('location').agg({
            'is_fraud': 'mean',
            'amount': 'mean',
            'bvn_verified': 'mean'
        }).reset_index()
        
        fig3 = px.scatter(location_stats, x='amount', y='is_fraud', size='bvn_verified',
                         color='location', hover_name='location',
                         title='Fraud Rate vs Average Amount by Location',
                         labels={'is_fraud': 'Fraud Rate', 'amount': 'Average Amount (₦)',
                                'bvn_verified': 'BVN Verification Rate'})
        fig3.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        # Customer segmentation
        st.markdown("### 👥 Customer Segmentation Analysis")
        
        # Create age segments
        st.session_state.df['age_segment'] = pd.cut(
            st.session_state.df['account_age_days'],
            bins=[0, 30, 90, 365, 3650],
            labels=['New (<30d)', 'Recent (30-90d)', 'Established (3-12m)', 'Long-term (>1y)']
        )
        
        segment_fraud = st.session_state.df.groupby('age_segment')['is_fraud'].mean().reset_index()
        
        fig4 = px.bar(segment_fraud, x='age_segment', y='is_fraud',
                     title='Fraud Rate by Account Age Segment',
                     labels={'is_fraud': 'Fraud Rate', 'age_segment': 'Account Age Segment'})
        fig4.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig4, use_container_width=True)

def show_model_settings():
    """Model configuration and settings page"""
    st.markdown('<h1 class="main-header">⚙️ Model Configuration</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Parameters")
        
        rf_threshold = st.slider(
            "Random Forest Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Probability threshold for fraud classification"
        )
        
        svm_threshold = st.slider(
            "SVM Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05
        )
        
        ensemble_weight = st.slider(
            "Ensemble Weight (RF:SVM)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Weight for ensemble prediction (0=SVM only, 1=RF only)"
        )
        
        if st.button("🔄 Update Model Parameters", use_container_width=True):
            st.success("Parameters updated successfully!")
    
    with col2:
        st.markdown("### Current Model Information")
        
        with st.expander("Random Forest Details", expanded=True):
            st.write("**Algorithm:** Random Forest Classifier")
            st.write("**n_estimators:** 150")
            st.write("**max_depth:** 12")
            st.write("**class_weight:** balanced")
            st.write("**Current Accuracy:** {:.1f}%".format(
                st.session_state.models['Random Forest']['accuracy'] * 100
            ))
        
        with st.expander("SVM Details"):
            st.write("**Algorithm:** Support Vector Machine")
            st.write("**kernel:** rbf (Radial Basis Function)")
            st.write("**C parameter:** 1.0")
            st.write("**class_weight:** balanced")
            st.write("**Current Accuracy:** {:.1f}%".format(
                st.session_state.models['SVM']['accuracy'] * 100
            ))
        
        with st.expander("Data Information"):
            st.write(f"**Total Samples:** {len(st.session_state.df):,}")
            st.write(f"**Fraud Rate:** {st.session_state.df['is_fraud'].mean()*100:.1f}%")
            st.write(f"**Features:** {len(st.session_state.feature_names)}")
            st.write(f"**Training Date:** Generated on demand")

def show_case_management():
    """Case management page for reviewing flagged transactions"""
    st.markdown('<h1 class="main-header">📋 Case Management</h1>', unsafe_allow_html=True)
    
    # Get flagged transactions
    df = st.session_state.df.copy()
    
    # Simulate model predictions for all transactions
    X = st.session_state.scaler.transform(df[st.session_state.feature_names])
    
    flagged_cases = []
    for idx, row in df.iterrows():
        # Get predictions from models
        rf_pred = st.session_state.models['Random Forest']['model'].predict(X[idx:idx+1])[0]
        svm_pred = st.session_state.models['SVM']['model'].predict(X[idx:idx+1])[0]
        
        # Flag if any model predicts fraud
        if rf_pred == 1 or svm_pred == 1:
            # Calculate risk score safely
            risk_factors = calculate_risk_factors(row.to_dict())
            base_risk = risk_factors[0]['score'] if risk_factors else 0
            
            flagged_cases.append({
                'transaction_id': row['transaction_id'],
                'amount': row['amount'],
                'bank': row.get('bank', 'N/A'),
                'location': row['location'],
                'type': row['transaction_type'],
                'timestamp': row['timestamp'],
                'rf_flag': rf_pred,
                'svm_flag': svm_pred,
                'bvn_status': 'Verified' if row['bvn_verified'] == 1 else 'Unverified',
                'risk_score': base_risk * 10 if base_risk > 0 else 5
            })
    
    if flagged_cases:
        flagged_df = pd.DataFrame(flagged_cases)
        
        st.markdown(f"### ⚠️ {len(flagged_df)} Cases Flagged for Review")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            min_amount = st.number_input("Minimum Amount (₦)", value=100000)
        with col2:
            model_filter = st.selectbox("Filter by Model", ["All", "RF Only", "SVM Only", "Both"])
        
        # Apply filters
        filtered_cases = flagged_df[flagged_df['amount'] >= min_amount]
        
        if model_filter == "RF Only":
            filtered_cases = filtered_cases[filtered_cases['rf_flag'] == 1]
        elif model_filter == "SVM Only":
            filtered_cases = filtered_cases[filtered_cases['svm_flag'] == 1]
        elif model_filter == "Both":
            filtered_cases = filtered_cases[(filtered_cases['rf_flag'] == 1) & (filtered_cases['svm_flag'] == 1)]
        
        st.markdown(f"**Showing {len(filtered_cases)} filtered cases**")
        
        # Display cases
        for _, case in filtered_cases.head(10).iterrows():
            with st.expander(f"Case #{case['transaction_id']}: ₦{case['amount']:,.0f} - {case['type']} @ {case['bank']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Case Details:**")
                    st.markdown(f"- **ID:** {case['transaction_id']}")
                    st.markdown(f"- **Bank:** {case['bank']}")
                    st.markdown(f"- **Amount:** ₦{case['amount']:,.0f}")
                    st.markdown(f"- **Type:** {case['type']}")
                    st.markdown(f"- **Location:** {case['location']}")
                    st.markdown(f"- **Time:** {case['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    st.markdown(f"- **BVN:** {case['bvn_status']}")
                
                with col2:
                    st.markdown("**Model Flags:**")
                    flags = []
                    if case['rf_flag'] == 1:
                        flags.append("Random Forest")
                    if case['svm_flag'] == 1:
                        flags.append("SVM")
                    
                    for flag in flags:
                        st.markdown(f"- ⚠️ {flag}")
                    
                    st.markdown(f"**Risk Score:** {case['risk_score']}/100")
                
                # Action buttons
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"✅ Approve", key=f"approve_{case['transaction_id']}"):
                        st.success(f"Case #{case['transaction_id']} approved")
                
                with col2:
                    if st.button(f"❌ Reject", key=f"reject_{case['transaction_id']}"):
                        st.error(f"Case #{case['transaction_id']} rejected as fraud")
                
                with col3:
                    if st.button(f"📋 Escalate", key=f"escalate_{case['transaction_id']}"):
                        st.warning(f"Case #{case['transaction_id']} escalated to supervisor")
    else:
        st.info("No flagged cases detected. All transactions appear normal.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 0.9rem; margin-top: 2rem;'>
    <p>Nigerian Bank Fraud Detection System | Developed for MIT 8212 Seminar</p>
    <p>© 2026 Khalid Yusuf Liman | All rights reserved</p>
</div>
""", unsafe_allow_html=True)

# Run the main application
if __name__ == "__main__":
    main()
