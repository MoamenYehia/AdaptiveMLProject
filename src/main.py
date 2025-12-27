"""
Main Streamlit Application for Adaptive Fraud Detection System
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Import custom modules
from config.settings import (
    DATA_PATH, APP_CONFIG, PROJECT_INFO, MODEL_CONFIG, TRAINING_CONFIG,
    FEATURES_TO_DROP, CATEGORICAL_FEATURES, TYPE_CATEGORIES
)
from models.fraud_detector import AdaptiveFraudDetector
from utils.data_loader import (
    load_data, prepare_features, compute_class_weights,
    evaluate_model, get_sample_weights
)


# ==================== PAGE CONFIGURATION ====================
st.set_page_config(**APP_CONFIG)

# Custom CSS Styling
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50; text-align: center;}
    h2, h3 {color: #34495e;}
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    .fraud {color: #e74c3c; font-weight: bold;}
    .safe {color: #27ae60; font-weight: bold;}
    .stButton > button {background-color: #3498db; color: white; border-radius: 8px;}
    .stButton > button:hover {background-color: #2980b9;}
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #3498db;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==================== HEADER SECTION ====================
st.title(PROJECT_INFO["title"])
st.markdown(f"<p style='text-align:center; color:#7f8c8d;'>{PROJECT_INFO['subtitle']}</p>", unsafe_allow_html=True)

# Project Information
with st.expander("📋 Project Information"):
    st.markdown(f"""
    **{PROJECT_INFO['project_name']}**  
    **Team:** {PROJECT_INFO['team']}  
    **Dataset:** {PROJECT_INFO['dataset']}
    """)


# ==================== DATA LOADING & INITIALIZATION ====================
@st.cache_data
def initialize_app_data():
    """
    Load data and prepare features for the application.
    
    Returns:
        tuple: (X, y, X_train, X_test, y_train, y_test, class_weight_dict, type_columns)
    """
    # Load data
    df = load_data(str(DATA_PATH))
    
    # Prepare features
    X, y = prepare_features(df, FEATURES_TO_DROP, CATEGORICAL_FEATURES)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TRAINING_CONFIG["test_size"],
        random_state=TRAINING_CONFIG["random_state"],
        stratify=y if TRAINING_CONFIG["stratify"] else None
    )
    
    # Compute class weights
    class_weight_dict = compute_class_weights(y_train)
    
    # Get type columns
    type_cols = [col for col in X.columns if col.startswith('type_')]
    
    return X, y, X_train, X_test, y_train, y_test, class_weight_dict, type_cols


# Load data
try:
    X, y, X_train, X_test, y_train, y_test, class_weight_dict, type_cols = initialize_app_data()
except FileNotFoundError as e:
    st.error(f"❌ Error: {str(e)}")
    st.stop()

# Display class weights
st.info(
    f"📊 **Balanced Class Weights:** Non-Fraud (0): {class_weight_dict[0]:.4f} | "
    f"Fraud (1): {class_weight_dict[1]:.2f}"
)


# ==================== MODEL INITIALIZATION ====================
if 'model' not in st.session_state:
    with st.spinner("🔧 Training initial model with balanced weighting..."):
        detector = AdaptiveFraudDetector(MODEL_CONFIG)
        detector.train_initial(
            X_train, y_train, class_weight_dict,
            batch_size=TRAINING_CONFIG["batch_size"]
        )
        st.session_state.model = detector
    
    st.success("✅ Initial model trained successfully!")

detector = st.session_state.model


# ==================== CURRENT PERFORMANCE SECTION ====================
st.header("📊 Current Model Performance (Before New Data)")

old_acc, old_recall, old_auc, _, _, old_cm = evaluate_model(
    detector.get_model(), X_test, y_test
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"<div class='metric-card'><h3>Accuracy</h3><h2>{old_acc:.4f}</h2></div>",
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"<div class='metric-card'><h3>Recall (Fraud)</h3><h2>{old_recall:.4f}</h2></div>",
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f"<div class='metric-card'><h3>ROC AUC</h3><h2>{old_auc:.4f}</h2></div>",
        unsafe_allow_html=True
    )


# ==================== NEW TRANSACTION INPUT SECTION ====================
st.markdown("---")
st.header("➕ Add New Transaction & Adapt Model")

col1, col2 = st.columns(2)

with col1:
    step = st.number_input('Step (hour)', min_value=1, value=1)
    amount = st.number_input('Amount', min_value=0.0, value=1000.0)
    oldbalanceOrg = st.number_input('Old Balance Origin', min_value=0.0, value=10000.0)
    newbalanceOrig = st.number_input('New Balance Origin', min_value=0.0, value=9000.0)

with col2:
    oldbalanceDest = st.number_input('Old Balance Destination', min_value=0.0, value=0.0)
    newbalanceDest = st.number_input('New Balance Destination', min_value=0.0, value=1000.0)
    isFlaggedFraud = st.selectbox('Is Flagged Fraud?', [0, 1])
    transaction_type = st.selectbox('Transaction Type', TYPE_CATEGORIES)

# Prepare new transaction data
new_row = {
    'step': step, 'amount': amount, 'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig, 'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest, 'isFlaggedFraud': isFlaggedFraud
}

for col in type_cols:
    new_row[col] = 1 if col == f'type_{transaction_type}' else 0

new_df = pd.DataFrame([new_row])[X.columns]


# ==================== PREDICTION SECTION ====================
st.markdown("---")
if st.button("🔮 Predict Fraud on New Transaction", key="predict_btn", use_container_width=True):
    pred = detector.predict(new_df)[0]
    proba = detector.predict_proba(new_df)[0][1]
    
    st.markdown("### 📈 Prediction Result")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if pred == 1:
            st.markdown(f"<h2 class='fraud'>⚠️ FRAUD DETECTED</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 class='safe'>✅ Transaction is Safe</h2>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<h3>Fraud Probability: <b>{proba:.2%}</b></h3>", unsafe_allow_html=True)


# ==================== MODEL ADAPTATION SECTION ====================
st.markdown("---")
st.subheader("🧠 Update Model with True Label (Adaptive Learning)")

st.markdown(
    "<p style='background-color: #fff3cd; padding: 10px; border-radius: 5px;'>"
    "📌 <b>Help the model learn:</b> After making a prediction, verify if it was correct "
    "and update the model with the true label to improve future predictions."
    "</p>",
    unsafe_allow_html=True
)

true_label = st.radio(
    "What was the **TRUE label** of this transaction?",
    options=[0, 1],
    format_func=lambda x: "✅ Not Fraud" if x == 0 else "⚠️ Fraud",
    horizontal=True
)

if st.button("🔄 Update Model with This New Labeled Data", key="update_btn", use_container_width=True):
    with st.spinner("Updating model adaptively..."):
        sample_weight = np.where(
            np.array([true_label]) == 0,
            class_weight_dict[0],
            class_weight_dict[1]
        )
        
        detector.adapt(new_df, np.array([true_label]), class_weight_dict)
    
    # Evaluate new performance
    new_acc, new_recall, new_auc, _, _, new_cm = evaluate_model(
        detector.get_model(), X_test, y_test
    )
    
    st.success("✅ Model successfully updated with new labeled data!")
    
    # ==================== PERFORMANCE COMPARISON ====================
    st.header("📈 Performance Comparison: Before vs After Adaptation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        acc_diff = new_acc - old_acc
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Accuracy</h3>
            <p>Before: <b>{old_acc:.4f}</b></p>
            <p>After: <b>{new_acc:.4f}</b></p>
            <p style='color:{"#27ae60" if acc_diff >= 0 else "#e74c3c"}; font-size: 16px;'>
                {'📈' if acc_diff >= 0 else '📉'} {abs(acc_diff):+.6f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        recall_diff = new_recall - old_recall
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Recall (Fraud)</h3>
            <p>Before: <b>{old_recall:.4f}</b></p>
            <p>After: <b>{new_recall:.4f}</b></p>
            <p style='color:{"#27ae60" if recall_diff >= 0 else "#e74c3c"}; font-size: 16px;'>
                {'📈' if recall_diff >= 0 else '📉'} {abs(recall_diff):+.6f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        auc_diff = new_auc - old_auc
        st.markdown(f"""
        <div class='metric-card'>
            <h3>ROC AUC</h3>
            <p>Before: <b>{old_auc:.4f}</b></p>
            <p>After: <b>{new_auc:.4f}</b></p>
            <p style='color:{"#27ae60" if auc_diff >= 0 else "#e74c3c"}; font-size: 16px;'>
                {'📈' if auc_diff >= 0 else '📉'} {abs(auc_diff):+.6f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== CONFUSION MATRICES ====================
    st.subheader("📊 Confusion Matrices: Before vs After")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(old_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0], cbar=False)
    axes[0].set_title('Before Update', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    sns.heatmap(new_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False)
    axes[1].set_title('After Update', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    st.pyplot(fig)
    
    st.info(
        "💡 **Tip:** The model now correctly handles class imbalance during online learning! "
        "Add more labeled transactions (especially fraud cases) to see improvements over time."
    )
