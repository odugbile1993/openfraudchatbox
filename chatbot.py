import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np

# --- Streamlit page setup ---
st.set_page_config(
    page_title="OpenFraudLabs AI Assistant", 
    page_icon="🛡️", 
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .fraud-alert {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- Load model ---
@st.cache_resource(show_spinner=False)
def load_model():
    """Load a model that works reliably on Streamlit Cloud"""
    try:
        # Use DialoGPT-small - faster and more reliable than GPT-2
        model = pipeline("text-generation", model="microsoft/DialoGPT-small")
        model.tokenizer.pad_token = model.tokenizer.eos_token
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

# --- Pre-built expert responses (GUARANTEED quality) ---
def get_expert_fraud_response(user_question):
    """Comprehensive pre-built responses for fraud detection topics"""
    user_lower = user_question.lower()
    
    # XGBoost and Machine Learning
    if any(word in user_lower for word in ['xgboost', 'machine learning', 'model', 'algorithm']):
        xgboost_code = """
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve

# Handle class imbalance
fraud_ratio = len(y_train[y_train==1]) / len(y_train[y_train==0])
scale_pos_weight = 1 / fraud_ratio

model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
"""
        return f"""**XGBoost for Fraud Detection**

```python
{xgboost_code}
