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
        return """**XGBoost for Fraud Detection**

```python
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
    scale_pos_weight=scale_pos_weight,  # Critical for fraud detection
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Use stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Essential fraud detection features
def create_fraud_features(df):
    # Transaction frequency
    df['tx_count_1h'] = df.groupby('user_id')['timestamp'].transform(
        lambda x: x.rolling('1h').count()
    )
    df['tx_count_24h'] = df.groupby('user_id')['timestamp'].transform(
        lambda x: x.rolling('24h').count()
    )
    
    # Amount patterns
    df['amount_to_avg'] = df['amount'] / df.groupby('user_id')['amount'].transform('mean')
    df['amount_zscore'] = df.groupby('user_id')['amount'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['is_night'] = (df['hour'] < 6) | (df['hour'] > 22)
    df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
    
    # Behavioral features
    df['time_since_last_tx'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
    
    return df

# Real-time scoring pipeline
from kafka import KafkaConsumer
import joblib
import redis

model = joblib.load('fraud_model.pkl')
r = redis.Redis(host='localhost', port=6379)

def score_transaction(transaction):
    # Get user features
    user_features = r.hgetall(f"user:{transaction['user_id']}")
    
    # Calculate real-time features
    features = {
        'amount': transaction['amount'],
        'tx_count_1h': get_1h_count(transaction['user_id']),
        'amount_to_avg': transaction['amount'] / float(user_features.get('avg_amount', 1)),
        'is_night': is_night_time(transaction['timestamp']),
        'new_device': transaction['device_id'] != user_features.get('last_device')
    }
    
    risk_score = model.predict_proba([list(features.values())])[0][1]
    
    # Risk-based actions
    if risk_score > 0.9:
        block_transaction(transaction)
    elif risk_score > 0.7:
        trigger_2fa(transaction)
    
    return risk_score

-- High-risk transaction patterns
SELECT 
    user_id,
    COUNT(*) as tx_count_24h,
    AVG(amount) as avg_amount,
    COUNT(DISTINCT merchant) as unique_merchants,
    COUNT(DISTINCT country) as countries,
    SUM(CASE WHEN amount > 5000 THEN 1 ELSE 0 END) as large_tx,
    TIMESTAMPDIFF(MINUTE, MIN(transaction_time), MAX(transaction_time)) as time_span
FROM transactions 
WHERE transaction_time >= NOW() - INTERVAL 24 HOUR
GROUP BY user_id
HAVING 
    tx_count_24h > 20 
    OR countries > 2
    OR large_tx > 3
    OR (time_span < 60 AND tx_count_24h > 5)
ORDER BY tx_count_24h DESC;

-- Velocity analysis
SELECT 
    user_id,
    HOUR(transaction_time) as tx_hour,
    COUNT(*) as tx_count,
    AVG(amount) as avg_amount
FROM transactions
WHERE transaction_time >= NOW() - INTERVAL 1 HOUR
GROUP BY user_id, HOUR(transaction_time)
HAVING COUNT(*) > 10;


# Credit risk features
risk_features = [
    'credit_score', 'income', 'employment_length', 
    'debt_to_income', 'payment_history', 'credit_utilization'
]

# Ensemble model for risk scoring
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    random_state=42
)




