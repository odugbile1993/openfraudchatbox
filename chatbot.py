import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np
from datetime import datetime

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
    .code-block {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# --- OpenFraudLabs System Prompt ---
OPENFRAUD_SYSTEM_PROMPT = """You are the OpenFraudLabs AI Assistant, an expert in financial fraud detection and credit risk analytics. You provide practical, technical guidance on:

- Machine learning for fraud detection (XGBoost, Random Forests, Neural Networks)
- Real-time transaction monitoring systems
- Credit risk modeling and NPL reduction strategies
- Feature engineering for financial data
- SQL analysis for fraud pattern detection
- Regulatory compliance and risk management

Always provide actionable advice with code examples when relevant. Focus on practical implementations.

Current conversation:
"""

# --- Load model ---
@st.cache_resource(show_spinner=False)
def load_fraud_model():
    try:
        text_generator = pipeline("text-generation", model="openai-community/gpt2")
        text_generator.tokenizer.pad_token = text_generator.tokenizer.eos_token
        return text_generator
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# --- FIXED: Better prompt building ---
def build_fraud_prompt(user_question, chat_history):
    """Build prompt with clear separation between system and user content"""
    prompt = OPENFRAUD_SYSTEM_PROMPT
    
    # Add only the last 2 exchanges to keep context manageable
    for user_msg, ai_msg in chat_history[-2:]:
        prompt += f"User: {user_msg}\n"
        prompt += f"Assistant: {ai_msg}\n"
    
    # Add current question with clear separator
    prompt += f"User: {user_question}\n"
    prompt += f"Assistant:"
    
    return prompt

# --- FIXED: Much better response extraction ---
def extract_correct_response(full_text, original_prompt, user_question):
    """
    Extract only the new assistant response by removing the original prompt
    and finding where the new response starts
    """
    try:
        # Method 1: Simple replacement - remove the original prompt
        response = full_text.replace(original_prompt, "").strip()
        
        # Method 2: If replacement didn't work well, try splitting
        if response == full_text or len(response) > 500:  # If replacement failed or response too long
            if "Assistant:" in full_text:
                # Split by Assistant: and take the last part (most recent response)
                parts = full_text.split("Assistant:")
                if len(parts) > 1:
                    response = parts[-1].strip()
                    
                    # Clean up: remove any subsequent User: parts
                    if "User:" in response:
                        response = response.split("User:")[0].strip()
        
        # Method 3: Final cleanup - ensure we're not repeating the question
        if user_question.lower() in response.lower():
            # Remove the question if it appears in the response
            response = response.replace(user_question, "").strip()
        
        # If response is empty or too similar to prompt, provide fallback
        if not response or len(response) < 10:
            response = f"I understand you're asking about '{user_question}'. As an OpenFraudLabs AI specializing in fraud detection, I can help with machine learning models for fraud classification, feature engineering for financial data, real-time monitoring systems, and risk assessment strategies. Could you be more specific about what aspect you'd like to explore?"
        
        return response
        
    except Exception as e:
        return f"I specialize in fraud detection and risk analytics. For your question about '{user_question}', I recommend focusing on practical implementations like XGBoost for classification, anomaly detection algorithms, or real-time monitoring architectures."

# --- Fraud detection utilities ---
def get_fraud_detection_code(algorithm):
    """Pre-built code snippets for instant responses"""
    code_snippets = {
        "xgboost": """
# XGBoost for Fraud Detection
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Prepare features and target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Handle class imbalance
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=10,  # Adjust based on fraud ratio
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train model
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred))
""",
        "feature_engineering": """
# Feature Engineering for Fraud Detection
def create_fraud_features(transactions_df):
    # Transaction frequency features
    transactions_df['tx_count_1h'] = transactions_df.groupby('user_id')['timestamp'].transform(
        lambda x: x.rolling('1h').count()
    )
    
    # Amount anomalies
    transactions_df['amount_zscore'] = transactions_df.groupby('user_id')['amount'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Time-based features
    transactions_df['hour'] = transactions_df['timestamp'].dt.hour
    transactions_df['is_night'] = (transactions_df['hour'] < 6) | (transactions_df['hour'] > 22)
    
    # Velocity features
    transactions_df['time_since_last_tx'] = transactions_df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
    
    return transactions_df
""",
        "sql_analysis": """
-- SQL for Fraud Pattern Analysis
SELECT 
    user_id,
    COUNT(*) as transaction_count,
    AVG(amount) as avg_amount,
    COUNT(DISTINCT merchant) as unique_merchants,
    COUNT(DISTINCT country) as countries,
    SUM(CASE WHEN amount > 5000 THEN 1 ELSE 0 END) as large_transactions,
    -- Time-based patterns
    TIMESTAMPDIFF(MINUTE, MIN(transaction_time), MAX(transaction_time)) as time_span_minutes
FROM transactions 
WHERE transaction_time >= NOW() - INTERVAL 24 HOUR
GROUP BY user_id
HAVING 
    transaction_count > 20 
    OR countries > 2
    OR large_transactions > 3
    OR (time_span_minutes < 60 AND transaction_count > 5)
ORDER BY transaction_count DESC;
"""
    }
    return code_snippets.get(algorithm, code_snippets["xgboost"])

# --- Check for instant responses first ---
def get_instant_fraud_response(user_question):
    """Provide instant, accurate responses for common fraud detection questions"""
    user_lower = user_question.lower()
    
    if any(word in user_lower for word in ['xgboost', 'xgb', 'boost']):
        return f"""**XGBoost for Fraud Detection:**

XGBoost is excellent for fraud detection due to its handling of class imbalance and feature importance. Here's a practical implementation:

```python
{get_fraud_detection_code('xgboost')}
