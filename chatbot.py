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
        response = "**XGBoost for Fraud Detection**\n\n"
        response += "```python\n"
        response += "import xgboost as xgb\n"
        response += "from sklearn.model_selection import train_test_split\n"
        response += "from sklearn.metrics import classification_report\n\n"
        response += "# Handle class imbalance\n"
        response += "fraud_ratio = len(y_train[y_train==1]) / len(y_train[y_train==0])\n"
        response += "scale_pos_weight = 1 / fraud_ratio\n\n"
        response += "model = xgb.XGBClassifier(\n"
        response += "    n_estimators=150,\n"
        response += "    max_depth=8,\n"
        response += "    learning_rate=0.05,\n"
        response += "    scale_pos_weight=scale_pos_weight,\n"
        response += "    subsample=0.8,\n"
        response += "    colsample_bytree=0.8,\n"
        response += "    random_state=42\n"
        response += ")\n\n"
        response += "X_train, X_test, y_train, y_test = train_test_split(\n"
        response += "    X, y, test_size=0.2, stratify=y, random_state=42\n"
        response += ")\n\n"
        response += "model.fit(X_train, y_train)\n"
        response += "y_pred = model.predict(X_test)\n"
        response += "print(classification_report(y_test, y_pred))\n"
        response += "```\n\n"
        response += "**Key Features for Fraud:**\n"
        response += "- Transaction frequency patterns\n"
        response += "- Amount anomalies (Z-scores)\n"
        response += "- Geographic velocity\n"
        response += "- Time-of-day deviations\n"
        response += "- Behavioral biometrics\n\n"
        response += "**Evaluation Metrics:**\n"
        response += "- Precision-Recall curves (not accuracy)\n"
        response += "- False Positive Rate\n"
        response += "- Fraud detection rate"
        return response

    # Feature Engineering
    elif any(word in user_lower for word in ['feature', 'engineering', 'variable']):
        response = "**Feature Engineering for Fraud Detection**\n\n"
        response += "```python\n"
        response += "def create_fraud_features(df):\n"
        response += "    # Transaction frequency\n"
        response += "    df['tx_count_1h'] = df.groupby('user_id')['timestamp'].transform(\n"
        response += "        lambda x: x.rolling('1h').count()\n"
        response += "    )\n"
        response += "    df['tx_count_24h'] = df.groupby('user_id')['timestamp'].transform(\n"
        response += "        lambda x: x.rolling('24h').count()\n"
        response += "    )\n"
        response += "    \n"
        response += "    # Amount patterns\n"
        response += "    df['amount_to_avg'] = df['amount'] / df.groupby('user_id')['amount'].transform('mean')\n"
        response += "    df['amount_zscore'] = df.groupby('user_id')['amount'].transform(\n"
        response += "        lambda x: (x - x.mean()) / x.std()\n"
        response += "    )\n"
        response += "    \n"
        response += "    # Time-based features\n"
        response += "    df['hour'] = df['timestamp'].dt.hour\n"
        response += "    df['is_night'] = (df['hour'] < 6) | (df['hour'] > 22)\n"
        response += "    df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5\n"
        response += "    \n"
        response += "    # Behavioral features\n"
        response += "    df['time_since_last_tx'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()\n"
        response += "    \n"
        response += "    return df\n"
        response += "```\n\n"
        response += "**Top Feature Categories:**\n"
        response += "1. Transaction frequency & velocity\n"
        response += "2. Amount anomalies & statistical outliers\n"
        response += "3. Time-based patterns & deviations\n"
        response += "4. Geographic inconsistencies\n"
        response += "5. Device & behavioral biometrics"
        return response

    # Real-time Monitoring
    elif any(word in user_lower for word in ['real-time', 'monitoring', 'streaming']):
        response = "**Real-Time Fraud Monitoring Architecture**\n\n"
        response += "**Tech Stack:**\n"
        response += "- Apache Kafka (event streaming)\n"
        response += "- Redis (real-time features)\n"
        response += "- XGBoost/TensorFlow (ML serving)\n"
        response += "- Prometheus (monitoring)\n\n"
        response += "**Implementation:**\n"
        response += "```python\n"
        response += "from kafka import KafkaConsumer\n"
        response += "import joblib\n"
        response += "import redis\n\n"
        response += "model = joblib.load('fraud_model.pkl')\n"
        response += "r = redis.Redis(host='localhost', port=6379)\n\n"
        response += "def score_transaction(transaction):\n"
        response += "    user_features = r.hgetall(f'user:{transaction['user_id']}')\n"
        response += "    \n"
        response += "    features = {\n"
        response += "        'amount': transaction['amount'],\n"
        response += "        'tx_count_1h': get_1h_count(transaction['user_id']),\n"
        response += "        'amount_to_avg': transaction['amount'] / float(user_features.get('avg_amount', 1)),\n"
        response += "        'is_night': is_night_time(transaction['timestamp']),\n"
        response += "        'new_device': transaction['device_id'] != user_features.get('last_device')\n"
        response += "    }\n"
        response += "    \n"
        response += "    risk_score = model.predict_proba([list(features.values())])[0][1]\n"
        response += "    \n"
        response += "    if risk_score > 0.9:\n"
        response += "        block_transaction(transaction)\n"
        response += "    elif risk_score > 0.7:\n"
        response += "        trigger_2fa(transaction)\n"
        response += "    \n"
        response += "    return risk_score\n"
        response += "```\n\n"
        response += "**Key Requirements:**\n"
        response += "- <100ms latency for feature engineering\n"
        response += "- Model performance monitoring\n"
        response += "- Feedback loops for retraining"
        return response

    # SQL Analysis
    elif any(word in user_lower for word in ['sql', 'query', 'database']):
        response = "**SQL for Fraud Pattern Analysis**\n\n"
        response += "```sql\n"
        response += "-- High-risk transaction patterns\n"
        response += "SELECT \n"
        response += "    user_id,\n"
        response += "    COUNT(*) as tx_count_24h,\n"
        response += "    AVG(amount) as avg_amount,\n"
        response += "    COUNT(DISTINCT merchant) as unique_merchants,\n"
        response += "    COUNT(DISTINCT country) as countries,\n"
        response += "    SUM(CASE WHEN amount > 5000 THEN 1 ELSE 0 END) as large_tx,\n"
        response += "    TIMESTAMPDIFF(MINUTE, MIN(transaction_time), MAX(transaction_time)) as time_span\n"
        response += "FROM transactions \n"
        response += "WHERE transaction_time >= NOW() - INTERVAL 24 HOUR\n"
        response += "GROUP BY user_id\n"
        response += "HAVING \n"
        response += "    tx_count_24h > 20 \n"
        response += "    OR countries > 2\n"
        response += "    OR large_tx > 3\n"
        response += "    OR (time_span < 60 AND tx_count_24h > 5)\n"
        response += "ORDER BY tx_count_24h DESC;\n\n"
        response += "-- Velocity analysis\n"
        response += "SELECT \n"
        response += "    user_id,\n"
        response += "    HOUR(transaction_time) as tx_hour,\n"
        response += "    COUNT(*) as tx_count,\n"
        response += "    AVG(amount) as avg_amount\n"
        response += "FROM transactions\n"
        response += "WHERE transaction_time >= NOW() - INTERVAL 1 HOUR\n"
        response += "GROUP BY user_id, HOUR(transaction_time)\n"
        response += "HAVING COUNT(*) > 10;\n"
        response += "```\n\n"
        response += "**Common Fraud Patterns:**\n"
        response += "- Rapid succession transactions\n"
        response += "- Geographic impossibilities\n"
        response += "- Amount testing patterns\n"
        response += "- New merchant testing"
        return response

    # Risk Management
    elif any(word in user_lower for word in ['risk', 'management', 'npl', 'credit']):
        response = "**Credit Risk Management & NPL Reduction**\n\n"
        response += "**Strategies for NPL Reduction:**\n\n"
        response += "1. **Predictive Modeling**\n"
        response += "```python\n"
        response += "risk_features = [\n"
        response += "    'credit_score', 'income', 'employment_length', \n"
        response += "    'debt_to_income', 'payment_history', 'credit_utilization'\n"
        response += "]\n\n"
        response += "from sklearn.ensemble import RandomForestClassifier\n"
        response += "model = RandomForestClassifier(\n"
        response += "    n_estimators=200,\n"
        response += "    max_depth=10,\n"
        response += "    min_samples_split=20,\n"
        response += "    random_state=42\n"
        response += ")\n"
        response += "```\n\n"
        response += "2. **Early Warning Systems**\n"
        response += "- 30-day payment delinquency patterns\n"
        response += "- Behavioral score changes\n"
        response += "- Economic indicator correlations\n\n"
        response += "3. **Collection Optimization**\n"
        response += "- Segmented collection strategies\n"
        response += "- Predictive recovery likelihood\n"
        response += "- Digital engagement channels\n\n"
        response += "**Key Metrics:**\n"
        response += "- NPL Ratio\n"
        response += "- Collection Effectiveness Index\n"
        response += "- Recovery Rate\n"
        response += "- Cost per Recovery"
        return response

    # General fraud detection
    elif any(word in user_lower for word in ['fraud', 'detection']):
        response = "**Comprehensive Fraud Detection Strategy**\n\n"
        response += "**Multi-Layered Approach:**\n\n"
        response += "1. **Rule-Based Systems**\n"
        response += "   - Transaction amount thresholds\n"
        response += "   - Geographic velocity checks\n"
        response += "   - Time pattern analysis\n\n"
        response += "2. **Machine Learning Models**\n"
        response += "   - Supervised: XGBoost, Random Forest\n"
        response += "   - Unsupervised: Isolation Forest, Autoencoders\n"
        response += "   - Ensemble methods for improved accuracy\n\n"
        response += "3. **Behavioral Analysis**\n"
        response += "   - User profiling and baselines\n"
        response += "   - Real-time anomaly detection\n"
        response += "   - Network analysis\n\n"
        response += "**Key Success Factors:**\n"
        response += "- Quality feature engineering\n"
        response += "- Proper handling of class imbalance\n"
        response += "- Continuous model monitoring\n"
        response += "- Feedback loops from fraud analysts\n\n"
        response += "Start with simple rules and gradually incorporate ML models."
        return response

    return None

# --- Simple model-based response for other questions ---
def get_model_response(model, user_question, chat_history):
    """Get response from the loaded model"""
    try:
        # Build prompt
        system_prompt = "You are OpenFraudLabs AI Assistant, expert in fraud detection and financial risk. Provide technical, actionable advice.\n\n"
        
        prompt = system_prompt
        for user_msg, ai_msg in chat_history[-2:]:
            prompt += f"User: {user_msg}\nAssistant: {ai_msg}\n"
        
        prompt += f"User: {user_question}\nAssistant:"
        
        # Generate response
        raw_output = model(
            prompt,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=model.tokenizer.eos_token_id,
        )[0]["generated_text"]
        
        # Extract response
        response = raw_output.replace(prompt, "").strip()
        return response
        
    except Exception as e:
        return "I specialize in fraud detection using machine learning, feature engineering, and real-time monitoring. Could you ask about these specific topics?"

# --- Header ---
st.markdown('<div class="main-header">🛡️ OpenFraudLabs AI Assistant</div>', unsafe_allow_html=True)
st.caption("Specialized in Financial Fraud Detection • Machine Learning • Real-time Analytics")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ OpenFraudLabs Tools")
    
    st.subheader("Quick Topics")
    topics = [
        "XGBoost Implementation", 
        "Feature Engineering", 
        "Real-time Monitoring", 
        "SQL Queries",
        "Risk Management",
        "Fraud Detection Strategy"
    ]
    for topic in topics:
        if st.button(topic):
            st.session_state.quick_question = topic
    
    st.subheader("Settings")
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Load model (cached) ---
model = load_model()

# --- Display chat history ---
for user_msg, ai_msg in st.session_state.chat_history:
    st.chat_message("user").write(user_msg)
    
    # Style technical responses
    if any(keyword in ai_msg for keyword in ['```python', '```sql', '**']):
        st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
        st.chat_message("assistant").write(ai_msg)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.chat_message("assistant").write(ai_msg)

# --- Main chat input ---
user_input = st.chat_input("Ask about fraud detection, machine learning, or risk analytics...")

if "quick_question" in st.session_state:
    user_input = st.session_state.quick_question
    del st.session_state.quick_question

if user_input:
    st.chat_message("user").write(user_input)
    
    # First try expert pre-built responses (GUARANTEED quality)
    expert_response = get_expert_fraud_response(user_input)
    
    if expert_response:
        response = expert_response
    elif model:
        # Use model for other questions
        with st.spinner("🛡️ Analyzing with OpenFraudLabs AI..."):
            response = get_model_response(model, user_input, st.session_state.chat_history)
    else:
        response = "I specialize in fraud detection and risk analytics. Ask me about XGBoost models, feature engineering, real-time monitoring, SQL analysis, or risk management strategies."

    # Display response
    st.chat_message("assistant").write(response)
    st.session_state.chat_history.append((user_input, response))
    
    # Keep only last 10 messages
    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[-10:]
    
    st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown("**OpenFraudLabs AI Assistant** • Enterprise Fraud Detection • Machine Learning Solutions")
