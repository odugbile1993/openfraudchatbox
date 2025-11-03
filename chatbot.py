import streamlit as st
import random
from datetime import datetime

# Setup the page
st.set_page_config(page_title="OpenFraudLabs AI", page_icon="🛡️")

# Title
st.title("🛡️ OpenFraudLabs AI Assistant")
st.write("**Expert in Financial Fraud Detection & Risk Analytics**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm OpenFraudLabs AI, your expert in fraud detection and risk analytics. How can I help you today?"}
    ]

# Pre-built expert knowledge base
EXPERT_KNOWLEDGE = {
    "greetings": [
        "Hello! I'm OpenFraudLabs AI. How can I assist with fraud detection today?",
        "Hi there! Ready to discuss fraud detection and risk analytics?",
        "Welcome! I specialize in financial fraud detection. What would you like to know?"
    ],
    
    "fraud_detection": [
        "Fraud detection involves identifying suspicious activities using machine learning, rule-based systems, and behavioral analysis.",
        "We detect fraud through anomaly detection, pattern recognition, and real-time monitoring systems.",
        "Key fraud detection methods include: transaction monitoring, device fingerprinting, behavioral biometrics, and network analysis."
    ],
    
    "xgboost": [
        "XGBoost is excellent for fraud detection due to its handling of class imbalance. Use `scale_pos_weight` parameter to adjust for fraud prevalence.",
        "For fraud detection with XGBoost: focus on precision-recall metrics, not accuracy. Handle class imbalance with proper weighting.",
        """XGBoost Implementation:
```python
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=8, 
    learning_rate=0.05,
    scale_pos_weight=10,  # Adjust based on your fraud ratio
    subsample=0.8,
    random_state=42
)
```"""
    ],
    
    "feature_engineering": [
        "Key features for fraud detection: transaction frequency, amount anomalies, geographic patterns, time-based features, and behavioral deviations.",
        """Essential fraud features:
- Transaction count per hour
- Amount Z-scores  
- Time since last transaction
- Geographic velocity
- Device fingerprint changes""",
        "Feature engineering should capture behavioral anomalies and statistical outliers from normal user patterns."
    ],
    
    "real_time": [
        "Real-time fraud monitoring requires: streaming architecture (Kafka), fast feature computation, and low-latency model inference.",
        "For real-time systems: use Redis for feature storage, Kafka for event streaming, and ensure <100ms response times.",
        "Real-time architecture: event streaming → feature computation → model scoring → decision engine → action"
    ],
    
    "sql_analysis": [
        """SQL for fraud patterns:
```sql
SELECT user_id, COUNT(*) as tx_count, 
       COUNT(DISTINCT country) as countries
FROM transactions 
WHERE created_at >= NOW() - INTERVAL 1 HOUR
GROUP BY user_id 
HAVING tx_count > 20 OR countries > 2;
```""",
        "Use SQL window functions to detect rapid succession transactions and geographic impossibilities.",
        "SQL can identify: velocity patterns, amount testing, new merchant testing, and geographic anomalies."
    ],
    
    "risk_management": [
        "Credit risk management involves: predictive modeling, early warning systems, and collection optimization.",
        "NPL reduction strategies: better underwriting, behavioral scoring, early intervention, and digital collections.",
        "Risk metrics to track: NPL ratio, collection effectiveness, recovery rate, and cost per recovery."
    ],
    
    "capabilities": [
        "I can help with: machine learning for fraud, feature engineering, real-time systems, SQL analysis, risk management, and technical implementation.",
        "My expertise includes: XGBoost models, anomaly detection, system architecture, fraud patterns, and risk analytics.",
        "I specialize in: fraud detection algorithms, risk modeling, technical implementation, and best practices in financial security."
    ],
    
    "fallback": [
        "I specialize in fraud detection and risk analytics. Could you ask about machine learning, feature engineering, real-time systems, or risk management?",
        "That's an interesting question! In fraud detection context, we typically focus on anomaly detection, pattern recognition, and risk assessment.",
        "As a fraud detection expert, I can help you with technical implementation, algorithm selection, or system design for financial security."
    ]
}

def get_ai_response(user_input):
    """Get immediate, intelligent response without external APIs"""
    input_lower = user_input.lower()
    
    # Greetings
    if any(word in input_lower for word in ['hi', 'hello', 'hey', 'greetings']):
        return random.choice(EXPERT_KNOWLEDGE["greetings"])
    
    # What can you do?
    elif any(word in input_lower for word in ['what can you do', 'capabilities', 'help']):
        return random.choice(EXPERT_KNOWLEDGE["capabilities"])
    
    # Fraud detection topics
    elif any(word in input_lower for word in ['fraud', 'detection']):
        return random.choice(EXPERT_KNOWLEDGE["fraud_detection"])
    
    # XGBoost
    elif any(word in input_lower for word in ['xgboost', 'machine learning', 'model']):
        return random.choice(EXPERT_KNOWLEDGE["xgboost"])
    
    # Feature engineering
    elif any(word in input_lower for word in ['feature', 'engineering', 'variable']):
        return random.choice(EXPERT_KNOWLEDGE["feature_engineering"])
    
    # Real-time
    elif any(word in input_lower for word in ['real-time', 'monitoring', 'streaming']):
        return random.choice(EXPERT_KNOWLEDGE["real_time"])
    
    # SQL
    elif any(word in input_lower for word in ['sql', 'query', 'database']):
        return random.choice(EXPERT_KNOWLEDGE["sql_analysis"])
    
    # Risk management
    elif any(word in input_lower for word in ['risk', 'management', 'npl', 'credit']):
        return random.choice(EXPERT_KNOWLEDGE["risk_management"])
    
    # Thank you
    elif any(word in input_lower for word in ['thank', 'thanks']):
        return "You're welcome! Feel free to ask more questions about fraud detection. I'm here to help! 🛡️"
    
    # Goodbye
    elif any(word in input_lower for word in ['bye', 'goodbye']):
        return "Goodbye! Stay secure and feel free to return with more fraud detection questions! 🛡️"
    
    # Fallback for unknown questions
    else:
        return random.choice(EXPERT_KNOWLEDGE["fallback"])

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about fraud detection..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get immediate AI response
    response = get_ai_response(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with quick actions
with st.sidebar:
    st.header("🚀 Quick Actions")
    
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm OpenFraudLabs AI, your expert in fraud detection and risk analytics. How can I help you today?"}
        ]
        st.rerun()
    
    st.header("💡 Quick Questions")
    
    quick_questions = [
        "What is fraud detection?",
        "How to use XGBoost for fraud?",
        "Best features for fraud detection",
        "SQL queries for fraud patterns",
        "Real-time monitoring architecture",
        "Credit risk management strategies"
    ]
    
    for question in quick_questions:
        if st.button(question):
            st.session_state.messages.append({"role": "user", "content": question})
            response = get_ai_response(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# Footer
st.markdown("---")
st.markdown("**OpenFraudLabs AI** • Instant Responses • Expert Knowledge • Always Available")
