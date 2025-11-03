import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np
from datetime import datetime

# --- Streamlit page setup ---
st.set_page_config(
    page_title="OpenFraudLabs AI Assistant", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for professional styling ---
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
    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-medium { color: #fd7e14; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- OpenFraudLabs System Prompt ---
OPENFRAUD_SYSTEM_PROMPT = """You are the OpenFraudLabs AI Assistant, specializing in financial fraud detection, credit risk analytics, and FinTech security. You provide expert guidance on:

CORE EXPERTISE:
- Machine learning for fraud detection (XGBoost, ensemble methods)
- Credit risk modeling and NPL reduction strategies
- Real-time transaction monitoring systems
- Financial data analysis and anomaly detection
- Regulatory compliance (CBN, AML/CFT guidelines)
- SQL/Python for financial analytics

RESPONSE GUIDELINES:
- Provide actionable, technical advice with code examples when relevant
- Reference real-world financial datasets and scenarios
- Suggest specific algorithms and evaluation metrics
- Consider both technical and business implications
- If uncertain, outline a research approach rather than guessing

Always structure responses clearly and include practical next steps."""

# --- Load the model with better configuration ---
@st.cache_resource(show_spinner=True)
def load_fraud_detection_model():
    try:
        # Using a more capable model for better responses
        text_generator = pipeline(
            "text-generation", 
            model="microsoft/DialoGPT-medium",
            torch_dtype="auto",
            device_map="auto"
        )
        return text_generator
    except:
        # Fallback to GPT-2 if primary model fails
        text_generator = pipeline("text-generation", model="openai-community/gpt2")
        text_generator.tokenizer.pad_token = text_generator.tokenizer.eos_token
        return text_generator

# --- Fraud Detection Utilities ---
def analyze_transaction_pattern(amount, frequency, location):
    """Simple fraud risk assessment"""
    risk_score = 0
    if amount > 10000: risk_score += 30
    if frequency > 50: risk_score += 25
    if location == "high_risk": risk_score += 45
    
    if risk_score >= 70: return "HIGH", risk_score
    elif risk_score >= 40: return "MEDIUM", risk_score
    else: return "LOW", risk_score

def generate_fraud_detection_code(algorithm="xgboost"):
    """Generate code snippets for fraud detection"""
    code_snippets = {
        "xgboost": """
# XGBoost for Fraud Detection
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=10  # Handle class imbalance
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
        """,
        "isolation_forest": """
# Isolation Forest for Anomaly Detection
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    contamination=0.01,  # Expected fraud rate
    random_state=42
)
fraud_predictions = iso_forest.fit_predict(X_transactions)
        """,
        "sql_analysis": """
-- SQL for Fraud Pattern Analysis
SELECT 
    user_id,
    COUNT(*) as transaction_count,
    AVG(amount) as avg_amount,
    SUM(CASE WHEN amount > 10000 THEN 1 ELSE 0 END) as large_transactions
FROM transactions 
WHERE transaction_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY user_id
HAVING COUNT(*) > 50 OR SUM(CASE WHEN amount > 10000 THEN 1 ELSE 0 END) > 3;
        """
    }
    return code_snippets.get(algorithm, code_snippets["xgboost"])

# --- Build enhanced conversation prompt ---
def build_openfraud_prompt(chat_history, user_question):
    formatted_conversation = []
    for previous_question, previous_answer in chat_history[-6:]:  # Keep last 6 exchanges
        formatted_conversation.append(f"User: {previous_question}\nAssistant: {previous_answer}\n")

    formatted_conversation.append(f"User: {user_question}\nAssistant:")
    return OPENFRAUD_SYSTEM_PROMPT + "\n\n" + "\n".join(formatted_conversation)

# --- Header Section ---
st.markdown('<div class="main-header">🛡️ OpenFraudLabs AI Assistant</div>', unsafe_allow_html=True)
st.caption("Specialized in Financial Fraud Detection • Credit Risk Analytics • FinTech Security")

# --- Sidebar with Enhanced Controls ---
with st.sidebar:
    st.header("⚙️ OpenFraudLabs Configuration")
    
    st.subheader("Model Settings")
    max_new_tokens = st.slider("Response Length", 50, 500, 200, 25)
    temperature = st.slider("Creativity", 0.1, 1.0, 0.7, 0.1)
    top_p = st.slider("Focus (Top-p)", 0.1, 1.0, 0.85, 0.05)
    
    st.subheader("🕵️ Fraud Analysis Tools")
    if st.button("Generate XGBoost Code"):
        st.code(generate_fraud_detection_code("xgboost"), language="python")
    
    if st.button("SQL Analysis Query"):
        st.code(generate_fraud_detection_code("sql_analysis"), language="sql")
    
    # Quick risk assessment
    st.subheader("Quick Risk Check")
    trans_amount = st.number_input("Transaction Amount ($)", min_value=0, value=1000)
    trans_freq = st.number_input("Daily Transactions", min_value=1, value=10)
    location_risk = st.selectbox("Location Risk", ["low", "medium", "high_risk"])
    
    if st.button("Assess Risk"):
        risk_level, score = analyze_transaction_pattern(trans_amount, trans_freq, location_risk)
        st.markdown(f"**Risk Level:** <span class='risk-{risk_level.lower()}'>{risk_level}</span>", unsafe_allow_html=True)
        st.markdown(f"**Risk Score:** {score}/100")
    
    st.subheader("Session Tools")
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Display session info
    st.markdown("---")
    st.markdown(f"**Session:** {len(st.session_state.get('chat_history', []))} messages")
    st.markdown(f"**Last update:** {datetime.now().strftime('%H:%M:%S')}")

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Display chat history with improved formatting ---
chat_container = st.container()
with chat_container:
    for i, (user_message, ai_reply) in enumerate(st.session_state.chat_history):
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.markdown("**You:**")
            st.markdown(user_message)
        
        with col2:
            # Add fraud-specific formatting
            if any(keyword in ai_reply.lower() for keyword in ['fraud', 'risk', 'alert', 'suspicious']):
                st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
                st.markdown("**OpenFraudLabs AI:**")
                st.markdown(ai_reply)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("**OpenFraudLabs AI:**")
                st.markdown(ai_reply)
        
        st.markdown("---")

# --- Enhanced input section ---
st.markdown("### 💬 Consult OpenFraudLabs AI")

# Suggested prompts for quick start
suggested_prompts = [
    "How can I improve my XGBoost model for credit card fraud detection?",
    "What are the best features for transaction fraud analysis?",
    "How to handle class imbalance in fraud datasets?",
    "Suggest a real-time monitoring architecture for digital banking",
    "What SQL queries are most useful for fraud pattern analysis?"
]

# Create columns for suggested prompts
cols = st.columns(2)
for i, prompt in enumerate(suggested_prompts):
    with cols[i % 2]:
        if st.button(prompt, key=f"prompt_{i}"):
            st.session_state.quick_prompt = prompt

# Main input
user_input = st.chat_input("Ask about fraud detection, risk models, or FinTech security...")

# Handle quick prompts
if "quick_prompt" in st.session_state:
    user_input = st.session_state.quick_prompt
    del st.session_state.quick_prompt

if user_input:
    # Display user message immediately
    st.chat_message("user").markdown(user_input)
    
    with st.spinner("🛡️ Analyzing with OpenFraudLabs expertise..."):
        try:
            text_generator = load_fraud_detection_model()
            prompt_text = build_openfraud_prompt(st.session_state.chat_history, user_input)

            generation_output = text_generator(
                prompt_text,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                pad_token_id=text_generator.tokenizer.eos_token_id,
                eos_token_id=text_generator.tokenizer.eos_token_id,
            )[0]["generated_text"]

            # Extract the model's answer
            generated_answer = generation_output.split("Assistant:")[-1].strip()
            if "User:" in generated_answer:
                generated_answer = generated_answer.split("User:")[0].strip()

        except Exception as e:
            generated_answer = f"I encountered an error: {str(e)}\n\nPlease try rephrasing your question or check the model configuration."

    # Display and store response
    st.chat_message("assistant").markdown(generated_answer)
    st.session_state.chat_history.append((user_input, generated_answer))
    
    # Auto-scroll to latest message
    st.rerun()

# --- Footer with OpenFraudLabs branding ---
st.markdown("---")
st.markdown(
    "**OpenFraudLabs AI Assistant** • Built for Financial Security Research • "
    "Leveraging Transformers for Fraud Analytics"
)
