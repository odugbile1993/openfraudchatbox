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
</style>
""", unsafe_allow_html=True)

# --- OpenFraudLabs System Prompt ---
OPENFRAUD_SYSTEM_PROMPT = """You are the OpenFraudLabs AI Assistant, specializing in financial fraud detection and credit risk analytics. Provide expert guidance on fraud detection algorithms, risk models, and financial security.

Response guidelines:
- Provide actionable, technical advice with code examples when relevant
- Focus on practical fraud detection implementations
- Suggest specific algorithms and evaluation metrics
- Structure responses clearly

"""

# --- Load model with better error handling ---
@st.cache_resource(show_spinner=False)
def load_fraud_model():
    try:
        # Use the smaller, faster model that we know works
        text_generator = pipeline("text-generation", model="openai-community/gpt2")
        text_generator.tokenizer.pad_token = text_generator.tokenizer.eos_token
        return text_generator
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# --- SIMPLIFIED: Build conversation prompt ---
def build_fraud_prompt(user_question, chat_history):
    """Simplified prompt building - FIXED"""
    # Start with system instruction
    prompt = OPENFRAUD_SYSTEM_PROMPT
    
    # Add recent conversation history (last 2 exchanges)
    for i, (user_msg, ai_msg) in enumerate(chat_history[-2:]):
        prompt += f"User: {user_msg}\nAssistant: {ai_msg}\n"
    
    # Add current question
    prompt += f"User: {user_question}\nAssistant:"
    
    return prompt

# --- FIXED: Better response extraction ---
def extract_fraud_response(generated_text, user_question):
    """Extract only the assistant's response"""
    try:
        # Split by "Assistant:" and take the last part
        if "Assistant:" in generated_text:
            parts = generated_text.split("Assistant:")
            response = parts[-1].strip()
            
            # Remove any user questions that might appear
            if "User:" in response:
                response = response.split("User:")[0].strip()
                
            return response
        else:
            # Fallback: return the text after system prompt
            return generated_text.replace(OPENFRAUD_SYSTEM_PROMPT, "").strip()
    except Exception as e:
        return f"I need to respond to your question about fraud detection. Please ask about specific fraud detection techniques, risk models, or financial security topics."

# --- Fraud detection utilities ---
def analyze_transaction_risk(amount, frequency, location):
    """Quick risk assessment"""
    risk_score = 0
    if amount > 10000: risk_score += 30
    if frequency > 20: risk_score += 25
    if location == "high_risk": risk_score += 45
    
    if risk_score >= 70: return "HIGH", risk_score
    elif risk_score >= 40: return "MEDIUM", risk_score
    else: return "LOW", risk_score

# --- Header ---
st.markdown('<div class="main-header">🛡️ OpenFraudLabs AI Assistant</div>', unsafe_allow_html=True)
st.caption("Specialized in Financial Fraud Detection • Credit Risk Analytics")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ OpenFraudLabs Tools")
    
    # Quick risk assessment
    st.subheader("Quick Risk Assessment")
    trans_amount = st.number_input("Amount ($)", value=1000, min_value=0)
    trans_freq = st.number_input("Daily Transactions", value=5, min_value=1)
    location = st.selectbox("Location Risk", ["low", "medium", "high_risk"])
    
    if st.button("Assess Risk"):
        risk_level, score = analyze_transaction_risk(trans_amount, trans_freq, location)
        st.write(f"**Risk Level:** {risk_level}")
        st.write(f"**Risk Score:** {score}/100")
    
    # Model settings
    st.subheader("Model Settings")
    max_tokens = st.slider("Response Length", 50, 300, 150)
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Display chat history ---
chat_container = st.container()
with chat_container:
    for user_msg, ai_msg in st.session_state.chat_history:
        st.chat_message("user").write(user_msg)
        
        # Add fraud-specific styling for relevant responses
        if any(keyword in ai_msg.lower() for keyword in ['fraud', 'risk', 'detection', 'xgboost']):
            st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
            st.chat_message("assistant").write(ai_msg)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.chat_message("assistant").write(ai_msg)

# --- Suggested prompts ---
st.write("### 💬 Ask about Fraud Detection")
suggested_prompts = [
    "How to detect credit card fraud with machine learning?",
    "Best features for transaction fraud detection?",
    "XGBoost parameters for fraud classification",
    "Real-time fraud monitoring architecture",
    "SQL queries for fraud pattern analysis"
]

cols = st.columns(2)
for i, prompt in enumerate(suggested_prompts):
    with cols[i % 2]:
        if st.button(prompt, use_container_width=True, key=f"btn_{i}"):
            st.session_state.quick_question = prompt

# --- Main chat input ---
user_input = st.chat_input("Ask about fraud detection, risk models, or financial security...")

# Handle quick prompts
if "quick_question" in st.session_state:
    user_input = st.session_state.quick_question
    del st.session_state.quick_question

if user_input:
    # Show user message immediately
    st.chat_message("user").write(user_input)
    
    with st.spinner("🛡️ Analyzing with OpenFraudLabs AI..."):
        try:
            # Load model
            model = load_fraud_model()
            
            if model is None:
                # Fallback response if model fails
                response = "I'm currently specializing in fraud detection techniques, XGBoost models for risk classification, and real-time monitoring systems. Please ask about these topics."
            else:
                # Build prompt and generate response
                prompt_text = build_fraud_prompt(user_input, st.session_state.chat_history)
                
                # Generate response with conservative settings
                raw_output = model(
                    prompt_text,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=model.tokenizer.eos_token_id,
                )[0]["generated_text"]
                
                # Extract clean response
                response = extract_fraud_response(raw_output, user_input)
                
        except Exception as e:
            # Error fallback
            response = f"I specialize in fraud detection and risk analytics. Ask me about XGBoost for fraud classification, feature engineering, or real-time monitoring systems. Error: {str(e)}"
    
    # Display response
    st.chat_message("assistant").write(response)
    
    # Store in history (limit to last 10 exchanges)
    st.session_state.chat_history.append((user_input, response))
    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[-10:]
    
    # Refresh to show new message
    st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown("**OpenFraudLabs AI Assistant** • Financial Fraud Detection • Credit Risk Analytics")
