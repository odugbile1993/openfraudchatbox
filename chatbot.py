import streamlit as st
from transformers import pipeline

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

# --- Load model ---
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = pipeline("text-generation", model="microsoft/DialoGPT-small")
        model.tokenizer.pad_token = model.tokenizer.eos_token
        return model
    except Exception as e:
        return None

# --- Header ---
st.markdown('<div class="main-header">🛡️ OpenFraudLabs AI Assistant</div>', unsafe_allow_html=True)
st.caption("Your Expert in Financial Fraud Detection & Risk Analytics")

# --- Sidebar with Educational Content ---
with st.sidebar:
    st.header("📚 Learn About Fraud Detection")
    
    st.subheader("Quick Guides")
    with st.expander("XGBoost for Fraud"):
        st.write("""
        **Best for:** Transaction fraud detection
        ```python
        model = xgb.XGBClassifier(
            scale_pos_weight=10,  # Handle imbalance
            n_estimators=150
        )
        ```
        """)
    
    with st.expander("Feature Engineering"):
        st.write("""
        **Key Features:**
        - Transaction frequency
        - Amount anomalies  
        - Geographic patterns
        - Time-based features
        """)
    
    with st.expander("Real-time Monitoring"):
        st.write("""
        **Tech Stack:**
        - Kafka (streaming)
        - Redis (features)
        - ML models (scoring)
        - Rules engine
        """)
    
    with st.expander("SQL Patterns"):
        st.write("""
        ```sql
        SELECT user_id, COUNT(*) as tx_count
        FROM transactions
        WHERE time > NOW() - INTERVAL 1 HOUR
        GROUP BY user_id
        HAVING tx_count > 20;
        ```
        """)
    
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "🛡️ Welcome to OpenFraudLabs AI! I specialize in financial fraud detection, machine learning, and risk analytics. How can I help you today?"}
    ]

# --- Display chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input ---
user_input = st.chat_input("Ask me anything about fraud detection...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            model = load_model()
            
            if model:
                try:
                    # Build conversation context
                    conversation = ""
                    for msg in st.session_state.messages[-4:]:  # Last 4 messages for context
                        role = "User" if msg["role"] == "user" else "Assistant"
                        conversation += f"{role}: {msg['content']}\n"
                    
                    prompt = f"""You are OpenFraudLabs AI Assistant, an expert in financial fraud detection, machine learning, and risk analytics.

Specialties:
- Fraud detection algorithms (XGBoost, Random Forest)
- Real-time transaction monitoring
- Feature engineering for financial data
- Credit risk modeling and NPL reduction
- SQL analysis for fraud patterns

Provide helpful, technical responses. If you don't know something, suggest related fraud detection topics.

{conversation}
User: {user_input}
Assistant:"""
                    
                    # Generate response
                    result = model(
                        prompt,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1
                    )
                    
                    response = result[0]['generated_text'].split("Assistant:")[-1].strip()
                    
                    # Clean up response
                    if "User:" in response:
                        response = response.split("User:")[0].strip()
                    
                except Exception as e:
                    response = "I specialize in fraud detection and risk analytics. Feel free to ask me about machine learning models, feature engineering, real-time monitoring, or any fraud-related topics!"
            else:
                # Fallback responses
                user_lower = user_input.lower()
                
                if any(word in user_lower for word in ['hi', 'hello', 'hey', 'greetings']):
                    response = "🛡️ Hello! I'm OpenFraudLabs AI, your expert in financial fraud detection and risk analytics. How can I assist you today?"
                
                elif any(word in user_lower for word in ['what can you do', 'help', 'capabilities']):
                    response = """**I can help you with:**

🔍 **Fraud Detection**
- Machine learning models (XGBoost, Random Forest)
- Real-time monitoring systems
- Anomaly detection algorithms

📊 **Risk Analytics**  
- Credit risk modeling
- NPL reduction strategies
- Portfolio risk analysis

💻 **Technical Implementation**
- Feature engineering
- SQL pattern analysis
- System architecture

What would you like to explore?"""
                
                elif any(word in user_lower for word in ['thank', 'thanks']):
                    response = "You're welcome! Feel free to ask more questions about fraud detection or risk analytics. 🛡️"
                
                elif any(word in user_lower for word in ['bye', 'goodbye']):
                    response = "Goodbye! Remember, OpenFraudLabs is here to help with all your fraud detection needs. Stay secure! 🛡️"
                
                else:
                    response = f"Thanks for your question about '{user_input}'. As OpenFraudLabs AI, I specialize in fraud detection and risk analytics. Could you tell me more about what specific area interests you? For example:\n\n- Machine learning for fraud detection\n- Real-time monitoring systems\n- Feature engineering\n- Risk management strategies"
        
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Footer ---
st.markdown("---")
st.markdown("**OpenFraudLabs AI** • Intelligent Fraud Detection • Risk Analytics")
