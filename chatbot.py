# import streamlit as st
# import random
# from datetime import datetime

# # Setup the page
# st.set_page_config(page_title="OpenFraudLabs AI", page_icon="🛡️")

# # Title
# st.title("🛡️ OpenFraudLabs AI Assistant")
# st.write("**Expert in Financial Fraud Detection & Risk Analytics**")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Hello! I'm OpenFraudLabs AI, your expert in fraud detection and risk analytics. How can I help you today?"}
#     ]

# # Pre-built expert knowledge base
# EXPERT_KNOWLEDGE = {
#     "greetings": [
#         "Hello! I'm OpenFraudLabs AI. How can I assist with fraud detection today?",
#         "Hi there! Ready to discuss fraud detection and risk analytics?",
#         "Welcome! I specialize in financial fraud detection. What would you like to know?"
#     ],
    
#     "fraud_detection": [
#         "Fraud detection involves identifying suspicious activities using machine learning, rule-based systems, and behavioral analysis.",
#         "We detect fraud through anomaly detection, pattern recognition, and real-time monitoring systems.",
#         "Key fraud detection methods include: transaction monitoring, device fingerprinting, behavioral biometrics, and network analysis."
#     ],
    
#     "xgboost": [
#         "XGBoost is excellent for fraud detection due to its handling of class imbalance. Use `scale_pos_weight` parameter to adjust for fraud prevalence.",
#         "For fraud detection with XGBoost: focus on precision-recall metrics, not accuracy. Handle class imbalance with proper weighting.",
#         """XGBoost Implementation:
# ```python
# model = xgb.XGBClassifier(
#     n_estimators=150,
#     max_depth=8, 
#     learning_rate=0.05,
#     scale_pos_weight=10,  # Adjust based on your fraud ratio
#     subsample=0.8,
#     random_state=42
# )
# ```"""
#     ],
    
#     "feature_engineering": [
#         "Key features for fraud detection: transaction frequency, amount anomalies, geographic patterns, time-based features, and behavioral deviations.",
#         """Essential fraud features:
# - Transaction count per hour
# - Amount Z-scores  
# - Time since last transaction
# - Geographic velocity
# - Device fingerprint changes""",
#         "Feature engineering should capture behavioral anomalies and statistical outliers from normal user patterns."
#     ],
    
#     "real_time": [
#         "Real-time fraud monitoring requires: streaming architecture (Kafka), fast feature computation, and low-latency model inference.",
#         "For real-time systems: use Redis for feature storage, Kafka for event streaming, and ensure <100ms response times.",
#         "Real-time architecture: event streaming → feature computation → model scoring → decision engine → action"
#     ],
    
#     "sql_analysis": [
#         """SQL for fraud patterns:
# ```sql
# SELECT user_id, COUNT(*) as tx_count, 
#        COUNT(DISTINCT country) as countries
# FROM transactions 
# WHERE created_at >= NOW() - INTERVAL 1 HOUR
# GROUP BY user_id 
# HAVING tx_count > 20 OR countries > 2;
# ```""",
#         "Use SQL window functions to detect rapid succession transactions and geographic impossibilities.",
#         "SQL can identify: velocity patterns, amount testing, new merchant testing, and geographic anomalies."
#     ],
    
#     "risk_management": [
#         "Credit risk management involves: predictive modeling, early warning systems, and collection optimization.",
#         "NPL reduction strategies: better underwriting, behavioral scoring, early intervention, and digital collections.",
#         "Risk metrics to track: NPL ratio, collection effectiveness, recovery rate, and cost per recovery."
#     ],
    
#     "capabilities": [
#         "I can help with: machine learning for fraud, feature engineering, real-time systems, SQL analysis, risk management, and technical implementation.",
#         "My expertise includes: XGBoost models, anomaly detection, system architecture, fraud patterns, and risk analytics.",
#         "I specialize in: fraud detection algorithms, risk modeling, technical implementation, and best practices in financial security."
#     ],
    
#     "fallback": [
#         "I specialize in fraud detection and risk analytics. Could you ask about machine learning, feature engineering, real-time systems, or risk management?",
#         "That's an interesting question! In fraud detection context, we typically focus on anomaly detection, pattern recognition, and risk assessment.",
#         "As a fraud detection expert, I can help you with technical implementation, algorithm selection, or system design for financial security."
#     ]
# }

# def get_ai_response(user_input):
#     """Get immediate, intelligent response without external APIs"""
#     input_lower = user_input.lower()
    
#     # Greetings - exact matches
#     if any(word in input_lower for word in ['hi', 'hello', 'hey', 'greetings']):
#         return random.choice(EXPERT_KNOWLEDGE["greetings"])
    
#     # What can you do?
#     elif any(phrase in input_lower for phrase in ['what can you do', 'capabilities', 'help me', 'what do you do']):
#         return random.choice(EXPERT_KNOWLEDGE["capabilities"])
    
#     # XGBoost - specific terms
#     elif any(term in input_lower for term in ['xgboost', 'xgb', 'boosted trees', 'gradient boosting']):
#         return random.choice(EXPERT_KNOWLEDGE["xgboost"])
    
#     # Feature engineering - specific terms
#     elif any(term in input_lower for term in ['feature engineering', 'create features', 'what features', 'feature selection']):
#         return random.choice(EXPERT_KNOWLEDGE["feature_engineering"])
    
#     # Real-time - specific terms
#     elif any(term in input_lower for term in ['real-time', 'real time', 'live monitoring', 'streaming data']):
#         return random.choice(EXPERT_KNOWLEDGE["real_time"])
    
#     # SQL - specific terms
#     elif any(term in input_lower for term in ['sql query', 'sql analysis', 'database query', 'write sql']):
#         return random.choice(EXPERT_KNOWLEDGE["sql_analysis"])
    
#     # Risk management - specific terms
#     elif any(term in input_lower for term in ['risk management', 'credit risk', 'npl', 'non-performing', 'portfolio risk']):
#         return random.choice(EXPERT_KNOWLEDGE["risk_management"])
    
#     # Fraud detection - only when specifically asked about fraud
#     elif any(phrase in input_lower for phrase in ['what is fraud', 'fraud detection', 'detect fraud', 'how to find fraud']):
#         return random.choice(EXPERT_KNOWLEDGE["fraud_detection"])
    
#     # Thank you
#     elif any(word in input_lower for word in ['thank', 'thanks']):
#         return "You're welcome! Feel free to ask more questions about fraud detection. I'm here to help! 🛡️"
    
#     # Goodbye
#     elif any(word in input_lower for word in ['bye', 'goodbye', 'see you']):
#         return "Goodbye! Stay secure and feel free to return with more fraud detection questions! 🛡️"
    
#     # Fallback for unknown questions - more specific matching
#     else:
#         # Check for general ML terms
#         if any(term in input_lower for term in ['machine learning', 'ml model', 'algorithm']):
#             return "For machine learning in fraud detection, XGBoost and Random Forests work well. Would you like specific implementation details?"
        
#         # Check for data terms
#         elif any(term in input_lower for term in ['data', 'dataset', 'training']):
#             return "For fraud detection data, focus on class imbalance handling and feature engineering. Need guidance on specific data aspects?"
        
#         # Check for technical terms
#         elif any(term in input_lower for term in ['code', 'programming', 'python', 'implement']):
#             return "I can provide code examples for fraud detection in Python, SQL, or system architecture. What specifically would you like to build?"
        
#         # Generic fallback
#         else:
#             return f"I understand you're asking about: '{user_input}'. As a fraud detection expert, I can help with machine learning models, real-time systems, risk management, or technical implementation. What specific area interests you?"

# # Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# # Chat input
# if prompt := st.chat_input("Ask me anything about fraud detection..."):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.write(prompt)
    
#     # Get immediate AI response
#     response = get_ai_response(prompt)
    
#     # Display assistant response
#     with st.chat_message("assistant"):
#         st.write(response)
    
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})

# # Sidebar with quick actions
# with st.sidebar:
#     st.header("🚀 Quick Actions")
    
#     if st.button("🧹 Clear Chat"):
#         st.session_state.messages = [
#             {"role": "assistant", "content": "Hello! I'm OpenFraudLabs AI, your expert in fraud detection and risk analytics. How can I help you today?"}
#         ]
#         st.rerun()
    
#     st.header("💡 Quick Questions")
    
#     quick_questions = {
#         "What is fraud detection?": "fraud_detection",
#         "How to use XGBoost for fraud?": "xgboost", 
#         "Best features for fraud detection": "feature_engineering",
#         "SQL queries for fraud patterns": "sql_analysis",
#         "Real-time monitoring architecture": "real_time",
#         "Credit risk management": "risk_management"
#     }
    
#     for question, category in quick_questions.items():
#         if st.button(question):
#             st.session_state.messages.append({"role": "user", "content": question})
#             response = random.choice(EXPERT_KNOWLEDGE[category])
#             st.session_state.messages.append({"role": "assistant", "content": response})
#             st.rerun()

# # Footer
# st.markdown("---")
# st.markdown("**OpenFraudLabs AI** • Instant Responses • Expert Knowledge • Always Available")






import streamlit as st

def get_bot_response(user_question):
    """
    This function contains the 'knowledge base' from the OpenLabs AI website.
    It performs simple keyword matching to find the best response.
    """
    question = user_question.lower()
    
    # --- Knowledge Base from https://openfraudlabs.github.io/lab-docs/ ---

    if "hello" in question or "hi" in question:
        return "Hello! I am a bot for OpenLabs AI. Ask me about our mission, solutions, or how to contribute."

    if "mission" in question:
        return """
        Our Mission: To democratize financial security through transparent AI.
        
        This includes:
        - 🔓 **Accessible Security:** Making fraud prevention tools available to every African SME.
        - 🌍 **Africa-First AI:** Building models trained on African transaction patterns.
        - 🤝 **Community Defense:** Using collective intelligence to fight financial threats.
        """

    if "solution" in question or "product" in question or "what do you do" in question:
        return """
        We offer several open-source solutions:

        1.  **Real-Time Fraud Detection:** Lightweight ML models (XGBoost, Python, FastAPI) to identify suspicious transactions.
        2.  **Transaction Pattern Library:** A collection of anonymized fraud signatures from across Africa (using SQLite, Pandas, SDV).
        3.  **Financial Literacy Tools:** Educational resources for merchants and consumers (Coming Soon).
        """

    if "fraud detection" in question:
        return "Our Real-Time Fraud Detection solution uses lightweight ML models (like XGBoost) with Python and FastAPI to identify suspicious transactions as they happen. You can find it on GitHub at OpenFraudLabs/fraud-detection-core."

    if "contribute" in question or "join" in question or "help" in question:
        return """
        You can join the 'Fraud Fighters' in several ways:

        1.  **Code Contributors:** Python developers, ML engineers, and DevOps specialists can help by fixing issues.
        2.  **Data Donors:** Share anonymized transaction patterns (no sensitive data).
        3.  **Community Advocates:** Technical writers, translators, and community organizers can join the discussion.
        
        Find out more on our GitHub page!
        """

    if "impact" in question:
        return "Our community-powered security innovation has led to over 50+ GitHub Stars, 15+ Tool Forks, and 42+ Active Contributors from 8+ countries. We also offer a Free Starter Kit."
    
    if "contact" in question or "email" in question or "founder" in question:
        return """
        You can find us at:
        - **GitHub:** https://github.com/OpenFraudLabs
        - **Twitter:** https://twitter.com/OpenFraudLabs
        - **LinkedIn:** https://linkedin.com/company/opnfraudlabs
        
        For security inquiries, email: security@opnfraudlabs.org
        The Founder & Lead Researcher is Ayodele Odugbile.
        """
        
    if "thank" in question or "thanks" in question:
        return "You're welcome! Happy to help."

    # --- Default Response ---
    return "I'm sorry, I don't have information about that. You can ask me about OpenLabs' mission, solutions, impact, or how to contribute."

# --- Streamlit App UI ---

st.set_page_config(page_title="OpenLabs AI Chatbot", page_icon="🤖")
st.title("OpenLabs AI Chatbot")
st.caption("Ask me about the open-source AI fintech for emerging markets")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am a bot for OpenLabs AI. How can I help you today?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about OpenLabs AI..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        response = get_bot_response(prompt)
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
