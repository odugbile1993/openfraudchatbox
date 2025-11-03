import streamlit as st
from transformers import pipeline
import requests
import json
from datetime import datetime
import pandas as pd

# Setup the page
st.set_page_config(page_title="OpenFraudLabs AI", page_icon="🛡️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .learning-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🛡️ OpenFraudLabs AI Assistant</div>', unsafe_allow_html=True)
st.caption("Advanced Fraud Detection Expert • Continuous Learning • Real-time Research")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {"interests": [], "expertise_level": "beginner"}

# Load AI model
@st.cache_resource
def load_ai_model():
    try:
        model = pipeline("text-generation", model="microsoft/DialoGPT-medium")
        return model
    except:
        return None

# Web search function
def search_web(query):
    """Search for real-time information online"""
    try:
        # Using DuckDuckGo instant answer API
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get('Abstract'):
            return data['Abstract']
        elif data.get('RelatedTopics'):
            return data['RelatedTopics'][0]['Text'] if data['RelatedTopics'] else None
        else:
            return None
    except:
        return None

# Learning and memory functions
def update_user_profile(user_input, ai_response):
    """Learn from conversations and update user profile"""
    # Detect user interests
    fraud_keywords = ['fraud', 'detection', 'xgboost', 'machine learning', 'risk', 'sql', 'feature', 'monitoring']
    user_interests = []
    
    for keyword in fraud_keywords:
        if keyword in user_input.lower():
            user_interests.append(keyword)
    
    # Update profile
    st.session_state.user_profile["interests"] = list(set(st.session_state.user_profile["interests"] + user_interests))
    
    # Store conversation for learning
    st.session_state.conversation_memory.append({
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "ai_response": ai_response,
        "topics": user_interests
    })

def get_contextual_response(user_input, conversation_history):
    """Generate intelligent response based on context and learning"""
    model = load_ai_model()
    
    # Build intelligent prompt with memory
    system_prompt = """You are OpenFraudLabs AI, an expert financial fraud detection assistant with deep knowledge in:

CORE EXPERTISE:
- Machine Learning: XGBoost, Random Forests, Neural Networks for fraud detection
- Real-time Systems: Kafka, Redis, streaming architecture
- Feature Engineering: Transaction patterns, behavioral analytics
- Risk Management: Credit risk, NPL reduction, portfolio analysis
- Technical Implementation: Python, SQL, system design

RESPONSE STYLE:
- Be conversational and engaging like a human expert
- Ask follow-up questions to understand user needs
- Provide detailed, actionable advice
- Admit when you don't know something and suggest research
- Reference previous conversation context
- Break down complex topics clearly

Current conversation context:
"""
    
    # Add recent conversation history
    conversation_context = ""
    for msg in conversation_history[-6:]:  # Last 6 exchanges for context
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_context += f"{role}: {msg['content']}\n"
    
    # Add user profile context
    profile_context = f"User interests: {', '.join(st.session_state.user_profile['interests'])}" if st.session_state.user_profile["interests"] else ""
    
    full_prompt = f"{system_prompt}{conversation_context}{profile_context}\nUser: {user_input}\nAssistant:"
    
    try:
        result = model(
            full_prompt,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1
        )
        response = result[0]['generated_text'].split("Assistant:")[-1].strip()
        return response
    except:
        return None

# Sidebar with learning features
with st.sidebar:
    st.header("🧠 Learning & Research")
    
    with st.expander("🔍 Real-time Search", expanded=False):
        search_query = st.text_input("Search online for:", placeholder="Latest fraud detection techniques...")
        if st.button("Search Web") and search_query:
            with st.spinner("Searching online..."):
                search_result = search_web(search_query)
                if search_result:
                    st.success("Found information:")
                    st.write(search_result)
                else:
                    st.info("No specific results found. Try different keywords.")
    
    with st.expander("📚 Your Learning Profile", expanded=False):
        st.write(f"**Detected Interests:** {', '.join(st.session_state.user_profile['interests']) if st.session_state.user_profile['interests'] else 'None yet'}")
        st.write(f"**Conversation History:** {len(st.session_state.conversation_memory)} exchanges")
        
        if st.button("Clear Learning Memory"):
            st.session_state.conversation_memory = []
            st.session_state.user_profile = {"interests": [], "expertise_level": "beginner"}
            st.rerun()
    
    with st.expander("💡 Quick Learning Topics", expanded=False):
        topics = [
            "XGBoost Hyperparameter Tuning",
            "Real-time Feature Engineering", 
            "Fraud Pattern SQL Queries",
            "Model Evaluation Metrics",
            "Risk-Based Authentication"
        ]
        for topic in topics:
            if st.button(f"📖 {topic}"):
                st.session_state.user_input = f"Explain {topic} in detail with examples"

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Conversation")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if "user_input" in st.session_state:
        user_input = st.session_state.user_input
        del st.session_state.user_input
    else:
        user_input = st.chat_input("Ask me anything about fraud detection...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate intelligent response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing and researching..."):
                # Try contextual AI response first
                ai_response = get_contextual_response(user_input, st.session_state.messages)
                
                # If AI fails or needs enhancement, try web search
                if not ai_response or "don't know" in ai_response.lower() or "uncertain" in ai_response.lower():
                    st.info("🔍 Researching this topic online...")
                    web_info = search_web(user_input)
                    if web_info:
                        ai_response = f"{ai_response}\n\n🔍 **Research Update:** I found some relevant information:\n\n{web_info}\n\nWould you like me to explore any specific aspect of this further?"
                    elif not ai_response:
                        ai_response = "I'm researching this topic. Based on my fraud detection expertise, I can help you with machine learning models, feature engineering, real-time systems, or risk management. Could you specify what aspect interests you most?"
                
                st.write(ai_response)
        
        # Add to conversation and update learning
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        update_user_profile(user_input, ai_response)

with col2:
    st.subheader("🎯 Quick Actions")
    
    if st.button("🧠 Summarize Conversation"):
        if st.session_state.messages:
            summary = f"Conversation Summary:\n- Topics: {', '.join(st.session_state.user_profile['interests'])}\n- Exchanges: {len(st.session_state.messages)//2}\n- Last active: {datetime.now().strftime('%H:%M')}"
            st.info(summary)
        else:
            st.info("No conversation yet")
    
    if st.button("📊 Suggest Learning Path"):
        interests = st.session_state.user_profile['interests']
        if interests:
            path = f"Based on your interest in {', '.join(interests)}, I recommend:\n\n"
            if 'xgboost' in interests or 'machine learning' in interests:
                path += "• Advanced XGBoost for fraud detection\n• Feature engineering techniques\n• Model evaluation metrics\n"
            if 'sql' in interests:
                path += "• Advanced fraud pattern queries\n• Real-time SQL monitoring\n• Performance optimization\n"
            if 'risk' in interests:
                path += "• Credit risk modeling\n• NPL reduction strategies\n• Portfolio analysis\n"
            st.success(path)
        else:
            st.info("Start chatting to get personalized learning suggestions!")
    
    if st.button("🔄 New Topic"):
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**OpenFraudLabs AI** • Continuous Learning • Real-time Research • Expert Fraud Detection")
