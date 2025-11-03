import streamlit as st
from transformers import pipeline
import sys

# --- Streamlit page setup ---
st.set_page_config(page_title="GPT-2 Chatbot", page_icon="🤖", layout="centered")

# --- Load the GPT-2 model with better error handling ---
@st.cache_resource(show_spinner=True)
def load_text_generator():
    st.write("🔄 Starting model load...")
    try:
        text_generator = pipeline("text-generation", model="openai-community/gpt2")
        st.write("✅ Model loaded successfully")
        text_generator.tokenizer.pad_token = text_generator.tokenizer.eos_token
        return text_generator
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

# --- System behavior prompt ---
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant for software engineering. "
    "Answer concisely and give short code examples when useful. "
    "If unsure, say you are unsure.\n\n"
)

# --- Build the conversation prompt for GPT-2 ---
def build_conversation_prompt(chat_history, user_question):
    formatted_conversation = []
    for previous_question, previous_answer in chat_history:
        formatted_conversation.append(f"Question: {previous_question}\nAnswer: {previous_answer}\n")

    formatted_conversation.append(f"Question: {user_question}\nAnswer:")
    return SYSTEM_INSTRUCTION + "\n".join(formatted_conversation)

# --- Page header ---
st.title("🤖 GPT-2 Chatbot (Debug Mode)")
st.caption("Testing and debugging the chatbot functionality")

# --- Sidebar for configuration options ---
with st.sidebar:
    st.header("⚙️ Model Settings")
    max_new_tokens = st.slider("Maximum new tokens", 20, 300, 120, 10)
    temperature = st.slider("Creativity (temperature)", 0.1, 1.0, 0.5, 0.1)
    top_p = st.slider("Top-p sampling", 0.1, 1.0, 0.9, 0.05)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.15, 0.05)

    if st.button("🧹 Clear chat history"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

    # Debug info
    st.header("🐛 Debug Info")
    st.write(f"Python version: {sys.version}")
    st.write(f"Transformers available: True")

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.write("✅ Chat history initialized")

# --- Display chat history ---
st.write("📜 Chat History:")
for i, (user_message, ai_reply) in enumerate(st.session_state.chat_history):
    st.chat_message("user").markdown(user_message)
    st.chat_message("assistant").markdown(ai_reply)

# --- Input box for new question ---
user_input = st.chat_input("Ask me about software engineering...")

if user_input:
    st.write(f"🎯 User input received: '{user_input}'")
    st.chat_message("user").markdown(user_input)

    with st.spinner("Thinking..."):
        st.write("🔄 Starting response generation...")
        
        try:
            # Try to load model
            st.write("📥 Loading model...")
            text_generator = load_text_generator()
            
            if text_generator is None:
                st.error("Model failed to load. Using fallback response.")
                fallback_response = "I'm experiencing technical difficulties. Please check the model setup."
                st.chat_message("assistant").markdown(fallback_response)
                st.session_state.chat_history.append((user_input, fallback_response))
            else:
                st.write("✅ Model loaded, building prompt...")
                prompt_text = build_conversation_prompt(st.session_state.chat_history, user_input)
                st.write(f"📝 Prompt length: {len(prompt_text)} characters")
                
                st.write("🎲 Generating response...")
                generation_output = text_generator(
                    prompt_text,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=text_generator.tokenizer.eos_token_id,
                    eos_token_id=text_generator.tokenizer.eos_token_id,
                )[0]["generated_text"]
                
                st.write("✅ Response generated, processing...")
                # Extract the model's answer from the text
                generated_answer = generation_output.split("Answer:")[-1].strip()
                if "Question:" in generated_answer:
                    generated_answer = generated_answer.split("Question:")[0].strip()
                
                st.write(f"📨 Final response: {len(generated_answer)} characters")
                
                # Display and store chatbot response
                st.chat_message("assistant").markdown(generated_answer)
                st.session_state.chat_history.append((user_input, generated_answer))
                st.success("✅ Response completed successfully!")
                
        except Exception as e:
            st.error(f"❌ Error during response generation: {str(e)}")
            error_response = f"I encountered an error: {str(e)}. Please try a simpler question."
            st.chat_message("assistant").markdown(error_response)
            st.session_state.chat_history.append((user_input, error_response))

# --- Final status ---
st.write("---")
st.write("🔍 Debug session active - check the messages above to see where it fails")
