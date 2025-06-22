import streamlit as st
from rag_chatbot import rag_chatbot

st.set_page_config(page_title="RAG Chatbot with Ollama", layout="wide")

st.title("💬 RAG Chatbot (Ollama + Local Embeddings)")
st.write("Ask any question based on your custom dataset!")

# Session state to hold chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", "")

if st.button("Send") and user_input.strip() != "":
    with st.spinner("Generating response..."):
        response, elapsed_time = rag_chatbot(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
        st.success(f"Response time: {elapsed_time:.2f} seconds")

# Display chat history
for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**{speaker}:** {message}")
    else:
        st.markdown(f"**{speaker}:** {message}")
        st.markdown("---")
