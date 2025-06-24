import streamlit as st
from langchain_chatbot.tools.langchain_service import ask_agent

st.set_page_config(page_title="Lex Fridman Podcast Bot", layout="wide")
st.title("🎙️ AI Explains What They Meant")
st.markdown("Ask a question about Lex Fridman podcasts and get topic-aware, emotionally intelligent answers.")

# Input box
user_input = st.text_input("Ask your question", placeholder="e.g., What did Elon Musk mean by AGI alignment?")

# Submit button
if user_input:
    with st.spinner("Analyzing..."):
        response = ask_agent(user_input)
        st.success("✅ Response:")
        st.markdown(response)
