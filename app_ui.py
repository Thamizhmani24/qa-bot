import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="QA Bot", page_icon="🤖")

st.title("🤖 AI Question Answering Bot")

@st.cache_resource
def load_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa = load_model()

context = st.text_area("📄 Enter Passage")
question = st.text_input("❓ Ask a Question")

if st.button("Get Answer"):
    if context and question:
        result = qa(question=question, context=context)
        st.success(f"Answer: {result['answer']}")
        st.info(f"Confidence: {round(result['score'], 2)}")
    else:
        st.warning("Please enter both fields")