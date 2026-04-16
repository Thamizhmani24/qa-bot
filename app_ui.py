import streamlit as st
from transformers import pipeline

# Load model
@st.cache_resource
def load_model():
    return pipeline("question-answering", model="deepset/bert-base-cased-squad2")

qa = load_model()

# UI
st.set_page_config(page_title="QA Bot", page_icon="🤖")

st.title("🤖 AI Question Answering Bot")
st.write("Ask questions based on a given passage")

# Input fields
context = st.text_area("📄 Enter Passage", height=200)
question = st.text_input("❓ Ask a Question")

# Button
if st.button("Get Answer"):
    if context and question:
        result = qa(question=question, context=context)
        
        st.success(f"Answer: {result['answer']}")
        st.info(f"Confidence: {round(result['score'], 2)}")
    else:
        st.warning("Please enter both passage and question")