import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Question Answering App", page_icon="🤖")

@st.cache_resource
def load_model():
    return pipeline(task="question-answering", model="timpal0l/mdeberta-v3-base-squad2")

qa_model = load_model()

st.title("🤖 AI Question Answering App")

col1, col2 = st.columns([2, 1])

with col1:
    context = st.text_area("Text Area for context",
                           height=150, placeholder="Enter Context Here...")
    question = st.text_input("Input Box for question",
                             height=150, placeholder="Enter Question Here...")

    submit_btn = st.button("Get Answer", type="primary")

with col2:
    st.markdown("Powered by C-Clarke Institute")

if context and question and submit_btn:
    with st.spinner("Generating Answer..."):
        result = qa_model(question=question, context=context)

        st.success(result['answer'])
        st.metric(result['score'])

else:
    st.markdown("Invalid Input...")
