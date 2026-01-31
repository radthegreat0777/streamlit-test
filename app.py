import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="AI Question Answering App",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        task="question-answering",
        model="timpal0l/mdeberta-v3-base-squad2"
    )

qa_model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🤖 AI Question Answering App")

col1, col2 = st.columns([2, 1])

with col1:
    context = st.text_area(
        "Context",
        height=180,
        placeholder="Enter Context here...Include all relevant information needed to answer the question."
    )

    question = st.text_input(
        "Question",
        placeholder="Enter your Question here...Make sure it relates to the provided context."
    )

    submit_btn = st.button("Get Answer", type="primary")

with col2:
    st.markdown("### ℹ️ About")
    st.markdown("Powered by **RadTech Systems**\n\n"
                "This AI-powered app answers questions based on a given text context using a transformer-based language model.")

# -----------------------------
# Logic
# -----------------------------
if submit_btn:
    if not context or not question:
        st.warning("Please provide both context and question.")
    else:
        with st.spinner("Generating answer..."):
            result = qa_model(
                question=question,
                context=context
            )

        st.success(result["answer"])
        st.metric(
            label="Confidence Score",
            value=f"{result['score']:.4f}"
        )

