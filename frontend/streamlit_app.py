import streamlit as st


st.set_page_config(
    page_title="Notes Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Notes Assistant")
st.caption("Upload a PDF and ask questions about its contents.")

st.divider()

st.subheader("Upload Document")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)

index_button = st.button(
    "Index Document",
    disabled=uploaded_file is None,
)

st.divider()

st.subheader("Ask a Question")

question = st.text_input(
    "Question",
    placeholder="Index a document to start asking questions.",
    disabled=True,
)

ask_button = st.button(
    "Ask",
    disabled=True,
)