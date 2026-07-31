import tempfile
from pathlib import Path

import streamlit as st

from app.services.rag_service import index_document

st.set_page_config(
    page_title="Notes Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Notes Assistant")
st.caption("Upload a PDF and ask questions about its contents.")

if "document_indexed" not in st.session_state:
    st.session_state.document_indexed = False

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

if index_button:
    if uploaded_file is None:
        st.error("Please upload a PDF.")
        st.stop()

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            pdf_path = Path(temp_file.name)

        with st.spinner("Indexing document..."):
            result = index_document(pdf_path)

        st.session_state.document_indexed = True

        st.success("Document indexed successfully!")
        st.write(f"**Pages:** {result.page_count}")
        st.write(f"**Chunks:** {result.chunk_count}")

    except Exception as error:
        st.error(f"Failed to index document.\n\n{error}")

st.divider()

st.subheader("Ask a Question")

question = st.text_input(
    "Question",
    placeholder="Index a document to start asking questions.",
    disabled=not st.session_state.document_indexed,
)

ask_button = st.button(
    "Ask",
    disabled=not st.session_state.document_indexed,
)