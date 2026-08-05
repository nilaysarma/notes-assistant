import tempfile
from pathlib import Path

import streamlit as st

from app.rag.vector_store import clear_collection
from app.services.rag_service import (
    answer_question,
    index_document,
)

st.set_page_config(
    page_title="Notes Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Notes Assistant")
st.caption("Upload a PDF and ask questions about its contents.")

if "document_indexed" not in st.session_state:
    st.session_state.document_indexed = False

if "indexing_result" not in st.session_state:
    st.session_state.indexing_result = None

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
    st.session_state.indexing_result = None
    st.session_state.answer_result = None

    if uploaded_file is None:
        st.error("Please upload a PDF.")
        st.stop()

    pdf_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            pdf_path = Path(temp_file.name)

        with st.spinner("Indexing document..."):
            clear_collection()
            result = index_document(pdf_path)

        st.session_state.document_indexed = True
        st.session_state.indexing_result = result

    except Exception as error:
        st.error(f"Failed to index document.\n\n{error}")

    finally:
        if pdf_path is not None and pdf_path.exists():
            pdf_path.unlink()

if st.session_state.indexing_result is not None:
    result = st.session_state.indexing_result

    st.success("Document indexed successfully!")

    st.write(f"**Pages:** {result.page_count}")
    st.write(f"**Chunks:** {result.chunk_count}")

st.divider()

st.subheader("Ask a Question")

if "answer_result" not in st.session_state:
    st.session_state.answer_result = None

with st.form(
    key="question_form",
):
    question = st.text_input(
        "Question",
        placeholder="Index a document to start asking questions.",
        disabled=not st.session_state.document_indexed,
    )

    ask_button = st.form_submit_button(
        "Ask",
        disabled=not st.session_state.document_indexed,
    )

if ask_button:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        with st.spinner("Generating answer..."):
            result = answer_question(question)

        st.session_state.answer_result = result

    except Exception as error:
        st.error(f"Failed to answer question.\n\n{error}")

if st.session_state.answer_result is not None:
    result = st.session_state.answer_result

    st.divider()

    st.subheader("Answer")

    st.markdown(result.answer)

    st.divider()

    st.subheader("Sources")

    for index, chunk in enumerate(
        result.chunks,
        start=1,
    ):
        with st.expander(
            (
                f"📄 Page {chunk.page_number}"
            )
        ):
            st.caption(
                f"Source: {chunk.source}"
            )

            st.text(chunk.text)