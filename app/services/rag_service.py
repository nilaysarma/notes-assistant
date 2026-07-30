from pathlib import Path

from app.rag.embeddings import embed_chunks
from app.rag.loader import load_pdf
from app.rag.splitter import split_pages
from app.rag.vector_store import add_chunks


def index_document(
    pdf_path: Path,
) -> None:
    """Index a PDF into the vector store."""

    pages = load_pdf(pdf_path)

    chunks = split_pages(pages)

    embeddings = embed_chunks(chunks)

    add_chunks(
        chunks,
        embeddings,
    )