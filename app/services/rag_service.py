from pathlib import Path

from app.llm.gemini import generate
from app.models.answer_result import AnswerResult
from app.models.indexing_result import IndexingResult
from app.rag import (
    add_chunks,
    build_prompt,
    embed_chunks,
    embed_query,
    load_pdf,
    search,
    split_pages,
)

NO_CONTEXT_RESPONSE = (
    "I couldn't find any relevant information "
    "in the indexed documents."
)


def index_document(
    pdf_path: Path,
) -> IndexingResult:
    """Index a PDF document."""

    pages = load_pdf(pdf_path)

    chunks = split_pages(
        pages,
    )

    embeddings = embed_chunks(
        chunks,
    )

    add_chunks(
        chunks,
        embeddings,
    )

    return IndexingResult(
        page_count=len(pages),
        chunk_count=len(chunks),
    )


def answer_question(
    question: str,
) -> AnswerResult:
    """Answer a question using RAG."""

    embedding = embed_query(
        question,
    )

    chunks = search(
        embedding,
    )

    if not chunks:
        return AnswerResult(
            answer=NO_CONTEXT_RESPONSE,
            chunks=[],
        )

    prompt = build_prompt(
        question,
        chunks,
    )

    answer = generate(
        prompt,
    )

    return AnswerResult(
        answer=answer,
        chunks=chunks,
    )