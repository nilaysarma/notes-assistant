from pathlib import Path
from unittest.mock import Mock

import pytest

from app.models.answer_result import AnswerResult
from app.models.chunk import Chunk
from app.models.indexing_result import IndexingResult
from app.models.page import Page
from app.services.rag_service import (
    NO_CONTEXT_RESPONSE,
    answer_question,
    index_document,
)


def _create_page() -> Page:
    """Create a sample page."""

    return Page(
        page_number=1,
        text="Sample page text.",
        source="sample.pdf",
    )


def _create_chunk() -> Chunk:
    """Create a sample chunk."""

    text = "Sample chunk."

    return Chunk(
        chunk_id="chunk_1",
        text=text,
        source="sample.pdf",
        page_number=1,
        start_char=0,
        end_char=len(text),
    )


def test_index_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test indexing a document."""

    pdf_path = Path("sample.pdf")

    pages = [_create_page()]
    chunks = [_create_chunk()]
    embeddings = [[0.1] * 384]

    load_pdf = Mock(return_value=pages)
    split_pages = Mock(return_value=chunks)
    embed_chunks = Mock(return_value=embeddings)
    add_chunks = Mock()

    monkeypatch.setattr(
        "app.services.rag_service.load_pdf",
        load_pdf,
    )

    monkeypatch.setattr(
        "app.services.rag_service.split_pages",
        split_pages,
    )

    monkeypatch.setattr(
        "app.services.rag_service.embed_chunks",
        embed_chunks,
    )

    monkeypatch.setattr(
        "app.services.rag_service.add_chunks",
        add_chunks,
    )

    result = index_document(
        pdf_path,
    )

    assert isinstance(
        result,
        IndexingResult,
    )

    assert result.page_count == 1
    assert result.chunk_count == 1

    load_pdf.assert_called_once_with(
        pdf_path,
    )

    split_pages.assert_called_once_with(
        pages,
    )

    embed_chunks.assert_called_once_with(
        chunks,
    )

    add_chunks.assert_called_once_with(
        chunks,
        embeddings,
    )


def test_answer_question(
    monkeypatch,
) -> None:
    """Test answering a question."""

    question = "What is AI?"

    embedding = [0.1] * 384

    chunks = [_create_chunk()]

    prompt = "Prompt"

    answer = "Artificial intelligence..."

    embed_query = Mock(
        return_value=embedding,
    )

    search = Mock(
        return_value=chunks,
    )

    build_prompt = Mock(
        return_value=prompt,
    )

    generate = Mock(
        return_value=answer,
    )

    monkeypatch.setattr(
        "app.services.rag_service.embed_query",
        embed_query,
    )

    monkeypatch.setattr(
        "app.services.rag_service.search",
        search,
    )

    monkeypatch.setattr(
        "app.services.rag_service.build_prompt",
        build_prompt,
    )

    monkeypatch.setattr(
        "app.services.rag_service.generate",
        generate,
    )

    result = answer_question(
        question,
    )

    assert isinstance(
        result,
        AnswerResult,
    )

    assert result.answer == answer

    assert result.chunks == chunks

    embed_query.assert_called_once_with(
        question,
    )

    search.assert_called_once_with(
        embedding,
    )

    build_prompt.assert_called_once_with(
        question,
        chunks,
    )

    generate.assert_called_once_with(
        prompt,
    )


def test_answer_question_without_context(
    monkeypatch,
) -> None:
    """Test answering a question when no context is found."""

    question = "What is AI?"

    embedding = [0.1] * 384

    embed_query = Mock(
        return_value=embedding,
    )

    search = Mock(
        return_value=[],
    )

    build_prompt = Mock()

    generate = Mock()

    monkeypatch.setattr(
        "app.services.rag_service.embed_query",
        embed_query,
    )

    monkeypatch.setattr(
        "app.services.rag_service.search",
        search,
    )

    monkeypatch.setattr(
        "app.services.rag_service.build_prompt",
        build_prompt,
    )

    monkeypatch.setattr(
        "app.services.rag_service.generate",
        generate,
    )

    result = answer_question(
        question,
    )

    assert isinstance(
        result,
        AnswerResult,
    )

    assert result.answer == NO_CONTEXT_RESPONSE

    assert result.chunks == []

    embed_query.assert_called_once_with(
        question,
    )

    search.assert_called_once_with(
        embedding,
    )

    build_prompt.assert_not_called()

    generate.assert_not_called()