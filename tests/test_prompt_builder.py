import pytest

from app.models.chunk import Chunk
from app.rag.prompt_builder import build_prompt


def _create_chunk(
    chunk_id: str,
    text: str,
) -> Chunk:
    """Create a sample chunk."""

    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source="sample.pdf",
        page_number=1,
        start_char=0,
        end_char=len(text),
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for prompt builder tests."""

    return [
        _create_chunk(
            "chunk_1",
            "Machine learning is a subset of artificial intelligence.",
        ),
        _create_chunk(
            "chunk_2",
            "Linear regression is a supervised learning algorithm.",
        ),
    ]


def test_build_prompt_returns_string() -> None:
    prompt = build_prompt(
        question="What is machine learning?",
        chunks=[],
    )

    assert isinstance(prompt, str)


def test_build_prompt_contains_question() -> None:
    question = "What is machine learning?"

    prompt = build_prompt(
        question=question,
        chunks=[],
    )

    assert question in prompt


def test_build_prompt_contains_all_chunks(
    sample_chunks: list[Chunk],
) -> None:
    prompt = build_prompt(
        question="Explain machine learning.",
        chunks=sample_chunks,
    )

    for chunk in sample_chunks:
        assert chunk.text in prompt


def test_build_prompt_with_empty_chunks() -> None:
    prompt = build_prompt(
        question="What is AI?",
        chunks=[],
    )

    assert "Context" in prompt
    assert "Question" in prompt
    assert "Answer" in prompt
    assert "[Chunk" not in prompt


def test_build_prompt_contains_instruction() -> None:
    prompt = build_prompt(
        question="What is AI?",
        chunks=[],
    )

    assert (
        "If the answer cannot be found in the context"
        in prompt
    )


def test_build_prompt_numbers_chunks(
    sample_chunks: list[Chunk],
) -> None:
    prompt = build_prompt(
        question="Explain machine learning.",
        chunks=sample_chunks,
    )

    assert "[Chunk 1]" in prompt
    assert "[Chunk 2]" in prompt