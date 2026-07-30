from pathlib import Path

import pytest

from app.models.chunk import Chunk
from app.models.page import Page
from app.rag.loader import load_pdf
from app.rag.splitter import split_pages

TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def loaded_pages() -> list[Page]:
    """Load the sample PDF."""

    return load_pdf(TEST_DATA_DIR / "sample.pdf")


@pytest.fixture
def chunks(loaded_pages: list[Page]) -> list[Chunk]:
    """Split the sample PDF into chunks."""

    return split_pages(loaded_pages)


def test_split_pages_returns_chunks(
    chunks: list[Chunk],
) -> None:
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_chunk_ids_are_unique(
    chunks: list[Chunk],
) -> None:
    ids = [chunk.chunk_id for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_chunk_metadata(
    chunks: list[Chunk],
) -> None:
    first_chunk = chunks[0]

    assert first_chunk.source == "sample.pdf"
    assert first_chunk.page_number == 1


def test_chunk_character_positions(
    chunks: list[Chunk],
) -> None:
    assert all(
        chunk.start_char < chunk.end_char
        for chunk in chunks
    )


def test_chunk_text_matches_original_page(
    loaded_pages: list[Page],
    chunks: list[Chunk],
) -> None:
    page_lookup = {
        page.page_number: page
        for page in loaded_pages
    }

    for chunk in chunks:
        page = page_lookup[chunk.page_number]

        extracted = page.text[
            chunk.start_char : chunk.end_char
        ]

        assert extracted == chunk.text


def test_chunk_text_is_not_empty(
    chunks: list[Chunk],
) -> None:
    assert all(chunk.text.strip() for chunk in chunks)