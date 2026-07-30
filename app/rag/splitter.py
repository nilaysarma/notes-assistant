from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.chunk import Chunk
from app.models.page import Page

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


def _create_splitter() -> RecursiveCharacterTextSplitter:
    """Create and configure the text splitter."""

    return RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )


def _generate_chunk_id(
    source: str,
    page_number: int,
    chunk_number: int,
) -> str:
    """Generate a human-readable chunk identifier."""

    document_name = Path(source).stem

    return f"{document_name}_p{page_number}_c{chunk_number}"


def _create_chunk(
    *,
    text: str,
    source: str,
    page_number: int,
    chunk_number: int,
    start_char: int,
    end_char: int,
) -> Chunk:
    """Create a Chunk model."""

    return Chunk(
        chunk_id=_generate_chunk_id(
            source=source,
            page_number=page_number,
            chunk_number=chunk_number,
        ),
        text=text,
        source=source,
        page_number=page_number,
        start_char=start_char,
        end_char=end_char,
    )


def split_pages(pages: list[Page]) -> list[Chunk]:
    """Split pages into overlapping chunks."""

    splitter = _create_splitter()

    chunks: list[Chunk] = []

    for page in pages:
        texts = splitter.split_text(page.text)

        search_from = 0

        for chunk_number, text in enumerate(texts):
            start = page.text.find(text, search_from)

            if start == -1:
                raise ValueError(
                    "Failed to locate chunk within the source page."
                )

            end = start + len(text)

            chunk = _create_chunk(
                text=text,
                source=page.source,
                page_number=page.page_number,
                chunk_number=chunk_number,
                start_char=start,
                end_char=end,
            )

            chunks.append(chunk)

            search_from = start + 1

    return chunks
