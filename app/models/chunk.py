from pydantic import BaseModel


class Chunk(BaseModel):
    """Represents a chunk of text extracted from a document."""

    chunk_id: str
    text: str
    source: str
    page_number: int
    start_char: int
    end_char: int