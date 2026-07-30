from pydantic import BaseModel


class Page(BaseModel):
    """Represents a single page extracted from a PDF."""

    page_number: int
    text: str
    source: str