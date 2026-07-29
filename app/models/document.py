from pydantic import BaseModel


class DocumentChunk(BaseModel):
    content: str
    source: str
    page: int
    chunk_id: int