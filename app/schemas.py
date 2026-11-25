from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    document: str
    section: str
    chunk_id: str


class AskResponse(BaseModel):
    status: str
    answer: str
    sources: list[Source]