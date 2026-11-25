from fastapi import FastAPI, HTTPException

from app.rag import answer_question
from app.schemas import AskRequest, AskResponse

app = FastAPI(title="Fin RAG Assistant")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        return answer_question(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {exc}") from exc