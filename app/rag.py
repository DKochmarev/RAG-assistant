from app.llm import ask_llm, get_embedding
from app.schemas import AskResponse, Source
from app.store import search_chunks


def answer_question(
    question: str,
    llm_model: str | None = None,
    embedding_model: str | None = None,
) -> AskResponse:
    question_embedding = get_embedding(question, embedding_model)
    chunks = search_chunks(question_embedding, limit=4)
    allowed_ids = {chunk["chunk_id"] for chunk in chunks}

    llm_result = ask_llm(question, chunks, llm_model)

    status = llm_result.get("status", "insufficient_context")
    answer = llm_result.get("answer", "Недостаточно информации.")
    source_ids = llm_result.get("source_chunk_ids", [])

    if not isinstance(source_ids, list):
        source_ids = []

    sources: list[Source] = []
    if status == "answered":
        for chunk in chunks:
            if chunk["chunk_id"] in source_ids and chunk["chunk_id"] in allowed_ids:
                sources.append(
                    Source(
                        document=chunk["document"],
                        section=chunk["section"],
                        chunk_id=chunk["chunk_id"],
                    )
                )

    # Если модель сказала "answered", но реальных источников нет — не доверяем ответу.
    if status != "answered" or not sources:
        return AskResponse(
            status="insufficient_context",
            answer="В предоставленных документах недостаточно информации для ответа.",
            sources=[],
        )

    return AskResponse(status="answered", answer=answer, sources=sources)