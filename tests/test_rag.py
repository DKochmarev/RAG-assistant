from unittest.mock import patch

from app.rag import answer_question


def fake_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "business_trips.md_2",
            "document": "business_trips.md",
            "section": "2. Суточные",
            "content": "Суточные при командировках по территории РФ составляют 2 500 рублей в день.",
            "embedding": [0.1, 0.2],
        }
    ]


@patch("app.rag.search_chunks")
@patch("app.rag.get_embedding")
@patch("app.rag.ask_llm")
def test_answered_returns_sources(mock_ask_llm, mock_get_embedding, mock_search_chunks):
    mock_get_embedding.return_value = [0.1, 0.2]
    mock_search_chunks.return_value = fake_chunks()
    mock_ask_llm.return_value = {
        "status": "answered",
        "answer": "Суточные составляют 2 500 рублей в день.",
        "source_chunk_ids": ["business_trips.md_2"],
    }

    result = answer_question("Какие суточные по России?")

    assert result.status == "answered"
    assert len(result.sources) == 1
    assert result.sources[0].document == "business_trips.md"


@patch("app.rag.search_chunks")
@patch("app.rag.get_embedding")
@patch("app.rag.ask_llm")
def test_unknown_source_id_becomes_insufficient(
    mock_ask_llm, mock_get_embedding, mock_search_chunks
):
    mock_get_embedding.return_value = [0.1, 0.2]
    mock_search_chunks.return_value = fake_chunks()
    mock_ask_llm.return_value = {
        "status": "answered",
        "answer": "Выдуманный ответ",
        "source_chunk_ids": ["not_existing_chunk"],
    }

    result = answer_question("Какая зарплата у директора?")

    assert result.status == "insufficient_context"
    assert result.sources == []


@patch("app.rag.search_chunks")
@patch("app.rag.get_embedding")
@patch("app.rag.ask_llm")
def test_insufficient_context_returns_empty_sources(
    mock_ask_llm, mock_get_embedding, mock_search_chunks
):
    mock_get_embedding.return_value = [0.1, 0.2]
    mock_search_chunks.return_value = fake_chunks()
    mock_ask_llm.return_value = {
        "status": "insufficient_context",
        "answer": "В документах нет информации.",
        "source_chunk_ids": [],
    }

    result = answer_question("Какая зарплата у директора?")

    assert result.status == "insufficient_context"
    assert result.sources == []