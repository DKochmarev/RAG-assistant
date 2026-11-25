# RAG-ассистент

Небольшой pet-проект: FastAPI-сервис, который отвечает на вопросы по выдуманным финансовым регламентам, опирается на найденные фрагменты документов и возвращает источники.

Если ответа в документах нет не выдумывает его.

> Все документы в проекте синтетические. Реальные данные компании не используются.

## Что умеет

- Загружает Markdown-документы из `data/documents/`.
- Разбивает документы на фрагменты и создаёт embeddings.
- Ищет релевантные фрагменты по вопросу пользователя.
- Отправляет вопрос и найденный контекст в LLM через OpenAI-compatible API.
- Возвращает структурированный ответ: `status`, `answer`, `sources`.
- Проверяет источники на стороне backend: модель возвращает только `chunk_id`, а документ и раздел берутся из найденных данных.

## Стек

- Python 3.11+
- FastAPI, Pydantic
- Ollama — локальный запуск LLM бесплатно
- OpenAI Python SDK — обращение к Ollama через OpenAI-compatible API
- `nomic-embed-text` — модель embeddings
- `qwen3.5:4b` — модель генерации ответа
- pytest

## Как работает

```text
Markdown-документы
    -> разбиение на chunks
    -> embeddings
    -> сохранение в data/chunks.json

Вопрос пользователя
    -> embedding вопроса
    -> поиск похожих chunks
    -> LLM получает вопрос + контекст
    -> backend проверяет JSON и источники
    -> ответ API: answered / insufficient_context
```

## Запуск

### 1. Установить зависимости

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Установить Ollama и скачать модели

```bash
ollama pull nomic-embed-text
ollama pull qwen3.5:4b
```

### 3. Создать `.env`

```env
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3.5:4b
EMBEDDING_MODEL=nomic-embed-text
```

### 4. Загрузить документы

```bash
python -m scripts.ingest
```

### 5. Запустить API

```bash
uvicorn app.main:app --reload
```

Swagger UI: http://127.0.0.1:8000/docs

## Пример запроса

`POST /ask`

```json
{
  "question": "Какие суточные положены при командировке по России?"
}
```

Пример ответа:

```json
{
  "status": "answered",
  "answer": "Суточные при командировках по территории РФ составляют 2 500 рублей в день.",
  "sources": [
    {
      "document": "business_trips.md",
      "section": "2. Суточные",
      "chunk_id": "business_trips.md_2"
    }
  ]
}
```

Если ответа в документах нет:

```json
{
  "question": "Какая зарплата у генерального директора?"
}
```

```json
{
  "status": "insufficient_context",
  "answer": "В предоставленных документах недостаточно информации для ответа.",
  "sources": []
}
```

## Тесты

```bash
pytest -v
```

Тесты используют mock LLM и не требуют запущенного Ollama.