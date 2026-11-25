import json
import re

from openai import OpenAI

from app.config import settings

client = OpenAI(
    base_url=settings.openai_base_url,
    api_key=settings.openai_api_key,
)


def get_embedding(text: str, model: str | None = None) -> list[float]:
    response = client.embeddings.create(
        model=model or settings.embedding_model,
        input=text,
    )
    return response.data[0].embedding


def ask_llm(question: str, chunks: list[dict], model: str | None = None) -> dict:
    context = "\n\n".join(
        f"[{c['chunk_id']}]\n"
        f"Документ: {c['document']}\n"
        f"Раздел: {c['section']}\n"
        f"{c['content']}"
        for c in chunks
    )

    prompt = f"""Ты помощник финансовой службы.
Отвечай только на основании CONTEXT.
Если ответа в CONTEXT нет, верни status="insufficient_context".
Не выдумывай факты и не используй общие знания.

Верни ТОЛЬКО JSON без лишнего текста:
{{
  "status": "answered" или "insufficient_context",
  "answer": "краткий ответ на русском языке",
  "source_chunk_ids": ["id_фрагмента"]
}}

CONTEXT:
{context}

ВОПРОС:
{question}
"""

    response = client.chat.completions.create(
        model=model or settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        extra_body={"think": False},
    )

    text = response.choices[0].message.content or ""
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return json.loads(text)