import json
import re
from pathlib import Path

from openai import OpenAI

from app.config import settings

client = OpenAI(
    base_url=settings.openai_base_url,
    api_key=settings.openai_api_key,
)

DOCS_DIR = Path("data/documents")
OUT_FILE = Path("data/chunks.json")


def split_document(text: str, document_name: str) -> list[dict]:
    # Простой способ: делим по заголовкам "## ..."
    parts = re.split(r"\n(?=## )", text)
    chunks: list[dict] = []

    for i, part in enumerate(parts, start=1):
        part = part.strip()
        if not part:
            continue

        lines = part.splitlines()
        section = lines[0].lstrip("# ").strip()

        chunks.append(
            {
                "chunk_id": f"{document_name}_{i}",
                "document": document_name,
                "section": section,
                "content": part,
            }
        )

    return chunks


def main() -> None:
    all_chunks: list[dict] = []

    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(split_document(text, path.name))

    if not all_chunks:
        raise RuntimeError("В папке data/documents нет .md файлов или они пустые.")

    for chunk in all_chunks:
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=chunk["content"],
        )
        chunk["embedding"] = response.data[0].embedding
        print(f"OK: {chunk['chunk_id']}")

    OUT_FILE.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Сохранено chunks: {len(all_chunks)} в {OUT_FILE}")


if __name__ == "__main__":
    main()