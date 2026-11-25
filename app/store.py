import json
import math
from pathlib import Path

CHUNKS_FILE = Path("data/chunks.json")


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "Файл data/chunks.json не найден. Сначала запусти: python -m scripts.ingest"
        )
    return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search_chunks(question_embedding: list[float], limit: int = 4) -> list[dict]:
    chunks = load_chunks()
    scored = []

    for chunk in chunks:
        score = cosine_similarity(question_embedding, chunk["embedding"])
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:limit]]