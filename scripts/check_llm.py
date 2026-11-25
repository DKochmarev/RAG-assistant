from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

chat = client.chat.completions.create(
    model="qwen3.5:4b",
    messages=[{"role": "user", "content": "Ответь одним словом: работает?"}],
    temperature=0,
    extra_body={"think": False},
)
print("Chat:", chat.choices[0].message.content)

emb = client.embeddings.create(
    model="nomic-embed-text",
    input="Тестовый текст",
)
print("Embedding length:", len(emb.data[0].embedding))