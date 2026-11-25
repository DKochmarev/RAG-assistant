from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = "ollama"
    openai_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3.5:4b"
    embedding_model: str = "nomic-embed-text"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()