from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    chunk_max_chars: int = 1200

    llm_provider: str = "none"  # none | anthropic | openai
    llm_model: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""


settings = Settings()
