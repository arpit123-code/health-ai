from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]

print (f"BASE_DIR: {BASE_DIR}")


class Settings(BaseSettings):
    app_name: str = "Voice healthcare RAG"
    app_env: str = "development"
    groq_api_key: str = ""
    tavily_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = "voice"
    pinecone_namespace: str = "voice-KB"
    embedding_model: str = "all-MiniLM-L6-v2"
    groq_model: str = "qwen/qwen3.8-27b"
    top_k: int = 4
    max_retries: int = 3
    # admin_api_key: str = "change-me"
    # audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    # upload_dir: str = str(BASE_DIR / "uploads")
    # sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

