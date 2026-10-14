import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "German-English RAG Tutor"
    DEBUG: bool = True

    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333

    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379

    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    LLM_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
    WHISPER_MODEL: str = "base"

    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    AUDIO_CACHE_DIR: str = os.path.join(BASE_DIR, "data", "audio_cache")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure local directories exist
os.makedirs(settings.AUDIO_CACHE_DIR, exist_ok=True)
