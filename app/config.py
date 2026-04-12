from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    LLM_BASE_URL: str
    LLM_API_KEY: str
    LLM_MODEL: str = "google/gemini-2.5-flash-lite-preview-06-2025"
    LLM_TIMEOUT: int = 60
    LLM_RETRY_COUNT: int = 2
    EMBEDDING_MODEL: str = "google/gemini-embedding-001"

    # Todoist
    TODOIST_API_TOKEN: str
    TODOIST_PROJECT_ID: str | None = None

    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379

    # Qdrant
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333

    # Agent
    MAX_REFINEMENT_COUNT: int = 2
    SESSION_TTL_SECONDS: int = 1800
    CLASSIFIER_DEFAULT: str = "work_meeting"
    TOKEN_BUDGET_WARNING: int = 500_000

    # Transcription backend: "local" (whisperx) | "deepgram"
    TRANSCRIBE_BACKEND: Literal["local", "deepgram"] = "local"

    # Whisperx (local backend)
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    HF_TOKEN: str | None = None

    # DeepGram (API backend)
    DEEPGRAM_API_URL: str = "https://api.deepgram.com/v1/listen"
    DEEPGRAM_API_KEY: str | None = None
    DEEPGRAM_MODEL: str = "nova-2"
    DEEPGRAM_LANGUAGE: str = "ru"
    DEEPGRAM_DIARIZE: bool = True

    # Limits
    MAX_AUDIO_DURATION_SECONDS: int = 5400

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
