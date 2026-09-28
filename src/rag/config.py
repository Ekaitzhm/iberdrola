from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: Literal["openai_compatible", "azure"] = "openai_compatible"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0

    # compatibles con openai (Groq, OpenAI, Ollama, etc.)
    llm_base_url: str | None = None
    llm_api_key: str | None = None

    # Azure Openai
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"


@lru_cache
def get_settings() -> Settings:
    return Settings()
