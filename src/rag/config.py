from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # que modelo uso y como de creativo es
    llm_provider: Literal["openai_compatible", "azure"] = "openai_compatible"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0

    # para cuando uso Groq, OpenAI, Ollama...
    llm_base_url: str | None = None
    llm_api_key: str | None = None

    # solo si uso Azure en vez de eso
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"


_settings = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
