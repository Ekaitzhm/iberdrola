from functools import lru_cache

from openai import AzureOpenAI, OpenAI

from rag.config import get_settings


@lru_cache
def get_client() -> OpenAI:
    s = get_settings()
    if s.llm_provider == "azure":
        if not (s.azure_openai_endpoint and s.azure_openai_api_key):
            raise ValueError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")
        return AzureOpenAI(
            azure_endpoint=s.azure_openai_endpoint,
            api_key=s.azure_openai_api_key,
            api_version=s.azure_openai_api_version,
        )
    if not s.llm_api_key:
        raise ValueError("Set LLM_API_KEY in .env")
    return OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key)


def chat(messages: list[dict], **kwargs) -> str:
    s = get_settings()
    response = get_client().chat.completions.create(
        model=s.llm_model,
        messages=messages,
        temperature=kwargs.pop("temperature", s.llm_temperature),
        **kwargs,
    )
    return response.choices[0].message.content or ""
