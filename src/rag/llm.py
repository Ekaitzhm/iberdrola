from openai import AzureOpenAI, OpenAI

from rag.config import get_settings

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    s = get_settings()
    if s.llm_provider == "azure":
        if not s.azure_openai_endpoint or not s.azure_openai_api_key:
            raise ValueError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")
        _client = AzureOpenAI(
            azure_endpoint=s.azure_openai_endpoint,
            api_key=s.azure_openai_api_key,
            api_version=s.azure_openai_api_version,
        )
        return _client

    if not s.llm_api_key:
        raise ValueError("Set LLM_API_KEY in .env")
    _client = OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key)
    return _client


def chat(messages: list[dict], temperature: float | None = None) -> str:
    s = get_settings()
    if temperature is None:
        temperature = s.llm_temperature

    response = get_client().chat.completions.create(
        model=s.llm_model,
        messages=messages,
        temperature=temperature,
    )

    content = response.choices[0].message.content
    if content is None:
        return ""
    return content
