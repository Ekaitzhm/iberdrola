from rag.config import get_settings
from rag.llm import chat

s = get_settings()
print(f"Provider: {s.llm_provider} | model: {s.llm_model}")
print(chat([{"role": "user", "content": "Responde solo con: OK"}]))
