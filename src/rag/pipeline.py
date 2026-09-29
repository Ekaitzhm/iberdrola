from rag.llm import chat
from rag.retrieval import retrieve

SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas sobre los informes de Iberdrola "
    "usando unicamente el contexto proporcionado. Si la respuesta no esta en el "
    "contexto, di que no lo sabes. Cita la fuente y pagina de cada dato que uses."
)


def build_context(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        part = f"[{chunk['source']} p.{chunk['page']}]\n{chunk['text']}"
        parts.append(part)
    return "\n\n---\n\n".join(parts)


def answer(question: str, n_results: int = 5) -> str:
    chunks = retrieve(question, n_results=n_results)
    context = build_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"},
    ]
    return chat(messages)
