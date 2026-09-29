from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"

COLLECTION_NAME = "iberdrola_reports"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        _collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)
    return _collection


def retrieve(query: str, n_results: int = 5) -> list[dict]:
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    chunks = []
    for i in range(len(documents)):
        chunk = {
            "text": documents[i],
            "source": metadatas[i]["source"],
            "page": metadatas[i]["page"],
        }
        chunks.append(chunk)
    return chunks
