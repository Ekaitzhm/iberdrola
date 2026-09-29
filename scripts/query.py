import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"

COLLECTION_NAME = "iberdrola_reports"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
N_RESULTS = 5


def main() -> None:
    query = " ".join(sys.argv[1:])
    if not query:
        raise SystemExit("Uso: python scripts/query.py <pregunta>")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)

    results = collection.query(query_texts=[query], n_results=N_RESULTS)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(documents)):
        doc = documents[i]
        meta = metadatas[i]
        dist = distances[i]
        print(f"[{meta['source']} p.{meta['page']}] (dist={dist:.3f})")
        print(doc[:300].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
