import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"

COLLECTION_NAME = "iberdrola_reports"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 100


def load_chunks() -> list[dict]:
    chunks = []
    for jsonl_path in sorted(PROCESSED_DIR.glob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("No hay chunks en data/processed/. Ejecuta scripts/ingest.py primero.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    existing_names = []
    for collection_info in client.list_collections():
        existing_names.append(collection_info.name)
    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME, embedding_function=embedding_fn)

    ids = []
    documents = []
    metadatas = []
    for chunk in chunks:
        chunk_id = f"{chunk['source']}-p{chunk['page']}-c{chunk['chunk_id']}"
        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append(
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
            }
        )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(chunks))
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )
        print(f"Insertados {batch_end}/{len(chunks)}")

    print(f"Colección '{COLLECTION_NAME}' creada en {CHROMA_DIR} con {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
