import json
from pathlib import Path

import pymupdf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "processed"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_pages(pdf_path: Path) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + size
        chunks.append(text[start:end])
        if end >= length:
            break
        start = end - overlap
    return chunks


def ingest_pdf(pdf_path: Path) -> list[dict]:
    records = []
    for page in extract_pages(pdf_path):
        for chunk_id, chunk in enumerate(chunk_text(page["text"])):
            records.append(
                {
                    "source": pdf_path.name,
                    "page": page["page"],
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )
    return records


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        records = ingest_pdf(pdf_path)
        out_path = OUTPUT_DIR / f"{pdf_path.stem}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{pdf_path.name}: {len(records)} chunks -> {out_path}")


if __name__ == "__main__":
    main()
