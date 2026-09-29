import sys

from rag.pipeline import answer


def main() -> None:
    question = " ".join(sys.argv[1:])
    if not question:
        raise SystemExit("Uso: python scripts/ask.py <pregunta>")
    print(answer(question))


if __name__ == "__main__":
    main()
