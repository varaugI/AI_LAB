"""Run a complete book-ingestion and question-answering demonstration."""

from pathlib import Path

from builder.books import BM25Index, NovelAssistant, chunk_documents, read_document


def main():
    sample = Path(__file__).resolve().parents[2] / "samples" / "sample_novel.txt"
    document = read_document(sample)
    assistant = NovelAssistant(BM25Index(chunk_documents([document], max_words=90, overlap_words=15)))

    questions = [
        "Why did Mira enter the ruined observatory?",
        "Who carried the silver compass?",
        "What happened when the bell rang?",
    ]
    for question in questions:
        reply = assistant.answer(question)
        print(f"\nQ: {question}\nA: {reply.answer}")
        for source in reply.sources:
            print(f"   [{source.number}] {source.title} - {source.location}")


if __name__ == "__main__":
    main()
