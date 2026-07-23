"""Ask grounded questions about novels stored in an AI LAB book index."""

from __future__ import annotations

import argparse

from builder.books import NovelAssistant


def print_reply(reply, show_sources=True):
    print("\nANSWER")
    print(reply.answer)
    print(f"\nConfidence: {reply.confidence:.0%}")
    if show_sources and reply.sources:
        print("\nSOURCES")
        for source in reply.sources:
            print(f"[{source.number}] {source.title} - {source.location}")
            print(f"    {source.excerpt}")


def build_parser():
    parser = argparse.ArgumentParser(description="Chat with imported PDF/EPUB/TXT novels.")
    parser.add_argument("--index", default="novel_library.json")
    parser.add_argument("--question", help="Ask one question and exit")
    parser.add_argument("--no-sources", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    assistant = NovelAssistant.load(args.index)

    if args.question:
        print_reply(assistant.answer(args.question), show_sources=not args.no_sources)
        return

    print("Book assistant ready. Ask about characters, places, events, or themes.")
    print("Commands: /search QUERY, /quit")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"/quit", "/exit", "quit", "exit"}:
            break
        if question.startswith("/search "):
            query = question[8:].strip()
            for number, result in enumerate(assistant.search(query, limit=5), start=1):
                print(f"[{number}] score={result.score:.3f} {result.chunk.title} - {result.chunk.location}")
                print(result.chunk.text[:400].replace("\n", " "))
            continue
        print_reply(assistant.answer(question), show_sources=not args.no_sources)


if __name__ == "__main__":
    main()
