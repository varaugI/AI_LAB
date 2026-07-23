"""Conversational, grounded chat over an AI LAB novel library."""

from __future__ import annotations

import argparse
from pathlib import Path

from builder.books import NovelLearningSystem


def print_sources(sources):
    if not sources:
        return
    print("\nSOURCES")
    for source in sources:
        print(f"[{source.number}] {source.title} - {source.location}")
        print(f"    {source.excerpt}")


def print_reply(reply, show_sources=True):
    print("\nANSWER")
    print(reply.answer)
    print(f"\nConfidence: {reply.confidence:.0%}")
    if show_sources:
        print_sources(reply.sources)


def build_parser():
    parser = argparse.ArgumentParser(description="Chat with imported PDF/EPUB/TXT novels.")
    parser.add_argument("--index", default="novel_library.json")
    parser.add_argument("--memory", default="conversation_memory.json")
    parser.add_argument("--question", help="Ask one question and exit")
    parser.add_argument("--no-sources", action="store_true")
    return parser


def show_characters(system, limit=20):
    profiles = system.analyze_characters().list(limit=limit)
    if not profiles:
        print("No recurring character names were detected.")
        return
    for profile in profiles:
        related = ", ".join(
            f"{name} ({count})" for name, count in list(profile.relationships.items())[:4]
        ) or "none detected"
        print(f"- {profile.name}: {profile.mentions} mentions; related: {related}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    system = NovelLearningSystem.load_library(args.index, memory_file=args.memory)

    if args.question:
        reply = system.chat(args.question)
        system.save_memory(args.memory)
        print_reply(reply, show_sources=not args.no_sources)
        return

    print("Book assistant ready. Follow-up questions use recent conversation context.")
    print("Commands: /search QUERY, /summary [TITLE], /characters, /character NAME,")
    print("          /memory, /clear, /quit")
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
            for number, result in enumerate(system.search(query, limit=5), start=1):
                print(f"[{number}] score={result.score:.3f} {result.chunk.title} - {result.chunk.location}")
                print(result.chunk.text[:400].replace("\n", " "))
            continue
        if question == "/summary" or question.startswith("/summary "):
            title = question[9:].strip() or None
            result = system.summarize(title=title)
            print(f"\nSUMMARY — {result.title}\n{result.summary}")
            print_sources(result.sources)
            continue
        if question == "/characters":
            show_characters(system)
            continue
        if question.startswith("/character "):
            name = question[11:].strip()
            profile = system.character(name)
            if not profile:
                print(f"No recurring character named {name!r} was found.")
                continue
            print(f"\n{profile.name} — {profile.mentions} mentions")
            print(f"First appearance: {profile.first_appearance}")
            for sentence in profile.descriptions:
                print(f"- {sentence}")
            if profile.relationships:
                print("Relationships:", ", ".join(
                    f"{other} ({count})" for other, count in profile.relationships.items()
                ))
            continue
        if question == "/memory":
            for turn in system.memory.recent(10):
                print(f"You: {turn.user}\nAI: {turn.assistant[:300]}\n")
            continue
        if question == "/clear":
            system.clear_memory()
            system.save_memory(args.memory)
            print("Conversation memory cleared. Book memory is unchanged.")
            continue

        print_reply(system.chat(question), show_sources=not args.no_sources)
        system.save_memory(args.memory)

    if system.memory.turns:
        system.save_memory(args.memory)


if __name__ == "__main__":
    main()
