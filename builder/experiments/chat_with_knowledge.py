"""Chat with an AI LAB mixed-document library."""

from __future__ import annotations

import argparse

from builder.books import KnowledgeLearningSystem


MODES = ("chat", "explain", "study", "code", "legal", "creative")
DOMAINS = ("all", "general", "fiction", "school", "law", "programming")


def print_reply(reply, show_sources=True):
    print("\nAI")
    print(reply.answer)
    engine = reply.backend + (f" / {reply.model}" if reply.model else "")
    print(f"\nEngine: {engine} | Grounding: {reply.confidence:.0%}")
    if show_sources and reply.sources:
        print("\nSOURCES")
        for source in reply.sources:
            print(f"[{source.number}] {source.title} — {source.location}")
            print(f"    {source.excerpt}")


def build_parser():
    parser = argparse.ArgumentParser(description="Chat with imported knowledge.")
    parser.add_argument("--index", default="knowledge_library.json")
    parser.add_argument("--memory", default="conversation_memory.json")
    parser.add_argument("--mode", choices=MODES, default="chat")
    parser.add_argument("--domain", choices=DOMAINS, default="all")
    parser.add_argument("--question")
    parser.add_argument("--no-sources", action="store_true")
    parser.add_argument("--library-only", action="store_true", help="Do not use model knowledge outside imported passages")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    system = KnowledgeLearningSystem.load_library(args.index, memory_file=args.memory)
    mode = args.mode
    domain = args.domain

    def ask(question):
        reply = system.chat(
            question,
            mode=mode,
            domain=None if domain == "all" else domain,
            allow_general_knowledge=not args.library_only,
        )
        system.save_memory(args.memory)
        print_reply(reply, not args.no_sources)

    if args.question:
        ask(args.question)
        return

    stats = system.library_stats()
    print(f"AI LAB ready: {stats['chunks']} passages, {len(stats['titles'])} documents")
    print(f"Reply engine: {stats['backend']} {stats['model']}".strip())
    print("Commands: /mode MODE, /domain DOMAIN, /search QUERY, /summary [TITLE],")
    print("          /memory, /clear, /stats, /quit")

    while True:
        try:
            question = input(f"\nYou [{mode}/{domain}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"/quit", "/exit", "quit", "exit"}:
            break
        if question.startswith("/mode "):
            selected = question[6:].strip().lower()
            if selected in MODES:
                mode = selected
                print(f"Mode changed to {mode}.")
            else:
                print("Available modes:", ", ".join(MODES))
            continue
        if question.startswith("/domain "):
            selected = question[8:].strip().lower()
            if selected in DOMAINS:
                domain = selected
                print(f"Domain changed to {domain}.")
            else:
                print("Available domains:", ", ".join(DOMAINS))
            continue
        if question.startswith("/search "):
            query = question[8:].strip()
            results = system.search(query, limit=6, domain=None if domain == "all" else domain)
            for number, result in enumerate(results, start=1):
                print(f"[{number}] {result.chunk.title} — {result.chunk.location} ({result.score:.2f})")
                print(result.chunk.text[:500].replace("\n", " "))
            continue
        if question == "/summary" or question.startswith("/summary "):
            title = question[9:].strip() or None
            result = system.summarize(title=title)
            print(f"\nSUMMARY — {result.title}\n{result.summary}")
            continue
        if question == "/memory":
            for turn in system.memory.recent(10):
                print(f"You: {turn.user}\nAI: {turn.assistant[:500]}\n")
            continue
        if question == "/clear":
            system.clear_memory()
            system.save_memory(args.memory)
            print("Conversation memory cleared; imported knowledge remains.")
            continue
        if question == "/stats":
            print(system.library_stats())
            continue
        ask(question)


if __name__ == "__main__":
    main()
