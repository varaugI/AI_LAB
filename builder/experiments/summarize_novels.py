"""Create an extractive summary from an imported novel library."""

import argparse

from builder.books import NovelLearningSystem


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize imported novels without generating unsupported facts.")
    parser.add_argument("--index", default="novel_library.json")
    parser.add_argument("--title", help="Exact book title; omit to summarize the whole library")
    parser.add_argument("--sentences", type=int, default=8)
    parser.add_argument("--characters", type=int, default=3000)
    args = parser.parse_args(argv)

    system = NovelLearningSystem.load_library(args.index)
    result = system.summarize(
        title=args.title,
        max_sentences=args.sentences,
        max_characters=args.characters,
    )
    print(f"SUMMARY — {result.title}\n")
    print(result.summary)
    if result.sources:
        print("\nSOURCES")
        for source in result.sources:
            print(f"[{source.number}] {source.title} - {source.location}")


if __name__ == "__main__":
    main()
