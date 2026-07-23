"""List recurring characters, evidence sentences, and co-occurrences."""

import argparse

from builder.books import NovelLearningSystem


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze recurring character names in imported novels.")
    parser.add_argument("--index", default="novel_library.json")
    parser.add_argument("--name", help="Show one character in detail")
    parser.add_argument("--minimum-mentions", type=int, default=2)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args(argv)

    system = NovelLearningSystem.load_library(args.index)
    tracker = system.analyze_characters(minimum_mentions=args.minimum_mentions)
    if args.name:
        profiles = [tracker.get(args.name)]
    else:
        profiles = tracker.list(args.limit)

    for profile in [item for item in profiles if item]:
        print(f"\n{profile.name} — {profile.mentions} mentions")
        print(f"First appearance: {profile.first_appearance}")
        for sentence in profile.descriptions:
            print(f"  • {sentence}")
        if profile.relationships:
            print("  Related:", ", ".join(
                f"{name} ({count})" for name, count in profile.relationships.items()
            ))


if __name__ == "__main__":
    main()
