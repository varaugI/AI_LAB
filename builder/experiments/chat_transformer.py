"""Chat directly with a trained AI LAB transformer checkpoint."""

from __future__ import annotations

import argparse

from builder.llm import LocalTransformerBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--system", default="You are a helpful, accurate assistant.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    backend = LocalTransformerBackend(args.checkpoint, args.tokenizer or None)
    messages = [{"role": "system", "content": args.system}]
    print("AI LAB local transformer. Type /quit to stop, /clear to reset.")
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user == "/quit":
            break
        if user == "/clear":
            messages = messages[:1]
            print("Conversation cleared.")
            continue
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        response = backend.generate(messages, temperature=args.temperature, max_tokens=args.max_tokens)
        print(f"AI: {response.text}")
        messages.append({"role": "assistant", "content": response.text})


if __name__ == "__main__":
    main()
