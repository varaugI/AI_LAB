"""Load the trained alphabet model and recognize a hand-entered 5x7 letter."""

import os

from builder.datasets.alphabet_data import LABELS, display_input, pattern_to_input
from builder.experiments.train_alphabet import build_alphabet_model


MODEL_FILE = "alphabet_network.json"


def read_letter():
    print("\nDraw one capital letter using 5 characters on each of 7 rows.")
    print("Use # for a filled pixel and . for an empty pixel.")
    print("Example row: .###.")
    rows = []
    for row_number in range(1, 8):
        rows.append(input(f"Row {row_number}: ").strip())
    return pattern_to_input(rows)


def main():
    if not os.path.exists(MODEL_FILE):
        print(f"{MODEL_FILE} was not found.")
        print("Train it first with: python -m builder.experiments.train_alphabet")
        return

    model = build_alphabet_model()
    model.load(MODEL_FILE)
    print("Alphabet recognizer loaded. Type 'exit' when asked for row 1 to stop.")

    while True:
        first = input("\nRow 1 (or exit): ").strip()
        if first.lower() == "exit":
            break

        rows = [first]
        for row_number in range(2, 8):
            rows.append(input(f"Row {row_number}: ").strip())

        try:
            vector = pattern_to_input(rows)
        except ValueError as error:
            print(error)
            continue

        probabilities = model.predict([vector])[0]
        ranked = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)

        print("\nYour bitmap:")
        print(display_input(vector))
        print(f"\nPrediction: {LABELS[ranked[0]]}")
        print("Top three:")
        for index in ranked[:3]:
            print(f"  {LABELS[index]}: {probabilities[index] * 100:.2f}%")


if __name__ == "__main__":
    main()
