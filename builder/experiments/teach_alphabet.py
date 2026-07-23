"""Teach the saved alphabet model a new hand-entered example."""

import os
import random

from builder.datasets.alphabet_data import (
    LABELS,
    augment_input,
    create_clean_alphabet_data,
    pattern_to_input,
)
from builder.experiments.train_alphabet import build_alphabet_model
from builder.framework import one_hot_encode


MODEL_FILE = "alphabet_network.json"


def main():
    if not os.path.exists(MODEL_FILE):
        print(f"{MODEL_FILE} was not found.")
        print("Train it first with: python -m builder.experiments.train_alphabet")
        return

    model = build_alphabet_model()
    model.load(MODEL_FILE)
    model.learning_rate = 0.001
    model.initial_learning_rate = 0.001

    print("Teach the model one of your own 5x7 drawings.")
    letter = input("Correct capital letter (A-Z): ").strip().upper()
    if letter not in LABELS:
        print("Please enter one capital letter from A to Z.")
        return

    rows = []
    for row_number in range(1, 8):
        rows.append(input(f"Row {row_number}: ").strip())

    try:
        custom_input = pattern_to_input(rows)
    except ValueError as error:
        print(error)
        return

    before = LABELS[model.predict_classes([custom_input])[0]]
    print(f"Prediction before teaching: {before}")

    custom_examples = augment_input(
        custom_input,
        copies=18,
        noise_probability=0.006,
        max_shift=1,
        seed=random.randint(1, 1_000_000),
    )
    class_index = LABELS.index(letter)
    custom_targets = [one_hot_encode(class_index, len(LABELS)) for _ in custom_examples]

    # Replay clean A-Z examples while learning the new drawing. This reduces
    # catastrophic forgetting of letters the model already knows.
    replay_inputs, replay_targets, _ = create_clean_alphabet_data()
    training_inputs = replay_inputs + custom_examples
    training_targets = replay_targets + custom_targets

    model.partial_fit(
        training_inputs,
        training_targets,
        epochs=45,
        batch_size=11,
        loss_type="cce",
        shuffle=True,
        seed=42,
    )

    probabilities = model.predict([custom_input])[0]
    ranked = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)
    print(f"Prediction after teaching: {LABELS[ranked[0]]}")
    print(f"Confidence: {probabilities[ranked[0]] * 100:.2f}%")

    model.save(MODEL_FILE)
    print(f"Updated {MODEL_FILE}")


if __name__ == "__main__":
    main()
