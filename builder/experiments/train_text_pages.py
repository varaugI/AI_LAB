"""Fine-tune the A-Z classifier using complete synthetic text pages.

The pages contain arbitrary character sequences, not a fixed list of answers.
Segmentation extracts labelled character crops and the existing neural network
continues learning from them.
"""

import random

from builder.ocr import TextRecognizer, load_default_recognizer, render_text


TRAINING_PAGES = [
    "THE QUICK BROWN FOX\nJUMPS OVER THE LAZY DOG",
    "NEURAL NETWORKS LEARN\nFROM EXAMPLES AND ERRORS",
    "WORDS CAN BE NEW\nTHE LETTERS ARE COMPOSED",
    "READ A LINE\nREAD A SENTENCE\nREAD A PARAGRAPH",
]


def random_page(seed, lines=3, words_per_line=5):
    rng = random.Random(seed)
    result = []
    for _ in range(lines):
        words = []
        for _ in range(words_per_line):
            length = rng.randint(2, 8)
            words.append("".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(length)))
        result.append(" ".join(words))
    return "\n".join(result)


def main():
    recognizer = load_default_recognizer()
    all_inputs = []
    all_targets = []

    pages = list(TRAINING_PAGES)
    pages.extend(random_page(seed) for seed in range(10, 20))

    for page_index, transcript in enumerate(pages):
        bitmap = render_text(
            transcript,
            scale=1 + page_index % 3,
            character_spacing=2,
            word_spacing=7,
        )
        inputs, targets = recognizer.collect_labeled_examples(
            bitmap,
            transcript,
            augment_copies=2,
            noise_probability=0.006,
            max_shift=1,
            seed=1000 + page_index,
        )
        all_inputs.extend(inputs)
        all_targets.extend(targets)

    print(f"Collected {len(all_inputs)} character examples from complete pages.")
    recognizer.model.train_mini_batch(
        all_inputs,
        all_targets,
        batch_size=52,
        epochs=100,
        print_every=10,
        loss_type="cce",
        validation_split=0.15,
        metrics=["accuracy"],
        early_stopping=True,
        patience=20,
        lr_schedule="plateau",
        schedule_params={"patience": 6, "factor": 0.5, "min_learning_rate": 0.0002},
        seed=42,
    )
    recognizer.model.save("text_network.json")
    print("Saved text_network.json")

    test_text = random_page(999, lines=3, words_per_line=4)
    test_bitmap = render_text(test_text, scale=2)
    result = TextRecognizer(recognizer.model).recognize(test_bitmap)
    print("\nUnseen test page:")
    print(test_text)
    print("\nPrediction:")
    print(result.text)


if __name__ == "__main__":
    main()
