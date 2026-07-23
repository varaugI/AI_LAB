"""Fine-tune the character network from a labelled word or paragraph image."""

import argparse
from pathlib import Path

from builder.ocr import load_bitmap, load_default_recognizer


def main():
    parser = argparse.ArgumentParser(
        description="Teach the OCR model from an image and its exact transcript."
    )
    parser.add_argument("image", help="TXT bitmap, PGM, PNG, JPEG, or BMP page.")
    transcript_group = parser.add_mutually_exclusive_group(required=True)
    transcript_group.add_argument("--text", help="Exact text shown in the image.")
    transcript_group.add_argument("--transcript", help="UTF-8 transcript file.")
    parser.add_argument("--model", default="alphabet_network.json")
    parser.add_argument("--output", default="text_network.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augment-copies", type=int, default=4)
    arguments = parser.parse_args()

    transcript = (
        arguments.text
        if arguments.text is not None
        else Path(arguments.transcript).read_text(encoding="utf-8").strip("\n")
    )
    recognizer = load_default_recognizer(arguments.model)
    bitmap = load_bitmap(arguments.image)

    before = recognizer.evaluate_page(bitmap, transcript)
    print("Before training:")
    print(before)

    recognizer.learn_from_labeled_page(
        bitmap,
        transcript,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        augment_copies=arguments.augment_copies,
        print_every=max(1, arguments.epochs // 10),
        seed=42,
    )
    recognizer.model.save(arguments.output)

    after = recognizer.evaluate_page(bitmap, transcript)
    print("\nAfter training:")
    print(after)
    print(f"\nSaved {arguments.output}")


if __name__ == "__main__":
    main()
