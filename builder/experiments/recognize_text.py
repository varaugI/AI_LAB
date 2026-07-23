"""Recognize uppercase words, sentences, or paragraphs from an image file."""

import argparse

from builder.ocr import load_default_recognizer


def build_parser():
    parser = argparse.ArgumentParser(
        description="Recognize A-Z text from TXT bitmap, PGM, PNG, JPEG, or BMP images."
    )
    parser.add_argument("image", help="Path to the text image.")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional path to alphabet_network.json.",
    )
    parser.add_argument(
        "--minimum-confidence",
        type=float,
        default=0.0,
        help="Replace lower-confidence characters with '?'.",
    )
    parser.add_argument(
        "--word-gap",
        type=float,
        default=None,
        help="Manual minimum blank-column gap for a word break.",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Remove isolated one-pixel noise before segmentation.",
    )
    return parser


def main():
    arguments = build_parser().parse_args()
    recognizer = load_default_recognizer(
        model_file=arguments.model,
        minimum_confidence=arguments.minimum_confidence,
    )
    result = recognizer.recognize_file(
        arguments.image,
        word_gap_threshold=arguments.word_gap,
        denoise=arguments.denoise,
    )
    print("\nRecognized text")
    print("=" * 60)
    print(result.text)
    print("=" * 60)
    print(f"Characters: {result.character_count}")
    print(f"Average confidence: {result.average_confidence * 100:.2f}%")


if __name__ == "__main__":
    main()
