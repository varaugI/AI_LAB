"""End-to-end demonstration: render and recognize an unseen paragraph."""

from builder.ocr import load_default_recognizer, render_text


TEXT = "HELLO WORLD\nMY AI READS WORDS\nAND COMPLETE PARAGRAPHS"


def main():
    recognizer = load_default_recognizer()
    bitmap = render_text(TEXT, scale=3)
    result = recognizer.recognize(bitmap)

    print("Expected:")
    print(TEXT)
    print("\nPredicted:")
    print(result.text)
    print(f"\nAverage confidence: {result.average_confidence * 100:.2f}%")
    print("\nEvaluation:")
    print(recognizer.evaluate_page(bitmap, TEXT))


if __name__ == "__main__":
    main()
