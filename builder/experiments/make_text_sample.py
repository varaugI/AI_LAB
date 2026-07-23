"""Create a sample paragraph image for the OCR recognizer."""

import argparse
from pathlib import Path

from builder.ocr import render_text, save_bitmap_text, save_png


DEFAULT_TEXT = "HELLO WORLD\nTHIS IS MY NEURAL NETWORK\nIT CAN READ NEW WORDS"


def main():
    parser = argparse.ArgumentParser(description="Render an OCR test page.")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to render.")
    parser.add_argument("--output", default="samples/sample_paragraph.png")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.0)
    arguments = parser.parse_args()

    bitmap = render_text(
        arguments.text,
        scale=arguments.scale,
        noise_probability=arguments.noise,
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".txt", ".bitmap", ".ascii"}:
        save_bitmap_text(bitmap, output)
    else:
        save_png(bitmap, output)
    print(f"Saved {output}")
    print("Recognize it with:")
    print(f"python -m builder.experiments.recognize_text {output}")


if __name__ == "__main__":
    main()
