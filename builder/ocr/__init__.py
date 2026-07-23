"""Small OCR utilities built around the from-scratch character network."""

from .image_io import (
    load_bitmap,
    load_bitmap_text,
    save_bitmap_text,
    save_png,
    bitmap_to_ascii,
)
from .metrics import character_error_rate, levenshtein_distance, word_error_rate
from .recognizer import OCRResult, TextRecognizer, load_default_recognizer
from .segmenter import normalize_glyph, segment_paragraph
from .text_renderer import render_text

__all__ = [
    "OCRResult",
    "TextRecognizer",
    "bitmap_to_ascii",
    "character_error_rate",
    "levenshtein_distance",
    "load_bitmap",
    "load_bitmap_text",
    "load_default_recognizer",
    "normalize_glyph",
    "render_text",
    "save_bitmap_text",
    "save_png",
    "segment_paragraph",
    "word_error_rate",
]
