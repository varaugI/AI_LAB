"""Render uppercase text with the same 5x7 glyphs used for training.

The renderer is useful for testing the OCR pipeline and for creating labelled
pages that can be used to fine-tune the character network.
"""

import random

from builder.datasets.alphabet_data import LETTER_PATTERNS, PATTERN_HEIGHT, PATTERN_WIDTH


def _line_width(line, scale, character_spacing, word_spacing):
    cursor = 0
    drew_character = False
    for character in line:
        if character == " ":
            cursor += word_spacing * scale
            drew_character = False
        elif character in LETTER_PATTERNS:
            cursor += PATTERN_WIDTH * scale
            cursor += character_spacing * scale
            drew_character = True
    if drew_character:
        cursor -= character_spacing * scale
    return max(1, cursor)


def render_text(
    text,
    scale=1,
    character_spacing=2,
    word_spacing=6,
    line_spacing=3,
    margin=2,
    noise_probability=0.0,
    seed=42,
    strict=False,
):
    """Render words, sentences, or paragraphs to a binary pixel matrix.

    Supported content is A-Z, spaces, and newlines. Lowercase text is converted
    to uppercase. With ``strict=False``, punctuation is skipped rather than
    crashing, because the current neural network has 26 A-Z output classes.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string.")
    if scale <= 0 or int(scale) != scale:
        raise ValueError("scale must be a positive integer.")
    if character_spacing < 1:
        raise ValueError("character_spacing must be at least 1.")
    if word_spacing <= character_spacing:
        raise ValueError("word_spacing must be larger than character_spacing.")
    if line_spacing < 1 or margin < 0:
        raise ValueError("line_spacing must be positive and margin non-negative.")
    if not 0.0 <= noise_probability < 1.0:
        raise ValueError("noise_probability must be in [0, 1).")

    scale = int(scale)
    normalized_lines = []
    for source_line in text.upper().splitlines() or [""]:
        line = []
        for character in source_line:
            if character in LETTER_PATTERNS or character == " ":
                line.append(character)
            elif strict:
                raise ValueError(
                    f"Unsupported character {character!r}. Use A-Z, spaces, and newlines."
                )
        normalized_lines.append("".join(line).rstrip())

    content_width = max(
        _line_width(line, scale, character_spacing, word_spacing)
        for line in normalized_lines
    )
    width = content_width + 2 * margin * scale
    line_height = PATTERN_HEIGHT * scale
    height = (
        len(normalized_lines) * line_height
        + max(0, len(normalized_lines) - 1) * line_spacing * scale
        + 2 * margin * scale
    )
    bitmap = [[0.0 for _ in range(width)] for _ in range(height)]

    y = margin * scale
    for line in normalized_lines:
        x = margin * scale
        for character in line:
            if character == " ":
                x += word_spacing * scale
                continue

            pattern = LETTER_PATTERNS[character]
            for source_y, row in enumerate(pattern):
                for source_x, pixel in enumerate(row):
                    if pixel != "#":
                        continue
                    for offset_y in range(scale):
                        for offset_x in range(scale):
                            bitmap[y + source_y * scale + offset_y][
                                x + source_x * scale + offset_x
                            ] = 1.0
            x += (PATTERN_WIDTH + character_spacing) * scale
        y += (PATTERN_HEIGHT + line_spacing) * scale

    if noise_probability:
        rng = random.Random(seed)
        for row in bitmap:
            for column in range(len(row)):
                if rng.random() < noise_probability:
                    row[column] = 1.0 - row[column]

    return bitmap
