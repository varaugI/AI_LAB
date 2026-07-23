import random

from builder.framework.data_utils import one_hot_encode


# Each capital letter is represented as a 5 x 7 bitmap.
# '#' means an active pixel and '.' means an empty pixel.
LETTER_PATTERNS = {
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "B": ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
    "C": [".####", "#....", "#....", "#....", "#....", "#....", ".####"],
    "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "F": ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
    "G": [".####", "#....", "#....", "#.###", "#...#", "#...#", ".###."],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "I": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"],
    "J": ["..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."],
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    "M": ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
    "N": ["#...#", "##..#", "##..#", "#.#.#", "#..##", "#..##", "#...#"],
    "O": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "P": ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    "Q": [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "U": ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "V": ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
    "W": ["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"],
    "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
    "Y": ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
    "Z": ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
}

LABELS = tuple(sorted(LETTER_PATTERNS))
PATTERN_WIDTH = 5
PATTERN_HEIGHT = 7
CANVAS_WIDTH = 7
CANVAS_HEIGHT = 9


def validate_patterns():
    if len(LETTER_PATTERNS) != 26:
        raise ValueError("The alphabet dataset must contain 26 letters.")
    for letter, rows in LETTER_PATTERNS.items():
        if len(rows) != PATTERN_HEIGHT:
            raise ValueError(f"{letter} must have {PATTERN_HEIGHT} rows.")
        if any(len(row) != PATTERN_WIDTH for row in rows):
            raise ValueError(f"Every row for {letter} must have width {PATTERN_WIDTH}.")


def render_pattern(pattern, horizontal_shift=0, vertical_shift=0,
                   noise_probability=0.0, rng=None):
    """Place a 5x7 pattern in a padded 7x9 canvas and flatten it."""
    rng = rng or random
    canvas = [[0.0 for _ in range(CANVAS_WIDTH)] for _ in range(CANVAS_HEIGHT)]

    base_x = (CANVAS_WIDTH - PATTERN_WIDTH) // 2 + horizontal_shift
    base_y = (CANVAS_HEIGHT - PATTERN_HEIGHT) // 2 + vertical_shift

    for source_y, row in enumerate(pattern):
        for source_x, character in enumerate(row):
            target_x = base_x + source_x
            target_y = base_y + source_y
            if 0 <= target_x < CANVAS_WIDTH and 0 <= target_y < CANVAS_HEIGHT:
                canvas[target_y][target_x] = 1.0 if character == "#" else 0.0

    if noise_probability:
        for y in range(CANVAS_HEIGHT):
            for x in range(CANVAS_WIDTH):
                if rng.random() < noise_probability:
                    canvas[y][x] = 1.0 - canvas[y][x]

    return [pixel for row in canvas for pixel in row]


def pattern_to_input(rows):
    """Convert seven user-provided rows of five pixels into a model input."""
    if len(rows) != PATTERN_HEIGHT:
        raise ValueError(f"Enter exactly {PATTERN_HEIGHT} rows.")
    normalized = []
    for row in rows:
        row = row.replace(" ", ".")
        if len(row) != PATTERN_WIDTH or any(char not in "#.*01" for char in row):
            raise ValueError("Each row must contain five characters: #/. or 1/0.")
        normalized.append("".join("#" if char in "#*1" else "." for char in row))
    return render_pattern(normalized)


def create_alphabet_dataset(copies_per_letter=10, noise_probability=0.015,
                            max_shift=1, seed=42, include_clean=True):
    """Create noisy and shifted A-Z training examples.

    The first copy of each letter is clean and centered. Remaining copies are
    randomly shifted by at most one pixel and receive light salt-and-pepper noise.
    """
    validate_patterns()
    if copies_per_letter <= 0:
        raise ValueError("copies_per_letter must be positive.")
    if not 0.0 <= noise_probability < 1.0:
        raise ValueError("noise_probability must be in [0, 1).")
    if max_shift < 0:
        raise ValueError("max_shift cannot be negative.")

    rng = random.Random(seed)
    inputs = []
    targets = []
    letters = []

    for class_index, letter in enumerate(LABELS):
        pattern = LETTER_PATTERNS[letter]
        for copy_index in range(copies_per_letter):
            clean = include_clean and copy_index == 0
            horizontal_shift = 0 if clean else rng.randint(-max_shift, max_shift)
            vertical_shift = 0 if clean else rng.randint(-max_shift, max_shift)
            noise = 0.0 if clean else noise_probability
            inputs.append(
                render_pattern(
                    pattern,
                    horizontal_shift=horizontal_shift,
                    vertical_shift=vertical_shift,
                    noise_probability=noise,
                    rng=rng,
                )
            )
            targets.append(one_hot_encode(class_index, len(LABELS)))
            letters.append(letter)

    combined = list(zip(inputs, targets, letters))
    rng.shuffle(combined)
    inputs, targets, letters = zip(*combined)
    return list(inputs), list(targets), list(letters)


def create_clean_alphabet_data():
    inputs = [render_pattern(LETTER_PATTERNS[letter]) for letter in LABELS]
    targets = [one_hot_encode(index, len(LABELS)) for index in range(len(LABELS))]
    return inputs, targets, list(LABELS)


def display_input(vector):
    if len(vector) != CANVAS_WIDTH * CANVAS_HEIGHT:
        raise ValueError("Vector has the wrong alphabet input size.")
    lines = []
    for start in range(0, len(vector), CANVAS_WIDTH):
        row = vector[start:start + CANVAS_WIDTH]
        lines.append("".join("##" if value >= 0.5 else "  " for value in row))
    return "\n".join(lines)


def augment_input(vector, copies=10, noise_probability=0.01, max_shift=1, seed=42):
    """Create shifted/noisy variants of a custom 7x9 input vector."""
    if len(vector) != CANVAS_WIDTH * CANVAS_HEIGHT:
        raise ValueError("Vector has the wrong alphabet input size.")
    if copies <= 0:
        raise ValueError("copies must be positive.")

    source = [
        vector[start:start + CANVAS_WIDTH]
        for start in range(0, len(vector), CANVAS_WIDTH)
    ]
    rng = random.Random(seed)
    results = [list(vector)]

    for _ in range(copies - 1):
        dx = rng.randint(-max_shift, max_shift)
        dy = rng.randint(-max_shift, max_shift)
        canvas = [[0.0 for _ in range(CANVAS_WIDTH)] for _ in range(CANVAS_HEIGHT)]

        for y in range(CANVAS_HEIGHT):
            for x in range(CANVAS_WIDTH):
                target_x = x + dx
                target_y = y + dy
                if 0 <= target_x < CANVAS_WIDTH and 0 <= target_y < CANVAS_HEIGHT:
                    canvas[target_y][target_x] = source[y][x]

        for y in range(CANVAS_HEIGHT):
            for x in range(CANVAS_WIDTH):
                if rng.random() < noise_probability:
                    canvas[y][x] = 1.0 - canvas[y][x]

        results.append([pixel for row in canvas for pixel in row])

    return results
