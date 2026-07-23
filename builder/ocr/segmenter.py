"""Split a binary page into lines, words, and character images."""

from statistics import median

from builder.datasets.alphabet_data import CANVAS_HEIGHT, CANVAS_WIDTH, PATTERN_HEIGHT, PATTERN_WIDTH


def _validate_bitmap(bitmap):
    if not bitmap or not bitmap[0]:
        raise ValueError("Bitmap cannot be empty.")
    width = len(bitmap[0])
    if any(len(row) != width for row in bitmap):
        raise ValueError("Bitmap rows must all have the same width.")


def _runs(active_flags):
    result = []
    start = None
    for index, active in enumerate(active_flags + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            result.append((start, index - 1))
            start = None
    return result


def crop_to_ink(bitmap):
    _validate_bitmap(bitmap)
    active_rows = [any(value >= 0.5 for value in row) for row in bitmap]
    active_columns = [
        any(bitmap[y][x] >= 0.5 for y in range(len(bitmap)))
        for x in range(len(bitmap[0]))
    ]
    row_runs = _runs(active_rows)
    column_runs = _runs(active_columns)
    if not row_runs or not column_runs:
        return []
    top, bottom = row_runs[0][0], row_runs[-1][1]
    left, right = column_runs[0][0], column_runs[-1][1]
    return [row[left:right + 1] for row in bitmap[top:bottom + 1]]


def remove_isolated_pixels(bitmap, minimum_neighbors=1):
    """Remove single specks without changing normal character strokes."""
    _validate_bitmap(bitmap)
    height, width = len(bitmap), len(bitmap[0])
    cleaned = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            if bitmap[y][x] < 0.5:
                continue
            neighbors = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and bitmap[ny][nx] >= 0.5:
                        neighbors += 1
            if neighbors >= minimum_neighbors:
                cleaned[y][x] = 1.0
    return cleaned


def segment_lines(bitmap, minimum_row_ink=1):
    """Return cropped line matrices from top to bottom."""
    _validate_bitmap(bitmap)
    active_rows = [sum(value >= 0.5 for value in row) >= minimum_row_ink for row in bitmap]
    lines = []
    for top, bottom in _runs(active_rows):
        line = crop_to_ink(bitmap[top:bottom + 1])
        if line:
            lines.append(line)
    return lines


def infer_word_gap(gaps):
    """Infer a word gap by finding a large jump in inter-glyph spacing."""
    if not gaps:
        return float("inf")
    positive = sorted(gap for gap in gaps if gap > 0)
    if not positive:
        return float("inf")
    if len(set(positive)) == 1:
        return positive[0] + 1

    best_jump = 0
    split = None
    for left, right in zip(positive, positive[1:]):
        jump = right - left
        if jump > best_jump:
            best_jump = jump
            split = (left + right) / 2.0

    typical = median(positive)
    largest = positive[-1]
    if split is not None and best_jump >= max(2, typical * 0.75) and largest >= typical * 1.8:
        return split
    return largest + 1


def segment_line(line, word_gap_threshold=None, minimum_glyph_ink=2):
    """Return ``[(glyph, space_before), ...]`` for one text line."""
    _validate_bitmap(line)
    height, width = len(line), len(line[0])
    active_columns = [
        any(line[y][x] >= 0.5 for y in range(height))
        for x in range(width)
    ]
    column_runs = _runs(active_columns)
    if not column_runs:
        return []

    gaps = [
        column_runs[index + 1][0] - column_runs[index][1] - 1
        for index in range(len(column_runs) - 1)
    ]
    if word_gap_threshold is None:
        inferred = infer_word_gap(gaps)
        # The bundled font uses roughly two blank columns between letters and
        # a much larger gap between words. Scaling this estimate by line height
        # also handles pages rendered at 2x, 3x, or 4x resolution.
        scale_hint = max(1.0, height / PATTERN_HEIGHT)
        geometric_threshold = max(2.0, 4.0 * scale_hint)
        threshold = min(inferred, geometric_threshold)
    else:
        threshold = word_gap_threshold

    segments = []
    previous_right = None
    for left, right in column_runs:
        glyph = crop_to_ink([row[left:right + 1] for row in line])
        ink_count = sum(value >= 0.5 for row in glyph for value in row) if glyph else 0
        if ink_count < minimum_glyph_ink:
            continue
        gap = 0 if previous_right is None else left - previous_right - 1
        segments.append((glyph, previous_right is not None and gap >= threshold))
        previous_right = right
    return segments


def _resize_nearest(bitmap, new_width, new_height):
    source_height = len(bitmap)
    source_width = len(bitmap[0])
    result = [[0.0 for _ in range(new_width)] for _ in range(new_height)]
    for y in range(new_height):
        source_y = min(source_height - 1, int((y + 0.5) * source_height / new_height))
        for x in range(new_width):
            source_x = min(source_width - 1, int((x + 0.5) * source_width / new_width))
            result[y][x] = 1.0 if bitmap[source_y][source_x] >= 0.5 else 0.0
    return result


def normalize_glyph(glyph):
    """Fit a segmented glyph into the classifier's padded 7x9 canvas."""
    cropped = crop_to_ink(glyph)
    if not cropped:
        return [0.0] * (CANVAS_WIDTH * CANVAS_HEIGHT)

    source_height = len(cropped)
    source_width = len(cropped[0])
    scale = min(PATTERN_WIDTH / source_width, PATTERN_HEIGHT / source_height)
    target_width = max(1, min(PATTERN_WIDTH, round(source_width * scale)))
    target_height = max(1, min(PATTERN_HEIGHT, round(source_height * scale)))
    resized = _resize_nearest(cropped, target_width, target_height)

    canvas = [[0.0 for _ in range(CANVAS_WIDTH)] for _ in range(CANVAS_HEIGHT)]
    offset_x = (CANVAS_WIDTH - target_width) // 2
    offset_y = (CANVAS_HEIGHT - target_height) // 2
    for y in range(target_height):
        for x in range(target_width):
            canvas[offset_y + y][offset_x + x] = resized[y][x]
    return [value for row in canvas for value in row]


def segment_paragraph(bitmap, word_gap_threshold=None, denoise=False):
    """Segment a page into lines of normalized character vectors."""
    working = remove_isolated_pixels(bitmap) if denoise else bitmap
    result = []
    for line in segment_lines(working):
        segments = segment_line(line, word_gap_threshold=word_gap_threshold)
        result.append([
            {
                "input": normalize_glyph(glyph),
                "space_before": space_before,
                "raw_glyph": glyph,
            }
            for glyph, space_before in segments
        ])
    return result
