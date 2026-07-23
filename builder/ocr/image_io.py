"""Load and save binary OCR images.

Text bitmaps and PGM images work using only Python's standard library. PNG,
JPEG, BMP, and similar formats are supported when Pillow is installed.
"""

from pathlib import Path


def _validate_bitmap(bitmap):
    if not bitmap or not bitmap[0]:
        raise ValueError("Bitmap cannot be empty.")
    width = len(bitmap[0])
    if any(len(row) != width for row in bitmap):
        raise ValueError("Bitmap rows must all have the same width.")


def bitmap_to_ascii(bitmap, ink="#", background="."):
    _validate_bitmap(bitmap)
    return "\n".join(
        "".join(ink if value >= 0.5 else background for value in row)
        for row in bitmap
    )


def save_bitmap_text(bitmap, filename):
    Path(filename).write_text(bitmap_to_ascii(bitmap) + "\n", encoding="utf-8")


def load_bitmap_text(filename):
    lines = Path(filename).read_text(encoding="utf-8").splitlines()
    lines = [line.rstrip("\n\r") for line in lines if line.strip()]
    if not lines:
        raise ValueError("Bitmap text file is empty.")

    width = max(len(line) for line in lines)
    bitmap = []
    active = {"#", "@", "*", "1", "X", "x"}
    inactive = {".", "0", " ", "_", "-"}
    for line in lines:
        padded = line.ljust(width)
        row = []
        for character in padded:
            if character in active:
                row.append(1.0)
            elif character in inactive:
                row.append(0.0)
            else:
                raise ValueError(
                    f"Unsupported bitmap character {character!r}; use #/. or 1/0."
                )
        bitmap.append(row)
    _validate_bitmap(bitmap)
    return bitmap


def _tokenize_pgm(data):
    tokens = []
    for raw_line in data.splitlines():
        line = raw_line.split(b"#", 1)[0]
        tokens.extend(line.split())
    return tokens


def _load_pgm(filename, threshold=None, invert=None):
    data = Path(filename).read_bytes()
    if data.startswith(b"P2"):
        tokens = _tokenize_pgm(data)
        if len(tokens) < 4:
            raise ValueError("Invalid P2 PGM file.")
        _, width, height, maximum, *pixels = tokens
        width, height, maximum = int(width), int(height), int(maximum)
        values = [int(value) for value in pixels]
    elif data.startswith(b"P5"):
        # Parse the small P5 header while respecting comments.
        index = 2
        header = []
        while len(header) < 3:
            while index < len(data) and data[index:index + 1].isspace():
                index += 1
            if index < len(data) and data[index:index + 1] == b"#":
                while index < len(data) and data[index:index + 1] not in {b"\n", b"\r"}:
                    index += 1
                continue
            start = index
            while index < len(data) and not data[index:index + 1].isspace():
                index += 1
            header.append(int(data[start:index]))
        width, height, maximum = header
        while index < len(data) and data[index:index + 1].isspace():
            index += 1
        if maximum > 255:
            raise ValueError("16-bit PGM files are not supported.")
        values = list(data[index:index + width * height])
    else:
        raise ValueError("Not a P2 or P5 PGM file.")

    if len(values) < width * height:
        raise ValueError("PGM pixel data is incomplete.")
    normalized = [value / maximum for value in values[:width * height]]
    return _grayscale_to_bitmap(normalized, width, height, threshold, invert)


def _otsu_threshold(values):
    histogram = [0] * 256
    for value in values:
        histogram[max(0, min(255, int(round(value * 255))))] += 1

    total = len(values)
    total_sum = sum(index * count for index, count in enumerate(histogram))
    background_weight = 0
    background_sum = 0
    best_variance = -1.0
    best_threshold = 127

    for threshold in range(256):
        background_weight += histogram[threshold]
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += threshold * histogram[threshold]
        background_mean = background_sum / background_weight
        foreground_mean = (total_sum - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (
            background_mean - foreground_mean
        ) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold / 255.0


def _grayscale_to_bitmap(values, width, height, threshold=None, invert=None):
    threshold = _otsu_threshold(values) if threshold is None else threshold
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    # Dark ink on a light page is the normal case.
    dark_ink = [1.0 if value <= threshold else 0.0 for value in values]
    if invert is None:
        # Ink should normally cover less than half of a page.
        invert = sum(dark_ink) > len(dark_ink) / 2
    binary = [1.0 - value if invert else value for value in dark_ink]
    return [binary[start:start + width] for start in range(0, width * height, width)]


def load_bitmap(filename, threshold=None, invert=None):
    """Load a .txt bitmap, .pgm image, or a Pillow-supported image."""
    path = Path(filename)
    extension = path.suffix.lower()
    if extension in {".txt", ".bitmap", ".ascii"}:
        return load_bitmap_text(path)
    if extension == ".pgm":
        return _load_pgm(path, threshold=threshold, invert=invert)

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required for PNG/JPEG images. Install it with: pip install pillow"
        ) from error

    with Image.open(path) as image:
        grayscale = image.convert("L")
        width, height = grayscale.size
        values = [pixel / 255.0 for pixel in grayscale.getdata()]
    return _grayscale_to_bitmap(values, width, height, threshold, invert)


def save_png(bitmap, filename, scale=1):
    """Save a bitmap as a PNG. Requires Pillow."""
    _validate_bitmap(bitmap)
    if scale <= 0 or int(scale) != scale:
        raise ValueError("scale must be a positive integer.")
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Install Pillow with: pip install pillow") from error

    scale = int(scale)
    height = len(bitmap)
    width = len(bitmap[0])
    image = Image.new("L", (width, height), color=255)
    pixels = image.load()
    for y, row in enumerate(bitmap):
        for x, value in enumerate(row):
            pixels[x, y] = 0 if value >= 0.5 else 255
    if scale != 1:
        image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    image.save(filename)
