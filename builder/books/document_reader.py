"""Read plain text, PDF, and EPUB documents into normalized text sections.

PDF text extraction uses the optional ``pypdf`` package. EPUB extraction uses
only Python's standard library. Scanned PDF OCR is optional and requires
PyMuPDF, Pillow, and pytesseract plus a local Tesseract installation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from pathlib import Path
import csv
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from urllib.parse import unquote


SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".rst", ".pdf", ".epub", ".docx", ".html", ".htm",
    ".json", ".csv", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".cs", ".go", ".rs", ".php", ".rb", ".sql", ".sh",
    ".ps1", ".kt", ".swift",
}

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".cs", ".go", ".rs", ".php", ".rb", ".sql", ".sh",
    ".ps1", ".kt", ".swift",
}


@dataclass
class DocumentSection:
    source: str
    title: str
    location: str
    text: str

    def to_dict(self):
        return asdict(self)


@dataclass
class LoadedDocument:
    path: str
    title: str
    kind: str
    sections: list[DocumentSection]
    domain: str = "general"

    @property
    def text(self) -> str:
        return "\n\n".join(section.text for section in self.sections if section.text)

    @property
    def word_count(self) -> int:
        return len(re.findall(r"\b\w+\b", self.text, flags=re.UNICODE))


class _HTMLTextExtractor(HTMLParser):
    """Small dependency-free HTML-to-text converter for EPUB chapters."""

    BLOCK_TAGS = {
        "p", "div", "section", "article", "header", "footer", "aside",
        "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote",
        "br", "hr", "tr", "table", "pre",
    }
    SKIP_TAGS = {"script", "style", "svg", "math", "noscript"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        return normalize_text("".join(self.parts))


def normalize_text(text: str) -> str:
    """Normalize document text while preserving paragraph boundaries."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\u00a0", " ")
    # Join words split by line-wrap hyphenation, but keep real hyphenated words.
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")



def infer_domain(title: str, text: str, extension: str = "") -> str:
    """Infer a broad library domain without requiring the user to classify files."""
    extension = extension.lower()
    if extension in CODE_EXTENSIONS:
        return "programming"
    sample = f"{title}\n{text[:12000]}".lower()
    signals = {
        "law": (
            "court", "statute", "plaintiff", "defendant", "jurisdiction", "legal",
            "constitution", "section ", "article ", "judgment", "liability", "contract",
        ),
        "programming": (
            "source code", "programming", "algorithm", "function ", "class ", "api ",
            "compiler", "database", "javascript", "python", "java ", "software",
        ),
        "school": (
            "exercise", "learning objective", "chapter", "textbook", "theorem", "formula",
            "question", "answer", "biology", "chemistry", "physics", "mathematics",
            "history", "geography",
        ),
        "fiction": (
            "chapter", "novel", "prologue", "epilogue", "said", "asked", "replied",
        ),
    }
    scores = {name: sum(sample.count(term) for term in terms) for name, terms in signals.items()}
    if extension == ".epub":
        scores["fiction"] += 2
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"


def read_html_document(path: str | Path) -> LoadedDocument:
    path = Path(path)
    text = _html_to_text(path.read_bytes())
    domain = infer_domain(path.stem, text, path.suffix)
    return LoadedDocument(
        str(path), path.stem, "html",
        [DocumentSection(str(path), path.stem, "document", text)],
        domain=domain,
    )


def read_docx(path: str | Path) -> LoadedDocument:
    """Read DOCX text with the standard library; no python-docx dependency."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a valid DOCX file: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
    except (KeyError, ET.ParseError) as exc:
        raise ValueError(f"Could not read DOCX text: {path}") from exc

    paragraphs = []
    for paragraph in root.iter():
        if _xml_local_name(paragraph.tag) != "p":
            continue
        pieces = []
        for node in paragraph.iter():
            name = _xml_local_name(node.tag)
            if name == "t" and node.text:
                pieces.append(node.text)
            elif name == "tab":
                pieces.append("\t")
            elif name in {"br", "cr"}:
                pieces.append("\n")
        value = "".join(pieces).strip()
        if value:
            paragraphs.append(value)
    text = normalize_text("\n\n".join(paragraphs))
    if not text:
        raise ValueError(f"No readable text was found in DOCX: {path}")
    domain = infer_domain(path.stem, text, path.suffix)
    return LoadedDocument(
        str(path), path.stem, "docx",
        [DocumentSection(str(path), path.stem, "document", text)],
        domain=domain,
    )


def read_json_document(path: str | Path) -> LoadedDocument:
    path = Path(path)
    raw = _decode_bytes(path.read_bytes())
    try:
        value = json.loads(raw)
        text = json.dumps(value, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        text = raw
    text = normalize_text(text)
    domain = infer_domain(path.stem, text, path.suffix)
    return LoadedDocument(
        str(path), path.stem, "json",
        [DocumentSection(str(path), path.stem, "document", text)],
        domain=domain,
    )


def read_csv_document(path: str | Path) -> LoadedDocument:
    path = Path(path)
    raw = _decode_bytes(path.read_bytes())
    rows = []
    for row_number, row in enumerate(csv.reader(raw.splitlines()), start=1):
        if not row:
            continue
        rows.append(f"Row {row_number}: " + " | ".join(cell.strip() for cell in row))
    text = normalize_text("\n".join(rows))
    domain = infer_domain(path.stem, text, path.suffix)
    return LoadedDocument(
        str(path), path.stem, "csv",
        [DocumentSection(str(path), path.stem, "table", text)],
        domain=domain,
    )

def read_text_document(path: str | Path) -> LoadedDocument:
    path = Path(path)
    text = normalize_text(_decode_bytes(path.read_bytes()))
    section = DocumentSection(
        source=str(path), title=path.stem, location="document", text=text
    )
    kind = path.suffix.lower().lstrip(".") or "text"
    return LoadedDocument(
        str(path), path.stem, kind, [section],
        domain=infer_domain(path.stem, text, path.suffix),
    )


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _epub_package_path(archive: zipfile.ZipFile) -> str | None:
    try:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        for element in container.iter():
            if _xml_local_name(element.tag) == "rootfile":
                return element.attrib.get("full-path")
    except (KeyError, ET.ParseError):
        return None
    return None


def _epub_spine_items(archive: zipfile.ZipFile, package_path: str):
    """Return (book_title, ordered chapter paths) from an EPUB OPF file."""
    root = ET.fromstring(archive.read(package_path))
    package_dir = Path(package_path).parent
    manifest: dict[str, str] = {}
    spine_ids: list[str] = []
    title = None

    for element in root.iter():
        name = _xml_local_name(element.tag)
        if name == "title" and title is None and element.text:
            title = normalize_text(element.text)
        elif name == "item":
            item_id = element.attrib.get("id")
            href = element.attrib.get("href")
            media_type = element.attrib.get("media-type", "")
            properties = element.attrib.get("properties", "").split()
            if (
                item_id and href and "nav" not in properties
                and ("html" in media_type or href.lower().endswith((".html", ".xhtml", ".htm")))
            ):
                manifest[item_id] = str((package_dir / unquote(href)).as_posix())
        elif name == "itemref":
            item_id = element.attrib.get("idref")
            if item_id:
                spine_ids.append(item_id)

    ordered = [manifest[item_id] for item_id in spine_ids if item_id in manifest]
    if not ordered:
        ordered = list(manifest.values())
    return title, ordered


def _html_to_text(data: bytes) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(_decode_bytes(data))
    parser.close()
    return parser.text()


def read_epub(path: str | Path, max_sections: int | None = None) -> LoadedDocument:
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a valid EPUB/ZIP file: {path}")

    with zipfile.ZipFile(path) as archive:
        package_path = _epub_package_path(archive)
        title = path.stem
        chapter_paths: list[str] = []

        if package_path:
            try:
                parsed_title, chapter_paths = _epub_spine_items(archive, package_path)
                if parsed_title:
                    title = parsed_title
            except (KeyError, ET.ParseError):
                chapter_paths = []

        if not chapter_paths:
            chapter_paths = sorted(
                name for name in archive.namelist()
                if name.lower().endswith((".html", ".xhtml", ".htm"))
                and not name.lower().startswith("meta-inf/")
            )

        if max_sections is not None:
            chapter_paths = chapter_paths[:max_sections]

        sections: list[DocumentSection] = []
        for chapter_number, chapter_path in enumerate(chapter_paths, start=1):
            try:
                text = _html_to_text(archive.read(chapter_path))
            except KeyError:
                continue
            if not text:
                continue
            sections.append(DocumentSection(
                source=str(path),
                title=title,
                location=f"chapter {chapter_number}: {chapter_path}",
                text=text,
            ))

    if not sections:
        raise ValueError(f"No readable HTML chapters were found in EPUB: {path}")
    return LoadedDocument(str(path), title, "epub", sections, domain=infer_domain(title, " ".join(s.text[:2000] for s in sections), ".epub"))


def _ocr_pdf_page(path: Path, page_index: int, dpi: int = 180) -> str:
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Scanned PDF OCR requires pymupdf, pillow, and pytesseract. "
            "Tesseract OCR must also be installed on the computer."
        ) from exc

    with fitz.open(path) as document:
        page = document.load_page(page_index)
        scale = dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        return normalize_text(pytesseract.image_to_string(image))


def read_pdf(
    path: str | Path,
    max_pages: int | None = None,
    ocr_scanned: bool = False,
    minimum_text_characters: int = 30,
) -> LoadedDocument:
    path = Path(path)
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF reading requires pypdf: pip install pypdf") from exc

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError(f"Cannot read encrypted PDF without its password: {path}") from exc

    metadata_title = None
    try:
        metadata_title = reader.metadata.title if reader.metadata else None
    except Exception:
        metadata_title = None
    title = normalize_text(metadata_title or path.stem)

    page_count = len(reader.pages)
    if max_pages is not None:
        page_count = min(page_count, max_pages)

    sections: list[DocumentSection] = []
    for page_index in range(page_count):
        try:
            text = normalize_text(reader.pages[page_index].extract_text() or "")
        except Exception:
            text = ""

        if ocr_scanned and len(text) < minimum_text_characters:
            ocr_text = _ocr_pdf_page(path, page_index)
            if len(ocr_text) > len(text):
                text = ocr_text

        if text:
            sections.append(DocumentSection(
                source=str(path),
                title=title,
                location=f"page {page_index + 1}",
                text=text,
            ))

    if not sections:
        hint = " Enable ocr_scanned=True for image-only pages." if not ocr_scanned else ""
        raise ValueError(f"No readable text was found in PDF: {path}.{hint}")
    return LoadedDocument(str(path), title, "pdf", sections, domain=infer_domain(title, " ".join(s.text[:2000] for s in sections), ".pdf"))


def read_document(
    path: str | Path,
    max_pages: int | None = None,
    max_sections: int | None = None,
    ocr_scanned: bool = False,
) -> LoadedDocument:
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type {extension!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if extension == ".pdf":
        return read_pdf(path, max_pages=max_pages, ocr_scanned=ocr_scanned)
    if extension == ".epub":
        return read_epub(path, max_sections=max_sections)
    if extension == ".docx":
        return read_docx(path)
    if extension in {".html", ".htm"}:
        return read_html_document(path)
    if extension == ".json":
        return read_json_document(path)
    if extension == ".csv":
        return read_csv_document(path)
    return read_text_document(path)


def discover_documents(paths: list[str | Path], recursive: bool = True) -> list[Path]:
    """Expand files/directories into a stable list of supported documents."""
    discovered: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            discovered.append(path)
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            discovered.extend(
                candidate for candidate in iterator
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        else:
            raise FileNotFoundError(path)
    # Resolve duplicates while preserving sorted, deterministic order.
    unique = {str(path.resolve()): path for path in discovered}
    return [unique[key] for key in sorted(unique)]


def read_documents(
    paths: list[str | Path],
    recursive: bool = True,
    max_pages: int | None = None,
    max_sections: int | None = None,
    ocr_scanned: bool = False,
) -> list[LoadedDocument]:
    documents = []
    for path in discover_documents(paths, recursive=recursive):
        documents.append(read_document(
            path,
            max_pages=max_pages,
            max_sections=max_sections,
            ocr_scanned=ocr_scanned,
        ))
    return documents
