"""
Gets raw text out of whatever format the user uploaded, and computes a
stable hash of the file's bytes so the caching layer can detect when the
projects file actually changed vs. was re-uploaded unchanged.
"""

import hashlib
import re
from pathlib import Path

import pdfplumber
from docx import Document


def hash_file_bytes(file_bytes: bytes) -> str:
    """Stable content hash — same content always produces the same hash,
    regardless of filename or upload timestamp."""
    return hashlib.sha256(file_bytes).hexdigest()


def extract_text(file_path: str, file_bytes: bytes = None) -> str:
    """
    Extracts plain text from a PDF or .docx file.
    Pass file_bytes directly if reading from an in-memory upload (e.g.
    Streamlit's UploadedFile) rather than a path on disk.
    """
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf_text(file_path)
    elif suffix == ".docx":
        return _extract_docx_text(file_path)
    elif suffix == ".txt":
        return Path(file_path).read_text()
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .docx, or .txt")


def _extract_pdf_text(file_path: str) -> str:
    """
    Extracts visible text AND real hyperlink targets. A PDF hyperlink
    lives as a separate annotation, not in the text stream — pdfplumber's
    page.extract_text() alone would silently drop it, same failure mode
    as the docx case below. For each link annotation, the target URL is
    appended in parentheses right after the visible text it's near
    (matched by vertical position on the page), e.g. a "GitHub" link
    becomes "GitHub (https://github.com/...)" in the extracted text —
    so downstream, a plain regex over the real extracted text is enough
    to recover it later (see parse_projects.py), no extra parsing needed.
    """
    text_chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if not page_text:
                continue

            links = getattr(page, "hyperlinks", None) or []
            if not links:
                text_chunks.append(page_text)
                continue

            # Append each link's real URL near the line it visually sits
            # on, matched by the annotation's vertical (top) position —
            # approximate, but good enough to keep the URL in the same
            # entry's line range once parse_projects.py does its
            # line-based boundary extraction.
            lines = page_text.split("\n")
            char_height = page.height / max(len(lines), 1)
            for link in links:
                url = link.get("uri")
                if not url:
                    continue
                line_idx = min(int(link["top"] / char_height), len(lines) - 1)
                lines[line_idx] = f"{lines[line_idx]} ({url})"
            text_chunks.append("\n".join(lines))

    return "\n\n".join(text_chunks)


def _extract_docx_text(file_path: str) -> str:
    """
    Extracts visible text AND real hyperlink targets.

    CRITICAL python-docx quirk: paragraph.text (and paragraph.runs)
    silently OMIT any run inside a <w:hyperlink> element — a hyperlink's
    visible text isn't just unlinked, it's invisible to plain-text
    extraction entirely. A project's real "GitHub" link would vanish
    before parse_projects.py ever saw it. This walks the paragraph's own
    XML directly instead of using paragraph.text, so both hyperlinked
    and plain runs are included, and each hyperlink's real target URL
    (resolved via the paragraph part's relationship map — never
    invented) is appended right after its visible text, e.g. a linked
    "GitHub" becomes "GitHub (https://github.com/...)" in the output —
    keeping the URL a verbatim, extractable part of the real text.
    """
    doc = Document(file_path)
    lines = []

    for paragraph in doc.paragraphs:
        pieces = []
        for node in paragraph._p.iter():
            tag = node.tag.rsplit("}", 1)[-1]
            if tag == "t" and node.text:
                pieces.append(node.text)
            elif tag == "hyperlink":
                r_id = node.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                )
                if r_id and r_id in paragraph.part.rels:
                    url = paragraph.part.rels[r_id].target_ref
                    display_text = "".join(
                        t.text for t in node.iter()
                        if t.tag.endswith("}t") and t.text
                    )
                    if display_text:
                        pieces.append(f"{display_text} ({url})")

        line = "".join(pieces).strip()
        if line:
            lines.append(line)

    return "\n".join(lines)