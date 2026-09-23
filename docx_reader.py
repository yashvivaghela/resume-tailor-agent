"""
docx_reader.py — extracts paragraph-level structure from a resume file,
generically (not tuned to any one resume's specific template).

Two-step design, same split as Phase 0's parse_projects.py:
  1. Mechanical extraction (this file): pull out every paragraph's text
     + structural signals (style name, bold, list-formatting) — no
     judgment involved, just reading what's actually in the file.
  2. LLM interpretation (docx_sections.py, next file): use those signals
     to identify which paragraphs are entry titles vs. bullets, since
     that requires judgment a fixed rule can't generalize across
     different resume templates.
"""

from docx import Document


def extract_paragraph_structure(docx_path: str) -> list[dict]:
    """
    Returns a list of paragraph records:
      {
        "index": int,
        "text": str,
        "style_name": str,       # e.g. "Heading 2", "Normal", "List Bullet"
        "is_bold": bool,         # True if the paragraph's first run is bold
        "is_list_item": bool,    # True if it's a Word list/bullet paragraph
      }
    Empty paragraphs are skipped (they carry no structural signal).
    """
    doc = Document(docx_path)
    records = []

    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if not text:
            continue

        # Check ALL runs, not just the first — a paragraph can have
        # bold applied partway through (e.g. only the job title bolded,
        # not a trailing comma), and checking only run[0] would miss
        # paragraphs that are still visually a bolded title.
        is_bold = any(run.bold for run in para.runs if run.bold is not None)

        # A paragraph is a "list item" if its style name suggests it
        # (e.g. "List Bullet", "List Paragraph") OR it has Word's
        # numbering/bullet XML properties directly applied.
        style_name = para.style.name if para.style else "Normal"
        has_numbering_xml = para._p.pPr is not None and para._p.pPr.numPr is not None
        is_list_item = "List" in style_name or has_numbering_xml

        # Section headers ("EXPERIENCE", "PROJECTS") are overwhelmingly
        # ALL CAPS in real resumes, while entry titles ("Teaching
        # Assistant, University CS Department") are normally mixed
        # case. This is a cheap, near-universal signal worth surfacing
        # explicitly — without it, an all-caps header and a bolded
        # title look IDENTICAL in every other signal we extract.
        is_all_caps = text.isupper()

        # Indentation is a more structural signal than bold/caps — it's
        # an actual document property capturing real nesting (section >
        # entry title > bullets), not just a stylistic convention. Many
        # people indent sub-items naturally even without ever touching
        # a Styles panel, so this often works even when bold/caps don't.
        left_indent = para.paragraph_format.left_indent
        indent_pt = round(left_indent.pt, 1) if left_indent is not None else 0.0

        records.append({
            "index": i,
            "text": text,
            "style_name": style_name,
            "is_bold": is_bold,
            "is_list_item": is_list_item,
            "is_all_caps": is_all_caps,
            "indent_pt": indent_pt,
        })

    return records


def format_for_llm(records: list[dict]) -> str:
    """
    Turns the structured records into a compact, annotated text dump
    the model can reason over — same spirit as Phase 0's line-numbered
    text, but with structural tags instead of just line numbers.
    """
    lines = []
    for r in records:
        tags = [r["style_name"]]
        if r["is_bold"]:
            tags.append("BOLD")
        if r["is_list_item"]:
            tags.append("LIST_ITEM")
        if r["is_all_caps"]:
            tags.append("ALL_CAPS")
        if r["indent_pt"] > 0:
            tags.append(f"INDENT={r['indent_pt']}pt")
        tag_str = ", ".join(tags)
        lines.append(f"{r['index']} [{tag_str}]: {r['text']}")
    return "\n".join(lines)


# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) < 2:
#         print("Usage: python docx_reader.py <path_to_resume.docx>")
#         sys.exit(1)

#     records = extract_paragraph_structure(sys.argv[1])
#     print(format_for_llm(records))