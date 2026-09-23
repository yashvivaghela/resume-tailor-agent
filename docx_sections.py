"""
docx_sections.py — groups extracted paragraphs (from docx_reader.py)
into resume entries: which paragraph is a title, which are its bullets.

CRITICAL DESIGN GUARANTEE, same as Phase 0's parse_projects.py: the
model is shown paragraph INDICES and structural tags only. It is NEVER
asked to reproduce, rewrite, or summarize any text. Its only output is
which index numbers belong together. The actual text used everywhere
downstream is looked up directly from the ALREADY-EXTRACTED records
(docx_reader.py's output), never from anything the model echoes back.
This makes it structurally impossible for a summarized/altered version
of a bullet to sneak into the pipeline at this step.
"""

import json

from llm_client import call_llm
from docx_reader import extract_paragraph_structure, format_for_llm

SECTION_KEYWORDS = {
    "experience": ["experience", "work history", "employment"],
    "project": ["project"],
}


def group_deterministically(records: list[dict]) -> list[dict] | None:
    """
    Pure-code, zero-LLM-cost first pass. Handles the common case where
    a resume cleanly uses ALL_CAPS section headers + bold entry titles
    + everything else as bullets. Returns None (not an empty list) if
    the pattern doesn't hold cleanly, signaling the caller to fall back
    to the LLM-based interpretation instead of guessing.

    This is checked FIRST because it's instant, free, and has zero risk
    of a hallucinated index — the LLM step only runs when this genuinely
    can't confidently produce an answer.
    """
    sections = []
    current_type = None
    current_title_idx = None
    current_bullet_idxs = []
    type_counts = {"experience": 0, "project": 0}

    def flush():
        nonlocal current_title_idx, current_bullet_idxs
        if current_title_idx is not None:
            type_counts[current_type] += 1
            sections.append({
                "section_name": f"{current_type.capitalize()} {type_counts[current_type]}",
                "entry_type": current_type,
                "title_index": current_title_idx,
                "bullet_indices": list(current_bullet_idxs),
            })
        current_title_idx = None
        current_bullet_idxs = []

    for r in records:
        if r["is_all_caps"] and r["is_bold"]:
            flush()
            text_lower = r["text"].lower()
            matched_type = None
            for entry_type, keywords in SECTION_KEYWORDS.items():
                if any(kw in text_lower for kw in keywords):
                    matched_type = entry_type
            if matched_type is None:
                # An all-caps bold header we don't recognize (e.g.
                # "SKILLS", "EDUCATION") — not ambiguous, just not a
                # section we're extracting entries from. Skip it and
                # anything under it until the next recognized header.
                current_type = None
            else:
                current_type = matched_type
            continue

        if current_type is None:
            continue  # not inside a recognized section — skip silently

        if r["is_bold"] and not r["is_all_caps"]:
            flush()
            current_title_idx = r["index"]
            continue

        if current_title_idx is not None:
            current_bullet_idxs.append(r["index"])
        else:
            # Content with no title established yet and no recognized
            # section context — the heuristic can't confidently place
            # this. Bail out entirely rather than guess.
            return None

    flush()
    return sections if sections else None


SYSTEM_PROMPT = """You are given a numbered, tagged dump of every
paragraph in a resume document. Your job is ONLY to identify structure
— which paragraph index is an ENTRY TITLE (a job title or project name),
and which paragraph indices immediately following it are that entry's
BULLET points — for every entry in BOTH the Experience and Projects
sections of the resume.

You do NOT see or need to reproduce any text yourself. You are working
PURELY with paragraph index numbers. Do not include any bullet or title
text in your response — only index numbers and short labels.

Resumes vary wildly in formatting. Some use real Word heading/list
styles (tagged "Heading" or "List Bullet" or "LIST_ITEM" below); others
use plain paragraphs with only bold formatting and manually-typed dash
or bullet characters, with no special styles at all; others use NO
bold/caps/list-styles at all and rely purely on INDENTATION to show
hierarchy. Use ALL the signals given (style name, BOLD, LIST_ITEM,
ALL_CAPS, INDENT, and the text's own shape — e.g. a line starting with
"-" is very likely a bullet even with no special style) to make your
best judgment.

INDENT is often the MOST reliable signal, since it's a real structural
property rather than just a stylistic choice: a section header
typically has the LEAST indent (often none), an entry title has MORE
indent than its section header, and that entry's bullets have MORE
indent still than their title. When indent values differ across
paragraphs, trust the indent hierarchy strongly — increasing indent
usually means deeper nesting (section > title > bullets).

A section header (like "Experience" or "Projects") is NOT itself an
entry title — skip over it, it doesn't belong to any entry. Section
headers and entry titles can look IDENTICAL in formatting tags (both
might be tagged BOLD with no other distinguishing style) — when that
happens, use these additional cues:
- ALL_CAPS + a single short word ("EXPERIENCE", "PROJECTS") is almost
  always a section header, never an entry title.
- An entry title is usually mixed-case and multi-word (a job title, a
  project name, often with a comma separating role/organization).
- POSITION matters: a section header is typically followed by ANOTHER
  bold/title-like line (the first entry's title). An entry title is
  typically followed by non-bold, list-like, or dash-prefixed lines
  (its bullets).

For each entry you identify, output:
- section_name: a short label for this entry, e.g. "Experience 1",
  "Project 2" (just number them in order within each type)
- entry_type: "experience" or "project"
- title_index: the paragraph index of this entry's title
- bullet_indices: a list of paragraph indices that are this entry's bullets

Respond with ONLY a JSON array, no other text, no markdown code fences:
[
  {"section_name": "Experience 1", "entry_type": "experience", "title_index": 1, "bullet_indices": [2, 3]}
]
"""


def extract_untailored_text(records: list[dict], sections: list[dict]) -> str:
    """
    Returns the plain text of every paragraph NOT claimed by any
    recognized Experience/Project section — i.e. Skills, Education,
    Summary, Certifications, or anything else group_deterministically()/
    the LLM step skipped over.

    This content is never edited by the pipeline (nothing here gets
    drafted, grounded, or written back to the docx) — it exists purely
    so agents that need to judge the resume AS A WHOLE (the Fit Scorer,
    currently) aren't reasoning off an incomplete picture. A resume
    that lists React/Docker/PostgreSQL/etc. under a Skills header is
    genuinely demonstrating those things even though this pipeline
    never touches that section's wording.
    """
    claimed_indices = set()
    for s in sections:
        claimed_indices.add(s["title_index"])
        claimed_indices.update(s["bullet_indices"])

    leftover = [r["text"] for r in records if r["index"] not in claimed_indices]
    return "\n".join(leftover)


def read_docx_sections(docx_path: str) -> tuple[list[dict], str]:
    """
    Returns (sections, untailored_text):
      sections: a list of section records, each with the REAL extracted
        text (never model-generated text) for the title and bullets:
          {
            "section_name": str,
            "entry_type": "experience" | "project",
            "title_text": str,
            "bullets": [str, ...],
            "bullet_count": int,
          }
      untailored_text: plain text of everything else in the doc (Skills,
        Education, Summary, ...) — see extract_untailored_text() above.
    """
    records = extract_paragraph_structure(docx_path)
    paragraph_dump = format_for_llm(records)

    # Scale with how many paragraphs are in the actual resume — a
    # longer, more detailed resume produces more entries in the output
    # array, same scaling risk we've now hit in three other files.
    token_budget = 800 + (len(records) * 60)
    response_text = call_llm(SYSTEM_PROMPT, paragraph_dump, max_tokens=token_budget, context="docx_sections")

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        raw_sections = json.loads(cleaned)
    except json.JSONDecodeError as e:
        likely_truncated = not cleaned.rstrip().endswith("]")
        hint = (
            " This looks like TRUNCATION (response doesn't end with ']') "
            "— the token_budget may need to be higher for this resume's length."
            if likely_truncated else ""
        )
        raise ValueError(
            f"Model didn't return valid JSON.{hint} Raw response:\n{response_text}"
        ) from e

    # Build a lookup from the ALREADY-EXTRACTED, guaranteed-real text —
    # the model's response is used ONLY for index numbers from here on.
    text_by_index = {r["index"]: r["text"] for r in records}

    sections = []
    for s in raw_sections:
        title_idx = s["title_index"]
        bullet_idxs = s["bullet_indices"]

        if title_idx not in text_by_index:
            raise ValueError(
                f"Model referenced title_index {title_idx}, which doesn't "
                f"exist in the extracted paragraphs."
            )
        for bi in bullet_idxs:
            if bi not in text_by_index:
                raise ValueError(
                    f"Model referenced bullet_index {bi}, which doesn't "
                    f"exist in the extracted paragraphs."
                )

        sections.append({
            "section_name": s["section_name"],
            "entry_type": s["entry_type"],
            "title_index": title_idx,
            "title_text": text_by_index[title_idx],       # real, extracted text
            "bullet_indices": bullet_idxs,
            "bullets": [text_by_index[bi] for bi in bullet_idxs],  # real, extracted text
            "bullet_count": len(bullet_idxs),
        })

    untailored_text = extract_untailored_text(records, sections)
    return sections, untailored_text


# if __name__ == "__main__":
#     import sys

#     path = sys.argv[1] if len(sys.argv) > 1 else "data/test_resume_A_headings.docx"
#     sections, untailored_text = read_docx_sections(path)

#     for s in sections:
#         print(f"\n=== {s['section_name']} ({s['entry_type']}) ===")
#         print(f"Title: {s['title_text']}")
#         print(f"Bullets ({s['bullet_count']}):")
#         for b in s["bullets"]:
#             print(f"  - {b}")