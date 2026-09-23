"""
match_sections_to_entries.py — maps each section actually found in a
real resume (via docx_sections.py) to the corresponding cached
project/experience entry, using semantic matching rather than exact
text comparison (a resume might say "Personal Finance Tracker" while
the projects doc says "Personal Finance Dashboard" — same thing,
different wording).

KNOWN SIMPLIFICATION: if a section genuinely has no match in the cached
data, this returns current_entry_id=None for it, and the Executor
integration currently leaves that section's paragraphs completely
untouched rather than treating the resume's own existing bullets as a
fallback ground truth (which was the fuller design intent — see
assign_projects.py's docstring). Revisit this as a fast-follow if
unmatched sections turn out to be common in practice.
"""

import json

from llm_client import call_llm
from project_queries import list_projects

SYSTEM_PROMPT = """You match sections from a real resume to entries in
a separate, structured list of the person's projects/experience. The
wording often won't match exactly — a resume might say "Personal
Finance Tracker" while the structured list says "Personal Finance
Dashboard." Match by MEANING (same underlying project/job), not exact
text.

For each resume section, decide which candidate entry (if any) it
refers to. If a section genuinely doesn't correspond to anything in the
candidate list, its match is null — do not force a weak or unrelated
match just to fill in an answer.

Respond with ONLY a JSON array, no other text, no markdown code fences:
[
  {"section_name": "<matches the input section_name exactly>", "matched_entry_id": "<id or null>"}
]
"""


def match_sections_to_entries(doc_sections: list[dict]) -> dict[str, str | None]:
    """
    doc_sections: output of read_docx_sections() — each with
                  section_name, title_text, bullets.
    Returns: {section_name: matched_entry_id_or_None}
    """
    candidates = list_projects()  # id, type, title, one_liner

    sections_summary = [
        {
            "section_name": s["section_name"],
            "title_text": s["title_text"],
            "bullets": s["bullets"],
        }
        for s in doc_sections
    ]

    user_message = (
        f"RESUME SECTIONS:\n{json.dumps(sections_summary, indent=2)}\n\n"
        f"CANDIDATE ENTRIES:\n{json.dumps(candidates, indent=2)}"
    )

    # 1000 was too tight when there are many resume sections/candidates
    # to match — bumped preemptively, same pattern seen elsewhere.
    response_text = call_llm(SYSTEM_PROMPT, user_message, max_tokens=1500, context="match_sections_to_entries")

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError as e:
        likely_truncated = not cleaned.rstrip().endswith("]")
        hint = (
            " This looks like TRUNCATION (response doesn't end with ']') "
            "— the max_tokens budget may need to be higher."
            if likely_truncated else ""
        )
        raise ValueError(
            f"Model didn't return valid JSON.{hint} Raw response:\n{response_text}"
        ) from e

    return {item["section_name"]: item["matched_entry_id"] for item in raw}


# if __name__ == "__main__":
#     import sys
#     from docx_sections import read_docx_sections

#     path = sys.argv[1] if len(sys.argv) > 1 else "data/test_resume_A_headings.docx"
#     doc_sections = read_docx_sections(path)
#     matches = match_sections_to_entries(doc_sections)

#     for section_name, entry_id in matches.items():
#         print(f"{section_name} -> {entry_id}")