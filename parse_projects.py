"""
Turns a raw, informally-written career document (work experience AND
projects, under their own headings) into structured records the rest of
the system can query.

This is a one-shot LLM call (not an agent — no tools, no loop, no
multi-step decisions) but it's a real structuring task: the model has
to decide where one entry ends and the next begins, since your doc
won't perfectly match a schema.
"""

import json
import re

from llm_client import call_llm
from file_parsing import extract_text, hash_file_bytes
from cache import get_cached_projects_if_unchanged, save_cache

# Matches a URL, optionally wrapped in the "(url)" form file_parsing.py
# appends after a hyperlink's visible text (see its docstring). Link
# extraction is done here with a plain regex over the entry's OWN real
# extracted text — never via the LLM — so it's structurally impossible
# to hallucinate a URL: either the real text contains one, or it doesn't.
_URL_RE = re.compile(r"https?://[^\s()]+")

SYSTEM_PROMPT = """You identify entry BOUNDARIES in a raw, informally-
written career document. The document has an "Experience" section (paid
jobs/roles) and a "Projects" section (personal/side projects), each
under its own heading. You do NOT write or copy any entry's description
text yourself — you only identify which LINE NUMBERS each entry's
description spans, so the exact original text can be extracted
programmatically. This guarantees the description is a true verbatim
copy, not a paraphrase.

The document may use inconsistent formatting — headings, line breaks,
bullet points, or just paragraphs. Use your judgment to determine where
one entry's content ends and the next begins, and which section
(Experience vs Projects) each entry belongs to.

The document will be given to you with line numbers prefixed, like:
0: Experience
1:
2: Software Engineer, Acme Corp
3: Built internal tools for...
...
10: Projects
11:
12: Self-Debugging Coding Agent
13: I built a Python tool that...

For EACH entry you identify (whether from Experience or Projects),
extract:
- id: a short slug, e.g. "exp_001" for experience entries, "proj_001"
  for project entries (number each type separately, in order)
- type: "experience" or "project" — based on which section it's under
- title: the entry's name/title (for experience: the job title; for
  projects: the project name). You may write this yourself, it's just
  a label.
- company: the employer name, ONLY for type "experience". Omit or use
  an empty string for type "project".
- skills: a list of technologies/skills actually mentioned for this entry
- start_line: the line number where this entry's descriptive text begins
  (usually right after the title line, or the title line itself if
  there's no separate heading)
- end_line: the line number where this entry's descriptive text ends
  (inclusive) — the line just before the next entry's title/heading, or
  the last line of the document for the final entry
- metrics: any concrete numbers/outcomes mentioned (if none are stated,
  use an empty string — do NOT invent a metric)
- one_liner: a single sentence summary IN YOUR OWN WORDS, for quick
  scanning only

Do NOT include the section heading line itself ("Experience" or
"Projects") within any entry's start_line/end_line range.

Respond with ONLY a JSON array of these records, no other text, no
markdown code fences.
"""


def parse_projects_doc(file_path: str) -> dict:
    """
    Main entry point. Handles the hash check + cache lookup, and only
    calls the LLM if the file content actually changed.
    Returns a dict keyed by project id, e.g. {"proj_001": {...}, ...}
    """
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    current_hash = hash_file_bytes(file_bytes)

    cached = get_cached_projects_if_unchanged(current_hash)
    if cached is not None:
        print(f"[cache hit] Using cached projects (hash {current_hash[:8]}...)")
        return cached

    print(f"[cache miss] Parsing projects doc fresh (hash {current_hash[:8]}...)")
    raw_text = extract_text(file_path)
    projects = _structure_projects(raw_text)

    save_cache(current_hash, projects)
    return projects


def _structure_projects(raw_text: str) -> list[dict]:
    lines = raw_text.split("\n")

    # Send the model a line-numbered version so it can reference exact
    # line indices — this is what makes boundary detection possible
    numbered_text = "\n".join(f"{i}: {line}" for i, line in enumerate(lines))

    response_text = call_llm(SYSTEM_PROMPT, numbered_text, max_tokens=3000, context="parse_projects_doc")

    # Models sometimes wrap JSON in code fences despite instructions —
    # strip that defensively rather than trusting compliance
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        boundary_records = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model didn't return valid JSON. Raw response:\n{response_text}"
        ) from e

    # Keyed by id, not a list — this makes get_project_details() an O(1)
    # dict lookup instead of a linear scan, and it's a more natural fit
    # for JSON (a JSON object IS a hash map already).
    projects = {}
    for record in boundary_records:
        start = record["start_line"]
        end = record["end_line"]

        # Defensive bounds-checking — if the model returns an out-of-range
        # or inverted line range, don't silently produce garbage/empty text
        if start < 0 or end >= len(lines) or start > end:
            raise ValueError(
                f"Project '{record.get('title')}' has invalid line range "
                f"({start}-{end}) for a document with {len(lines)} lines."
            )

        # Join with spaces (not newlines) since the line breaks in the
        # source doc are just wrapping artifacts, not meaningful
        # structure. Also drop any blank lines within the range (e.g. a
        # blank line separating paragraphs). This means the result is no
        # longer a literal substring of the raw file (exact newlines are
        # gone), but it IS still every original word, unchanged and
        # unparaphrased — just with whitespace normalized into one clean
        # paragraph, which is also easier for the grounding critic to
        # read later.
        raw_slice = lines[start:end + 1]
        joined_raw = " ".join(line.strip() for line in raw_slice if line.strip())

        # Pull out a real URL if this entry's own real text contains one
        # (file_parsing.py inlines hyperlink targets as "text (url)" —
        # see its docstring). Regex over guaranteed-real text, not an
        # LLM guess, so this can never invent a link. Only the FIRST URL
        # found is kept — an entry with a GitHub link AND a live-demo
        # link would need a richer schema than "one link"; out of scope
        # for now, but this at least never fabricates the missing one.
        url_match = _URL_RE.search(joined_raw)
        link = url_match.group(0).rstrip(").,;") if url_match else None

        # Strip the "(url)" annotation back out of the description text
        # itself — it's real, but it's link metadata, not part of what
        # the project actually did, and would otherwise leak into
        # drafted bullets as noise (e.g. "...using React (https://...).")
        verbatim_description = _URL_RE.sub("", joined_raw)
        verbatim_description = re.sub(r"\(\s*\)", "", verbatim_description).strip()

        entry_id = record["id"]
        projects[entry_id] = {
            "type": record["type"],           # "experience" or "project"
            "title": record["title"],
            "company": record.get("company", ""),  # only meaningful for experience
            "skills": record["skills"],
            "description": verbatim_description,  # guaranteed exact words
            "metrics": record["metrics"],
            "one_liner": record["one_liner"],
            "link": link,  # real URL from the source doc, or None — never invented
        }

    return projects


# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) < 2:
#         print("Usage: python parse_projects.py <path_to_projects_file>")
#         sys.exit(1)

#     result = parse_projects_doc(sys.argv[1])
#     print(json.dumps(result, indent=2))