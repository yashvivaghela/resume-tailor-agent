"""
docx_writer.py — writes new text into an existing resume, safely.

Core safety guarantee: this NEVER adds or removes a PARAGRAPH. Bullet
edits additionally never add/remove a RUN either (replace_paragraph_text
below) — because a bullet is always "the same content, reworded," so
its run structure should never need to change.

A title line is different: which OPTIONAL segments it shows (a
tech-stack list, a link) can legitimately differ from one project to
another, so render_title_line() below IS allowed to add/remove runs
and hyperlink elements within a title paragraph — see its docstring.
This does not weaken the paragraph-count guarantee: adding a run inside
an existing paragraph doesn't change how many paragraphs exist, so
write_tailored_docx()'s structural check further down is unaffected.
"""

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import re

_BOLD_MARKER_RE = re.compile(r"\*\*(.+?)\*\*")


def _strip_bold_markers(text: str) -> str:
    """Removes **markers** but keeps the enclosed text — used when this
    resume's convention does NOT bold inline terms, so a marker that
    leaked through anyway (a model slip) never shows up literally."""
    return _BOLD_MARKER_RE.sub(lambda m: m.group(1), text)


def render_bullet_runs(paragraph, marked_text: str) -> None:
    """
    Writes `marked_text` into `paragraph`, splitting it into multiple
    runs wherever a **bold** marker appears — used ONLY when this
    resume's own convention already bolds specific terms inline within
    bullets (style_profile.detect_bullet_bold_style() returned True).
    This is the one exception to bullets' normal "text only, never
    structure" rule (see module docstring) — justified the same way
    render_title_line()'s exception is: matching a resume's OWN
    demonstrated convention, never inventing a new one, and adding runs
    within an existing paragraph doesn't affect the paragraph-count
    guarantee either way.

    Segments reuse the paragraph's EXISTING runs first (for formatting
    continuity), then append new runs — copying run 0's font name/size
    — for anything beyond the original run count.
    """
    segments = []
    last_end = 0
    for m in _BOLD_MARKER_RE.finditer(marked_text):
        if m.start() > last_end:
            segments.append((marked_text[last_end:m.start()], False))
        segments.append((m.group(1), True))
        last_end = m.end()
    if last_end < len(marked_text):
        segments.append((marked_text[last_end:], False))
    if not segments:
        segments = [(marked_text, False)]

    existing_runs = list(paragraph.runs)
    base_font = existing_runs[0].font if existing_runs else None

    for i, (text, is_bold) in enumerate(segments):
        if i < len(existing_runs):
            run = existing_runs[i]
            run.text = text
            run.bold = is_bold
        else:
            new_run = paragraph.add_run(text)
            new_run.bold = is_bold
            if base_font is not None:
                new_run.font.name = base_font.name
                new_run.font.size = base_font.size

    for run in existing_runs[len(segments):]:
        run.text = ""


def replace_paragraph_text(paragraph, new_text: str) -> None:
    """
    Replaces a paragraph's visible text with new_text, in place.

    If the paragraph has multiple runs (e.g. part bold, part not), the
    new text goes entirely into the FIRST run — which keeps that run's
    formatting (font, size, bold, color) — and every other run's text
    is cleared to "" rather than deleted. This means: run count is
    unchanged, paragraph count is unchanged, only visible characters
    change. Mixed intra-paragraph formatting (e.g. one bolded word
    inside an otherwise normal sentence) is not preserved in that case
    — the whole new text takes on run 0's formatting — which is an
    acceptable, honest tradeoff for a resume bullet, where formatting
    is normally uniform across the whole line anyway.
    """
    if not paragraph.runs:
        # No runs at all (rare — an empty or field-only paragraph).
        # Nothing safe to edit; leave it alone rather than guessing.
        return

    paragraph.runs[0].text = new_text
    for run in paragraph.runs[1:]:
        run.text = ""


def _remove_hyperlinks(paragraph) -> None:
    """
    Removes every <w:hyperlink> element from a paragraph ENTIRELY (the
    whole element, not just its visible text) — used when this slot's
    new project has no known link but the paragraph originally held one
    belonging to a DIFFERENT project. Blanking the hyperlink run's text
    (the old approach) would leave a dangling empty underlined stub;
    removing the element is a clean, honest "no link here."

    The relationship this hyperlink pointed to is left in the document
    part, unused — both python-docx and Word tolerate an orphaned
    relationship silently, so this is safe to leave rather than also
    hunting down and removing the relationship itself.
    """
    for node in list(paragraph._p.iter()):
        if node.tag.endswith("}hyperlink"):
            node.getparent().remove(node)


def _add_hyperlink(paragraph, display_text: str, url: str) -> None:
    """
    Appends a new hyperlink run to the END of a paragraph. This is
    additive only — it never touches any existing run — so it can't
    corrupt whatever title text render_title_line() already placed in
    run 0 before calling this.
    """
    part = paragraph.part
    r_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )

    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    new_run = OxmlElement("w:r")
    run_props = OxmlElement("w:rPr")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    run_props.append(underline)
    new_run.append(run_props)

    text_el = OxmlElement("w:t")
    text_el.text = display_text
    new_run.append(text_el)

    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)


def render_title_line(paragraph, title: str, skills: list[str], link: str | None, style: dict) -> None:
    """
    Rewrites a project/experience TITLE paragraph to show `title`, plus
    whichever OPTIONAL segments this specific resume's own convention
    actually uses — determined by `style` (from
    style_profile.detect_title_line_style()), never hardcoded:

    - tech-stack segment: added ONLY if style["has_tech_stack"] is True
      (i.e. most of this resume's OTHER title lines already show one).
      Built from `skills` — already-verified real data, never invented
      — bracketed in whatever style this resume uses
      (style["bracket_style"]).
    - link segment: added ONLY if style["has_link"] is True AND `link`
      is not None. An entry with no real link (parse_projects.py found
      none in the source doc) NEVER gets one forced on just because
      other projects in this resume happen to have one — that would
      misrepresent something that doesn't exist. Conversely, if this
      resume's own convention doesn't use links at all
      (style["has_link"] is False), a link is never added even if this
      particular entry happens to have one on file — matching THIS
      resume's format takes priority.

    Any hyperlink the paragraph originally had is removed first,
    unconditionally — a stale link belonging to whatever project used
    to occupy this slot must never survive under the new title.

    KNOWN LIMITATION: the separator between segments is currently a
    fixed " | " and stack entries are joined with ", " — this covers
    the common case but doesn't yet detect a resume's own separator
    style (e.g. " — " instead of " | "). Good enough for now; a more
    thorough version would extract this from the same untouched title
    lines style_profile.py already scans.
    """
    _remove_hyperlinks(paragraph)

    if not paragraph.runs:
        paragraph.add_run(title)
    else:
        paragraph.runs[0].text = title
        for run in paragraph.runs[1:]:
            run.text = ""

    if style.get("has_tech_stack") and skills:
        open_b, close_b = ("[", "]") if style.get("bracket_style") == "[]" else ("(", ")")
        stack_run = paragraph.add_run(f" {open_b}{', '.join(skills)}{close_b}")
        stack_run.italic = True

    if style.get("has_link") and link:
        paragraph.add_run(" | ")
        _add_hyperlink(paragraph, "Link", link)


def write_tailored_docx(
    source_path: str,
    output_path: str,
    edits: dict[int, str],
    title_renders: dict[int, dict] | None = None,
    bullet_bold: bool = False,
) -> None:
    """
    ... (see other params above)
    title_renders: {paragraph_index: {"title": str, "skills": [...],
                 "link": str | None, "style": dict}} — for a title line
                 whose PROJECT changed (replaced/recovered). "style" is
                 THIS section's own entry-type style profile (e.g.
                 style_by_type["project"] vs style_by_type["experience"]
                 from style_profile.detect_title_line_style()) — passed
                 per-entry rather than once globally, since a resume's
                 Experience and Project title lines routinely follow
                 different conventions (see that function's docstring).
    bullet_bold: whether THIS resume's own convention bolds specific
                 terms inline within bullets (from
                 style_profile.detect_bullet_bold_style()). When True,
                 every entry in `edits` is written via
                 render_bullet_runs() so **marker**-wrapped text (from
                 draft_bullets(), see its docstring) becomes real bold
                 runs. When False, any stray ** marker is stripped back
                 to plain text defensively, so a slip never surfaces as
                 literal asterisks in a resume that doesn't use bolding.
    """
    doc = Document(source_path)
    original_paragraph_count = len(doc.paragraphs)

    for index, new_text in edits.items():
        if index >= len(doc.paragraphs):
            raise ValueError(
                f"Edit references paragraph index {index}, but the "
                f"document only has {len(doc.paragraphs)} paragraphs."
            )
        if bullet_bold:
            render_bullet_runs(doc.paragraphs[index], new_text)
        else:
            replace_paragraph_text(doc.paragraphs[index], _strip_bold_markers(new_text))

    for index, seg in (title_renders or {}).items():
        if index >= len(doc.paragraphs):
            raise ValueError(
                f"Title render references paragraph index {index}, but "
                f"the document only has {len(doc.paragraphs)} paragraphs."
            )
        render_title_line(
            doc.paragraphs[index], seg["title"], seg.get("skills", []),
            seg.get("link"), seg.get("style", {}),
        )

    # Verify the structural guarantee held — this should be impossible
    # to violate given neither replace_paragraph_text() nor
    # render_title_line() ever add/remove a PARAGRAPH (render_title_line
    # may add runs/hyperlinks WITHIN a paragraph, which doesn't affect
    # this count) — checking costs nothing and catches any future
    # change to this file that might accidentally break that invariant.
    if len(doc.paragraphs) != original_paragraph_count:
        raise RuntimeError(
            f"Paragraph count changed during editing "
            f"({original_paragraph_count} -> {len(doc.paragraphs)}). "
            f"Aborting save — this should never happen."
        )

    doc.save(output_path)


# if __name__ == "__main__":
#     from docx_reader import extract_paragraph_structure

#     source = "data/test_resume_A_headings.docx"
#     output = "data/test_resume_A_TAILORED.docx"

#     # Confirm original paragraph 5 (a bullet under Personal Finance
#     # Dashboard) before editing, so we have something to compare against
#     before = extract_paragraph_structure(source)
#     print("Paragraph 5 BEFORE edit:")
#     print(f"  text: {before[5]['text']!r}")

#     # Edit just that one bullet
#     write_tailored_docx(source, output, {5: "Rewrote this bullet to prove the edit worked."})

#     after = extract_paragraph_structure(output)
#     print("\nParagraph 5 AFTER edit (in the OUTPUT file):")
#     print(f"  text: {after[5]['text']!r}")

#     print(f"\nParagraph count unchanged: {len(before) == len(after)}")

#     # Confirm every OTHER paragraph is untouched
#     untouched_ok = all(
#         before[i]["text"] == after[i]["text"]
#         for i in range(len(before)) if i != 5
#     )
#     print(f"All other paragraphs untouched: {untouched_ok}")

#     # Confirm the ORIGINAL file was never modified
#     original_still_intact = extract_paragraph_structure(source)[5]["text"] == before[5]["text"]
#     print(f"Original source file left untouched: {original_still_intact}")