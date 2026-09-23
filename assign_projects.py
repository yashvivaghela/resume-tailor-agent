"""
assign_projects_to_sections — the Project Assignment agent.

For each EXISTING section in the resume (with its current entry and
bullet count), decides whether that entry is still the strongest fit
for the JD, or whether a different entry from the full project/
experience list would serve better. This is a text-only swap (title +
bullets) — never a structural change to the resume — so it stays inside
the formatting-safety boundary from Phase 3's design.

IMPORTANT — what this function does NOT do:
- It does NOT draft or reword any bullets. draft_bullets() (Phase 2)
  runs on the RESULT of this step, unconditionally — for every assigned
  entry, whether it was replaced or kept. Assignment only decides WHICH
  entry fills a slot; the bullets always get freshly tailored to the JD
  regardless, since even a kept project's existing wording may not speak
  to this specific JD.
- `was_replaced` is metadata for two OTHER consumers downstream, not a
  signal about whether to redraft: (1) Phase 3's docx editor needs it to
  know whether it must also swap the section's title/project-name line,
  not just the bullets, and (2) the Fit Scorer's transparency report.

KNOWN GAP, to be resolved in Phase 3: in real usage, `resume_sections`
won't come with entry ids pre-attached — a real resume might say
"Personal Finance Tracker" while the projects doc says "Personal Finance
Dashboard." Phase 3 needs a separate matching step (fuzzy/semantic, not
exact string match) that maps each real resume section's title/bullets
to the corresponding cached entry id BEFORE this function can run on
real data. The test below hand-supplies ids to work around this for now.

If a resume section has no match in the projects doc at all, that does
NOT mean there's no ground truth — the section's OWN current bullets
are real, user-authored source material. Phase 3's matching step should
fall back to extracting those bullets verbatim as this section's
description, rather than treating an unmatched section as untouchable.
"""

import json
import re

from llm_client import call_llm
from schemas import SectionAssignment, JDRequirements
from project_queries import list_projects, get_project_details


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace, so word-level comparisons
    aren't defeated by trivial differences (extra spaces, line breaks,
    case) that don't change whether the phrase is genuinely present."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _evidence_supported(phrase: str, real_text: str, min_overlap: float = 0.6) -> bool:
    """
    Checks that `phrase` is genuinely grounded in `real_text`, WITHOUT
    requiring an exact contiguous substring match. An earlier version
    used `_normalize(phrase) in _normalize(real_text)` — an exact
    substring check — which turned out to be too strict in practice: a
    model can quote faithfully while still rewording punctuation,
    trimming a clause, or lightly paraphrasing, and any of those breaks
    a substring match even though the underlying claim is completely
    real. That false-positive rejection rate is exactly what silently
    turned every genuine swap back into "kept" in production.

    Instead: tokenize both into words (3+ chars, to skip noise words
    like "a"/"to"/"in") and require most of the evidence phrase's OWN
    words to actually appear somewhere in the real text. A genuinely
    fabricated claim (a skill/technology the entry never mentions) will
    share very few words with the real description no matter how it's
    phrased, and will still fail this; a faithful paraphrase of real
    content will pass even with different wording/punctuation.
    """
    phrase_words = [w for w in re.findall(r"[a-z0-9]+", _normalize(phrase)) if len(w) > 2]
    if not phrase_words:
        return False
    real_words = set(re.findall(r"[a-z0-9]+", _normalize(real_text)))
    matched = sum(1 for w in phrase_words if w in real_words)
    return (matched / len(phrase_words)) >= min_overlap

SYSTEM_PROMPT = """You decide which project or experience entry should
fill each section of a resume, given a target job description.

You'll be given:
1. The resume's CURRENT sections — each with a section name, its TYPE
   ("experience" or "project"), the entry currently filling it, and how
   many bullets that section has (this bullet count is FIXED — whatever
   entry you assign must work within that many bullets, you cannot add
   or remove sections)
2. The FULL list of available entries (both work experience and
   projects) the person could draw from — including ones not currently
   in the resume at all
3. The parsed job description requirements

For EACH section, decide: is the CURRENT entry still the strongest
available fit for this JD, or does a DIFFERENT entry from the full list
serve the JD better? You may only assign an entry that isn't already
used in another section — each entry can fill at most one section.

CRITICAL CONSTRAINT: a section's TYPE must never change. An
"experience" section (a real job) may ONLY be filled by another entry
whose type is "experience" — never by a "project." Likewise a "project"
section may ONLY be filled by another "project" entry. Mixing these
would put personal project content under a section literally titled
"Experience," misrepresenting it as paid work history — never do this,
even if a project would otherwise be a stronger content match for the
JD than any available experience entry.

Prefer keeping the current entry unless a genuinely stronger fit exists
— don't replace things just to be different. But don't be shy about
replacing a weak fit either: if a section holds an entry with little
relevance to this JD and a much more relevant entry of the SAME TYPE
exists elsewhere in the list, propose the swap.

IMPORTANT — do not swap one irrelevant entry for another. Only propose
a replacement if the new entry concretely demonstrates at least one
REQUIRED skill/technology from the JD through what was actually built
or done there — not merely "slightly more technical" or "closer to
engineering than the current one." If every same-type candidate is a
weak match, KEEP the current entry (was_replaced: false) and let the
Fit Scorer surface the gap honestly, rather than shuffling to a
different weak entry that creates the appearance of progress without
actually improving the JD match. A swap should make a specific,
namable requirement demonstrably better covered — be ready to name
which one in `reason`.

EVIDENCE REQUIREMENT for any swap (was_replaced: true): you MUST also
give `evidence_phrase` — copy the exact phrase, as close to verbatim
as you can, from that entry's OWN description that demonstrates the
skill you're citing in `reason`. Do not paraphrase, infer, or add
technologies the entry doesn't actually mention just because they'd
make the justification sound better — this is checked word-for-word
against the entry's real text after you respond, and a swap whose
evidence doesn't actually appear in the entry will be reverted. If you
can't point to an exact phrase that proves the claim, that's a sign
the swap isn't justified — keep the current entry instead. When
was_replaced is false, set evidence_phrase to null.

Respond with ONLY a JSON array, one object per section, no other text,
no markdown code fences:
[
  {
    "section_name": "<matches the input section name exactly>",
    "assigned_entry_id": "<id of whichever entry now fills this slot>",
    "was_replaced": true | false,
    "reason": "<short explanation of why this entry was chosen or kept>",
    "evidence_phrase": "<exact phrase from the new entry's real description proving the claim, or null if was_replaced is false>"
  }
]
"""


def assign_projects_to_sections(
    resume_sections: list[dict],
    jd_requirements: JDRequirements,
) -> list[SectionAssignment]:
    """
    resume_sections: e.g.
        [{"section_name": "Project 1", "entry_type": "project",
          "current_entry_id": "proj_002", "bullet_count": 3}, ...]
        (this will come from read_docx_sections() once Phase 3 exists;
        for now it's passed in directly, e.g. from a hand-built test dict)

    entry_type on each section is REQUIRED — it's what enforces that an
    "experience" slot never gets filled by a "project" entry or vice
    versa (see CRITICAL CONSTRAINT in the prompt, and the code-level
    check below that doesn't just trust the prompt).
    """
    candidates = list_projects()  # cheap summaries: id, type, title, one_liner
    candidate_type_by_id = {c["id"]: c["type"] for c in candidates}
    section_type_by_name = {s["section_name"]: s["entry_type"] for s in resume_sections}
    current_entry_by_section = {
        s["section_name"]: s.get("current_entry_id") for s in resume_sections
    }

    # The evidence-phrase check below needs something REAL to check
    # against — a one_liner is already an LLM paraphrase, so quoting
    # "from" it proves nothing. Attach each candidate's verbatim
    # description (from Phase 0's cache, never model-generated) so the
    # model can actually ground a claimed skill in real text, and so
    # the code-level check after the call has real text to check it
    # against.
    full_candidates = [
        {**c, "description": get_project_details(c["id"])["description"]}
        for c in candidates
    ]
    description_by_id = {c["id"]: c["description"] for c in full_candidates}

    user_message = (
        f"CURRENT RESUME SECTIONS:\n{json.dumps(resume_sections, indent=2)}\n\n"
        f"AVAILABLE ENTRIES (experience + projects):\n{json.dumps(full_candidates, indent=2)}\n\n"
        f"JOB REQUIREMENTS:\n{jd_requirements.model_dump_json(indent=2)}"
    )

    # Scale the token budget with how many sections are being assigned
    # — one JSON object per section, so more sections genuinely need
    # more room. A fixed 1500 was enough for our 5-section test set but
    # not for a real resume with more sections and a larger candidate
    # pool (longer per-section reasoning too).
    token_budget = 800 + (len(resume_sections) * 500)
    response_text = call_llm(SYSTEM_PROMPT, user_message, max_tokens=token_budget, context="assign_projects")

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        raw_list = json.loads(cleaned)
    except json.JSONDecodeError as e:
        likely_truncated = not cleaned.rstrip().endswith("]")
        hint = (
            " This looks like TRUNCATION (response doesn't end with ']') "
            "— the token_budget may need to be higher for this many "
            "sections/candidates."
            if likely_truncated else ""
        )
        raise ValueError(
            f"Model didn't return valid JSON.{hint} Raw response:\n{response_text}"
        ) from e

    assignments = [SectionAssignment(**item) for item in raw_list]

    # DESIGN GUARANTEE, not just a convention: SectionAssignment only
    # carries an id, never text content. This is deliberate — it forces
    # whatever calls draft_bullets() next to fetch the FULL entry via
    # get_project_details(id), which returns the verbatim `description`
    # field. It structurally prevents accidentally reusing the one_liner
    # (an LLM paraphrase, used only for THIS function's own scanning/
    # decision-making) as if it were grounded source text. Do not add a
    # `description` or `one_liner` field to SectionAssignment later —
    # that would reopen this exact risk.
    #
    # Validate the "each entry used at most once" rule ourselves — the
    # prompt asks for this, but a prompt instruction is not a guarantee.
    # Fail loudly here rather than silently letting a resume claim the
    # same project twice under two different section headers.
    seen_entry_ids = {}
    for a in assignments:
        if a.assigned_entry_id in seen_entry_ids:
            raise ValueError(
                f"Model assigned entry '{a.assigned_entry_id}' to both "
                f"'{seen_entry_ids[a.assigned_entry_id]}' and "
                f"'{a.section_name}' — an entry can only fill one section."
            )
        seen_entry_ids[a.assigned_entry_id] = a.section_name

    # Enforce the type constraint ourselves too — don't just trust the
    # prompt. An "experience" section must be filled by an "experience"
    # entry, never a "project" (and vice versa), or the resume would
    # show project work under a heading that implies paid employment.
    for a in assignments:
        expected_type = section_type_by_name.get(a.section_name)
        actual_type = candidate_type_by_id.get(a.assigned_entry_id)
        if expected_type is not None and actual_type != expected_type:
            raise ValueError(
                f"Type mismatch: section '{a.section_name}' is type "
                f"'{expected_type}' but was assigned entry "
                f"'{a.assigned_entry_id}' (type '{actual_type}'). "
                f"An experience slot can never be filled by a project, "
                f"or vice versa."
            )

    # Verify every proposed swap's evidence against the entry's REAL
    # text — a plausible-sounding `reason` is not proof, and the model
    # has already been observed inventing a supporting detail (e.g.
    # claiming "Python" for an entry whose real text never mentions it)
    # to satisfy its own justification requirement. Trust the entry's
    # verbatim description (from Phase 0's cache), not the model's
    # claim about it. A swap that fails this check is reverted to
    # keeping the section's original entry, rather than raising —
    # a single bad swap shouldn't fail the whole assignment step when
    # "just keep what was there" is always a safe fallback.
    verified_assignments = []
    for a in assignments:
        if not a.was_replaced:
            verified_assignments.append(a)
            continue

        real_text = description_by_id.get(a.assigned_entry_id, "")
        phrase = (a.evidence_phrase or "").strip()

        evidence_checks_out = bool(phrase) and _evidence_supported(phrase, real_text)

        if evidence_checks_out:
            verified_assignments.append(a)
        else:
            fallback_id = current_entry_by_section.get(a.section_name)
            if fallback_id is None:
                raise ValueError(
                    f"Swap for '{a.section_name}' failed evidence "
                    f"verification, but no current_entry_id was provided "
                    f"for that section to fall back to. Every section in "
                    f"resume_sections must include current_entry_id."
                )
            print(
                f"[assign_projects] REJECTED swap for '{a.section_name}': "
                f"claimed evidence_phrase {phrase!r} was not sufficiently "
                f"supported by the real description of entry "
                f"'{a.assigned_entry_id}' (real text: {real_text[:200]!r}...). "
                f"Reverting to current entry '{fallback_id}'. "
                f"(Model's stated reason was: {a.reason!r})"
            )
            verified_assignments.append(
                SectionAssignment(
                    section_name=a.section_name,
                    assigned_entry_id=fallback_id,
                    was_replaced=False,
                    reason=(
                        "Kept current entry — a proposed swap was rejected "
                        "because its supporting evidence didn't check out "
                        "against the candidate entry's real description."
                    ),
                    evidence_phrase=None,
                )
            )

    return verified_assignments


# if __name__ == "__main__":
#     from parse_jd import parse_jd

#     # Fake resume structure — stands in for what read_docx_sections()
#     # will provide once Phase 3 exists. Deliberately puts a weak-fit
#     # project (the weather bot) in a slot, to test whether the model
#     # actually proposes swapping it out for something more relevant.
#     fake_resume_sections = [
#         {"section_name": "Project 1", "entry_type": "project", "current_entry_id": "proj_003", "bullet_count": 2},
#         {"section_name": "Project 2", "entry_type": "project", "current_entry_id": "proj_002", "bullet_count": 3},
#     ]

#     sample_jd = """
#     Senior Backend Engineer

#     Requirements:
#     - 5+ years of experience with Python
#     - Experience designing and building REST APIs at scale
#     - Familiarity with automated testing and CI/CD practices
#     - Bachelor's degree in Computer Science or equivalent experience
#     """

#     jd = parse_jd(sample_jd)
#     assignments = assign_projects_to_sections(fake_resume_sections, jd)

#     for a in assignments:
#         print(a.model_dump_json(indent=2))