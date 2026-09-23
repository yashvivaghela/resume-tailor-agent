"""
fit_scorer.py — the Fit Scorer, the 3rd agent in the system.

Unlike the Grounding Critic (which checks one bullet's truthfulness) or
the Executor (which plans and drafts), this agent makes a genuinely
different kind of judgment: reading the ENTIRE finished resume against
the ENTIRE JD and reporting how good a fit it actually is — including
saying so when the fit is weak. This is deliberately NOT another
"is this claim true" check (that's the critic's job, already done by
the time this runs) — it's "even if every claim here is true, does this
resume actually make a compelling case for this specific role?"

The core risk this prompt is written to resist: an LLM asked to score
something the user clearly wants scored well tends toward flattery. The
prompt actively pushes against that.
"""

import json

from llm_client import call_llm
from schemas import FitScore, JDRequirements

SYSTEM_PROMPT = """You are an honest, slightly skeptical hiring manager
reviewing a candidate's tailored resume against a specific job's
requirements. Your job is to give a CANDID assessment — not an
encouraging one. The candidate needs the truth, not reassurance.

You will be given the final resume content (all section titles and
their bullets) and the structured job requirements.

Score based on:
- How many of the REQUIRED skills/experience are genuinely, concretely
  demonstrated in the resume content — actually shown through what was
  built/done, not just present as a keyword somewhere
- Whether the seniority level and years of experience implied by the
  resume plausibly match what's required
- Whether the education requirement (if any) is addressed at all —
  note if it's simply never mentioned, don't assume it's satisfied
- Nice-to-have skills count, but far less than required ones

DO NOT be encouraging by default. A resume that only weakly covers the
JD should receive a LOW score and a blunt honest_note explaining why —
not a softened "pretty good fit!" verdict. If you notice yourself
wanting to round a mediocre fit up to sound more positive, don't —
report what's actually there. A low score here is useful information
for the candidate, not a failure of the resume.

Use THREE tiers, not two, for how a required item is covered:
- strong_areas: genuinely demonstrated — a bullet/description actually
  shows the skill being used (built, implemented, led, etc.)
- weak_areas: EITHER thinly demonstrated, OR present only as a bare
  keyword with no supporting bullet anywhere (e.g. a skill that only
  appears in a Skills/Tools list, never mentioned in any project or
  experience description). Label which case it is, e.g. "Django —
  listed in skills, not demonstrated in any project."
- unaddressed_requirements: the item does not appear ANYWHERE in the
  resume content — not in a bullet, not as a keyword, not at all.

This distinction matters: a skill the candidate has bothered to list
is real signal, worth less than a demonstrated skill but not worth
treating as absent. Only use unaddressed_requirements for genuine
silence on a requirement.

Respond with ONLY a JSON object matching this exact shape, no other
text, no markdown code fences:
{
  "overall_score": "<e.g. '6/10'>",
  "strong_areas": ["<requirement genuinely well-covered>", ...],
  "weak_areas": ["<requirement covered thinly>", ...],
  "unaddressed_requirements": ["<required item with no real evidence>", ...],
  "honest_note": "<a candid 1-3 sentence summary, blunt if the fit is weak>"
}
"""


def score_fit(final_resume_text: str, jd_requirements: JDRequirements) -> FitScore:
    user_message = (
        f"FINAL RESUME CONTENT:\n{final_resume_text}\n\n"
        f"JOB REQUIREMENTS:\n{jd_requirements.model_dump_json(indent=2)}"
    )

    # 1000 was enough for a short JD with few requirements, but a JD
    # with many distinct requirements (common for real postings) makes
    # strong_areas/weak_areas/unaddressed_requirements genuinely long,
    # and the old fixed budget truncated the JSON before it could close.
    response_text = call_llm(SYSTEM_PROMPT, user_message, max_tokens=1800, context="fit_scorer")

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # A response that doesn't end with "}" almost always means it
        # got cut off by max_tokens mid-generation, not that the model
        # wrote malformed JSON — surface that distinction directly.
        likely_truncated = not cleaned.rstrip().endswith("}")
        hint = (
            " This looks like TRUNCATION (response doesn't end with '}') "
            "— the max_tokens budget may need to be higher, not a "
            "prompt/formatting problem."
            if likely_truncated else ""
        )
        raise ValueError(
            f"Model didn't return valid JSON.{hint} Raw response:\n{response_text}"
        ) from e

    return FitScore(**raw)


def format_resume_text(results: list[dict], untailored_text: str = "") -> str:
    """
    Turns run_executor()'s results into plain text the scorer can read
    — just titles and final bullets, no internal metadata (assignment
    reasons, gap details) that isn't actually part of the resume.

    untailored_text: everything else in the resume this pipeline never
    edits (Skills, Education, Summary, ...) — see
    docx_sections.extract_untailored_text(). Without this, the scorer
    only ever sees the Experience/Project sections and will wrongly
    call out a required skill as "unaddressed" even when it's sitting
    right there in a Skills section, because that section is never
    part of `results` at all. Optional/empty for the fake-resume test
    path (run_executor() called directly, no real docx behind it).
    """
    lines = []
    for r in results:
        lines.append(f"\n{r['section_name']}:")
        for b in r["final_bullets"]:
            lines.append(f"  - {b}")

    if untailored_text.strip():
        lines.append(
            "\n\nOTHER RESUME CONTENT (not tailored by this tool, shown "
            "here only so you can judge the resume as a whole — e.g. "
            "Skills, Education, Summary):"
        )
        lines.append(untailored_text)

    return "\n".join(lines)


# if __name__ == "__main__":
#     from parse_jd import parse_jd

#     # Deliberately weak-fit test case: a JD asking for things the
#     # sample resume content genuinely doesn't demonstrate, to confirm
#     # the scorer actually reports a low score rather than flattering it
#     fake_resume_text = """
# Project 1 (Self-Debugging Coding Agent):
#   - Engineered a Python debugging agent that automatically fixes failing pytest unit tests.
#   - Automated the unit-test debugging workflow using the Anthropic API.

# Project 2 (Personal Finance Dashboard):
#   - Implemented a Flask-based backend in Python for CSV transaction processing.
#   - Designed JWT-protected REST endpoints for 200 beta users.
# """

#     demanding_jd = """
#     Staff Machine Learning Engineer

#     Requirements:
#     - 8+ years of experience building production ML systems at scale
#     - Deep expertise in distributed training (PyTorch, multi-GPU/TPU)
#     - Experience leading a team of 5+ engineers
#     - PhD in Machine Learning, Computer Science, or related field
#     - Track record of published research or patents
#     """

#     jd = parse_jd(demanding_jd)
#     result = score_fit(fake_resume_text, jd)
#     print(result.model_dump_json(indent=2))