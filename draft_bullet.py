"""
draft_bullets — writes resume bullets targeting JD requirements, using
ONLY the given project's data as source material.

ONE function handles both situations, since a "retry" is really just
the general case with max_count=1 and guidance filled in:
  - First pass for a whole section: guidance=None, target_requirements
    has multiple items, max_count is the section's bullet slot count.
    Writes multiple DISTINCT bullets in ONE call, so the model can see
    its own other bullets and avoid repeating the same claim.
  - Guided retry for one rejected bullet: guidance=the critic's
    suggested_fix, target_requirements has exactly ONE item, max_count=1.
"""

import json
import re

from llm_client import call_llm

# Matches a leading list marker the model might still add despite being
# told not to (a dash, bullet char, or asterisk, optionally followed by
# more whitespace) — stripped defensively so a prompt violation can
# never surface as a literal "- " or "• " baked into the resume text
# itself, on top of the document's own bullet formatting.
_LEADING_MARKER_RE = re.compile(r"^[\-\u2022\*]\s+")
_BOLD_MARKER_RE = re.compile(r"\*\*(.+?)\*\*")


def _clean_bullet_text(raw: str, keep_bold_markers: bool) -> str:
    text = _LEADING_MARKER_RE.sub("", raw.strip().strip('"'))
    if not keep_bold_markers:
        # This resume doesn't bold inline terms — strip any ** the
        # model added anyway (a prompt slip), keeping the enclosed
        # text, so a stray marker never surfaces as literal asterisks.
        text = _BOLD_MARKER_RE.sub(lambda m: m.group(1), text)
    return text

BULLET_STYLE_GUIDE = """Resume bullet writing principles (apply with
judgment, not as a checklist to mechanically satisfy):

- Lead with a strong, specific action verb (e.g. "Architected", "Built",
  "Automated", "Reduced", "Designed") — avoid weak openers like "Worked
  on", "Helped with", "Responsible for". Vary the verb naturally based
  on what actually happened; don't force a dramatic verb onto something
  small, and don't reuse the same verb across multiple bullets for the
  same project if you can help it.
- Favor an impact-first structure where the source supports it: what you
  built/did, and what result it had — not just a list of technologies
  used. "Built X, resulting in Y" reads stronger than "Used X and Y and
  Z technologies."
- Be concrete and specific rather than generic wherever the source data
  allows it (name the actual technology, the actual scale, the actual
  outcome) — but NEVER invent specificity the source doesn't support.
  Vague-but-true beats specific-but-fabricated, always.
- No first person ("I", "my", "our"), no filler adjectives that carry no
  information ("dynamic", "cutting-edge", "innovative", "passionate").
- One sentence per bullet, past tense.
- Do NOT prefix the bullet with a dash, bullet character, or any list
  marker ("-", "•", "*", "1)", etc.) — the document already applies its
  own bullet formatting to whatever text you write. `bullet` should be
  the sentence itself, starting directly with the action verb.
"""

SYSTEM_PROMPT = f"""You write resume bullet points for a single project,
each speaking to a specific job requirement, using ONLY the PROJECT DATA
given — never information from outside it.

{BULLET_STYLE_GUIDE}

Rules:
- Use ONLY the information given in PROJECT DATA below. Do not invent
  technologies, scale, outcomes, or numbers that aren't there.
- Write AT MOST max_count bullets — write FEWER if the project genuinely
  doesn't support that many distinct, well-evidenced angles. Do NOT pad
  the list with a weaker rephrasing of a bullet you already wrote just
  to hit the count.
- EXCEPTION: if "MUST_FILL_ALL_SLOTS: true" appears below, this section
  is REPLACING a different project entirely, and every slot must
  describe THIS project — leaving a slot unfilled would leave stale
  content from the old, unrelated project behind, which is worse than a
  general-but-true bullet. In that case, write EXACTLY max_count
  bullets. It's fine for some to be general strengths of the project
  not tied to any specific listed requirement (use
  targets_requirement: "general") — general-but-true is fine, but
  fabricated specifics are still never acceptable.
- If writing more than one bullet, each must be genuinely distinct
  (different requirement, different angle, different specific detail),
  and vary your opening verbs across the set.
- If FACT-CHECKER FEEDBACK is provided below, you are revising a single
  previously-rejected bullet, not writing fresh ones: rewrite it using
  ONLY what the fact-checker confirmed is actually supported by the
  project data. Do not reintroduce the same issue, and do not invent a
  new unverifiable claim to replace the old one. If the project
  genuinely doesn't support strong evidence for this requirement, write
  a more modest, accurate bullet rather than stretching again.

Respond with ONLY a JSON array of objects, no other text, no markdown
code fences. Each object has "bullet" and "targets_requirement" (which
of the listed job requirements this bullet speaks to — copy it exactly
as given, or write "general" if the bullet is a general strength not
tied to one specific listed requirement):
[
  {{"bullet": "Bullet one.", "targets_requirement": "<matching requirement or 'general'>"}},
  {{"bullet": "Bullet two.", "targets_requirement": "<matching requirement or 'general'>"}}
]
"""


def draft_bullets(
    project: dict,
    target_requirements: list[str],
    max_count: int,
    guidance: str | None = None,
    require_exact_count: bool = False,
    use_inline_bold: bool = False,
) -> list[dict]:
    """
    project: a single project/experience record, e.g. {"title": ...,
             "description": ..., "skills": [...], "metrics": ...}
    target_requirements: JD requirements this project's bullets should
             speak to. For a guided retry, pass exactly ONE requirement
             (the one the rejected bullet was targeting).
    max_count: ceiling on how many bullets to write. For a guided retry,
             this should be 1 — you're fixing one bullet, not writing more.
    guidance: pass the critic's suggested_fix here for a guided retry.
             Leave as None for a normal first-pass draft.
    require_exact_count: set True when this section's project is being
             REPLACED (not just re-tailored) — every slot MUST describe
             the new project, since leaving a slot unfilled would leave
             the OLD project's stale content behind after editing.
             Leave False for the normal "write fewer if evidence is
             thin" behavior, which is fine when the project ISN'T
             changing (an untouched slot still describes the same
             project either way).
    use_inline_bold: True if THIS resume's own convention bolds
             specific terms inline within bullets (from
             style_profile.detect_bullet_bold_style()). When True, the
             model is asked to wrap skill terms in **markers**
             (rendered as real bold runs by
             docx_writer.render_bullet_runs()); the returned bullet
             text KEEPS these markers as-is — stripping only happens
             downstream, once, based on whether this resume actually
             uses bolding (see write_tailored_docx()'s bullet_bold).
    Returns a list of {"bullet": str, "targets_requirement": str} dicts
    — the requirement tag lets the Executor retry a SPECIFIC bullet
    against the SAME requirement it was originally targeting, rather
    than losing that context after the first draft.
    """
    project_data = (
        f"Title: {project['title']}\n"
        f"Description: {project['description']}\n"
        f"Skills: {', '.join(project['skills'])}\n"
        f"Metrics: {project['metrics'] or '(none stated)'}"
    )

    user_message = (
        f"PROJECT DATA:\n{project_data}\n\n"
        f"TARGET JOB REQUIREMENTS:\n"
        f"{chr(10).join(f'- {r}' for r in target_requirements)}\n\n"
        f"max_count: {max_count}\n"
        f"MUST_FILL_ALL_SLOTS: {'true' if require_exact_count else 'false'}"
    )
    if guidance is not None:
        user_message += f"\n\nFACT-CHECKER FEEDBACK ON PREVIOUS ATTEMPT:\n{guidance}"
    if use_inline_bold:
        user_message += (
            "\n\nFORMATTING: this resume bolds technology/skill terms "
            "inline within its bullets. In each bullet you write, wrap "
            "ONLY terms taken verbatim from the Skills list above in "
            "**double asterisks** (e.g. \"Built a pipeline using "
            "**FastAPI** and **LangChain**\"). Do not bold anything "
            "else — not verbs, not numbers, not general phrases — only "
            "real skill/technology names from PROJECT DATA."
        )

    # Scale the token budget with how many bullets are actually being
    # requested — a fixed 800 was enough for 1-2 short bullets but
    # truncated mid-JSON once max_count grew or bullets got more
    # detailed (the style guide actively encourages more specific,
    # impact-heavy phrasing, which uses more tokens per bullet).
    token_budget = 300 + (max_count * 350)
    mode = "guided retry" if guidance is not None else "first pass"
    response_text = call_llm(
        SYSTEM_PROMPT, user_message, max_tokens=token_budget,
        context=f"draft_bullets ({mode}, {project.get('title', '?')})",
    )

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        drafts = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # A response that doesn't end with "]" almost always means it
        # got cut off by max_tokens mid-generation, not that the model
        # wrote malformed JSON — worth surfacing that distinction
        # directly rather than making the next person diagnose it from
        # the raw text themselves.
        likely_truncated = not cleaned.rstrip().endswith("]")
        hint = (
            " This looks like TRUNCATION (response doesn't end with ']') "
            "— the token_budget in draft_bullets() may need to be higher "
            "for this max_count, not a prompt/formatting problem."
            if likely_truncated else ""
        )
        raise ValueError(
            f"Model didn't return valid JSON.{hint} Raw response:\n{response_text}"
        ) from e

    if len(drafts) > max_count:
        raise ValueError(
            f"Model returned {len(drafts)} bullets but max_count was "
            f"{max_count}. Bullets: {drafts}"
        )

    return [
        {
            "bullet": _clean_bullet_text(d["bullet"], keep_bold_markers=use_inline_bold),
            "targets_requirement": d["targets_requirement"],
        }
        for d in drafts
    ]

# if __name__ == "__main__":
#     project = {
#         "title": "Self-Debugging Coding Agent",
#         "description": (
#             "I built a Python tool that automatically fixes failing unit "
#             "tests. It uses the Anthropic API with tool-calling. In my "
#             "testing it took the debug time from about 20 minutes down to "
#             "around 3 minutes for a handful of intentionally-buggy sample "
#             "bugs I planted."
#         ),
#         "skills": ["python", "anthropic api", "tool-calling", "pytest"],
#         "metrics": "debug time from about 20 minutes down to around 3 minutes",
#     }

#     print("--- Single bullet, first pass (max_count=1) ---")
#     result = draft_bullets(
#         project,
#         ["Experience building autonomous AI agents with tool use"],
#         max_count=1,
#     )
#     print(result)
#     print()

#     print("--- Guided retry (max_count=1, guidance provided) ---")
#     fake_guidance = (
#         "Rewrite without the '85% reduction' claim — the source only "
#         "states the raw before/after times, not a percentage."
#     )
#     result = draft_bullets(
#         project,
#         ["Experience building autonomous AI agents with tool use"],
#         max_count=1,
#         guidance=fake_guidance,
#     )
#     print(result)
#     print()

#     print("--- Multiple bullets, first pass (3 requirements, max_count=3) ---")
#     requirements = [
#         "Experience building autonomous AI agents with tool use",
#         "Strong debugging and testing practices",
#         "Experience with the Anthropic API or similar LLM tooling",
#     ]
#     result = draft_bullets(project, requirements, max_count=3)
#     print(f"Model wrote {len(result)} bullet(s):")
#     for d in result:
#         print(f"  [{d['targets_requirement']}] {d['bullet']}")