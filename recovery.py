"""
recovery.py — the decision that closes the outer loop diagrammed
earlier: given the Fit Scorer's unaddressed_requirements, is there an
UNUSED entry (not already assigned to any section) that could fill one
of those gaps by reassigning a whole section to it?

Deliberately scoped to AT MOST ONE section per run (checked by the
caller) — this is a recovery attempt, not a full re-planning pass.
Recovery only ever reassigns a WHOLE section (never patches a single
bullet with a foreign project's content), because doing otherwise would
reintroduce the mixed-project bug fixed earlier.
"""

import json

from llm_client import call_llm
from schemas import RecoveryDecision, FitScore, JDRequirements

SYSTEM_PROMPT = """You decide whether ONE resume section should be
reassigned to a different, currently-unused project/experience entry,
specifically to address requirements the finished resume doesn't
currently cover.

You'll be given:
1. unaddressed_requirements — JD requirements with no real evidence
   anywhere in the current resume
2. CURRENT SECTIONS — each with its type, currently assigned entry, and
   any existing honest gaps in that section
3. UNUSED ENTRIES — entries not currently assigned to any section

Decide: is there ONE unused entry that would genuinely help address one
or more unaddressed requirements, if swapped into a section of the SAME
TYPE (an experience section can only take an unused experience entry,
a project section only an unused project entry)?

Only recommend a change if it would ACTUALLY help — don't force a
recommendation just because one is being asked for. If no unused entry
would genuinely improve coverage of the unaddressed requirements,
return null for both section_name and new_entry_id, and say so plainly
in reason.

Respond with ONLY a JSON object, no other text, no markdown code fences:
{
  "section_name": "<section to reassign, or null>",
  "new_entry_id": "<unused entry id to assign there, or null>",
  "reason": "<why this recovery helps, or why none was found>"
}
"""


def attempt_recovery(
    fit: FitScore,
    results: list[dict],
    resume_sections: list[dict],
    unused_entries: list[dict],
    jd_requirements: JDRequirements,
) -> RecoveryDecision:
    """
    results: run_executor()'s per-section results (has assigned_entry_id,
              gaps, entry_type via resume_sections lookup)
    resume_sections: the original section list (has entry_type, bullet_count)
    unused_entries: list_projects() filtered to ids NOT already assigned
    """
    section_type_by_name = {s["section_name"]: s["entry_type"] for s in resume_sections}

    current_sections_summary = [
        {
            "section_name": r["section_name"],
            "entry_type": section_type_by_name.get(r["section_name"]),
            "current_entry_id": r["assigned_entry_id"],
            "existing_gaps": r["gaps"],
        }
        for r in results
    ]

    user_message = (
        f"UNADDRESSED REQUIREMENTS:\n{json.dumps(fit.unaddressed_requirements, indent=2)}\n\n"
        f"CURRENT SECTIONS:\n{json.dumps(current_sections_summary, indent=2)}\n\n"
        f"UNUSED ENTRIES:\n{json.dumps(unused_entries, indent=2)}\n\n"
        f"JOB REQUIREMENTS:\n{jd_requirements.model_dump_json(indent=2)}"
    )

    response_text = call_llm(SYSTEM_PROMPT, user_message, max_tokens=600, context="recovery")

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model didn't return valid JSON. Raw response:\n{response_text}"
        ) from e

    decision = RecoveryDecision(**raw)

    # Verify the type constraint ourselves too — same discipline as
    # assign_projects.py, don't just trust the prompt.
    if decision.section_name is not None:
        expected_type = section_type_by_name.get(decision.section_name)
        actual_type = next(
            (e["type"] for e in unused_entries if e["id"] == decision.new_entry_id), None
        )
        if expected_type is not None and actual_type != expected_type:
            raise ValueError(
                f"Recovery type mismatch: section '{decision.section_name}' "
                f"is type '{expected_type}' but recovery proposed entry "
                f"'{decision.new_entry_id}' (type '{actual_type}')."
            )

    return decision