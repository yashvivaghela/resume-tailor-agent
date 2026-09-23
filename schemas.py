"""
Structured output schemas for Phase 2's agent calls.

Using Pydantic here (rather than trusting raw LLM JSON blindly) means
malformed model output fails loudly and immediately, at the boundary,
instead of causing a confusing error somewhere downstream.
"""

from enum import Enum
from pydantic import BaseModel


class GroundingVerdict(str, Enum):
    VERIFIED = "verified"        # every claim traces back to the source
    UNSUPPORTED = "unsupported"  # no evidence either way — not mentioned at all
    FABRICATED = "fabricated"    # actively contradicts or has zero basis
    OVERSTATED = "overstated"    # real thing, but exaggerated (e.g. invented a number)


class GroundingCheck(BaseModel):
    bullet: str
    verdict: GroundingVerdict
    unverified_claims: list[str]  # empty list if verdict is VERIFIED
    evidence: str | None          # supporting quote from source, or None
    suggested_fix: str | None     # how to fix it, or None if VERIFIED


class JDRequirements(BaseModel):
    job_title: str                    # e.g. "Senior Backend Engineer"
    seniority_level: str              # e.g. "entry" | "mid" | "senior" | "staff" | "unspecified"
    required_experience_years: str    # e.g. "5+ years", or "" if not stated
    education_requirement: str        # e.g. "Bachelor's in CS or equivalent experience", or "" if not stated
    required_skills: list[str]
    nice_to_have: list[str]
    role_focus: str                   # short phrase, e.g. "backend API development"


class SectionAssignment(BaseModel):
    section_name: str              # e.g. "Project 2" — matches the resume slot
    assigned_entry_id: str         # the project/experience id now filling this slot
    was_replaced: bool             # True if this differs from what was originally there
    reason: str                    # why this entry was chosen (or kept)
    evidence_phrase: str | None = None  # required when was_replaced=True: the
                                         # exact phrase from the NEW entry's real
                                         # description that demonstrates the
                                         # claimed skill. Verified in code against
                                         # the entry's actual text (assign_projects.py)
                                         # — a swap whose evidence doesn't check out
                                         # gets reverted, since the model can state
                                         # a plausible-sounding justification that
                                         # isn't actually true of the entry's text.

class FitScore(BaseModel):
    overall_score: str                       # e.g. "7/10"
    strong_areas: list[str]                   # requirements the resume genuinely addresses well
    weak_areas: list[str]                      # requirements addressed, but thinly
    unaddressed_requirements: list[str]        # requirements with no real evidence anywhere
    honest_note: str                           # a candid, non-flattering summary if the fit is weak
 
class RecoveryDecision(BaseModel):
    section_name: str | None    # which section to reassign, or None if no recovery is worth it
    new_entry_id: str | None    # the unused entry to reassign it to, or None
    reason: str