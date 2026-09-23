"""
parse_jd — turns a raw job description (pasted text) into structured
requirements. One-shot LLM call, not an agent — fixed extraction task,
no tools, no loop.
"""

import json

from llm_client import call_llm
from schemas import JDRequirements

SYSTEM_PROMPT = """You extract structured requirements from a raw job
description. Job postings vary wildly in format — some have clear
"Requirements" and "Nice to have" sections, others bury everything in
prose. Use your judgment to identify what's actually being asked for.

Extract:
- job_title: the job title as stated in the posting (e.g. "Senior
  Backend Engineer"). If genuinely not stated, use "unspecified".
- seniority_level: one of "entry", "mid", "senior", "staff", or
  "unspecified" if the posting doesn't make this clear. Infer this from
  the title, years of experience required, and language used (e.g.
  "lead", "mentor junior engineers" implies senior/staff).
- required_experience_years: the years of experience explicitly stated
  (e.g. "5+ years"). Use an empty string if not stated — do NOT guess
  a number that isn't written.
- education_requirement: any degree requirement explicitly stated (e.g.
  "Bachelor's in Computer Science or related field", "Master's
  preferred", "PhD required"). Use an empty string if not mentioned —
  many postings don't require one, and that absence matters, don't fill
  it in with an assumption.
- required_skills: skills/technologies/experience explicitly stated as
  required, must-have, or clearly essential to the role (e.g. "3+ years
  of Python", "experience with distributed systems")
- nice_to_have: skills mentioned as a bonus, preferred, or "a plus" —
  not strictly required
- role_focus: a short phrase capturing what this role is fundamentally
  about (e.g. "backend API development", "full-stack web applications",
  "data pipeline engineering") — this helps later steps understand the
  role's overall shape, not just its skill checklist

Be specific rather than generic — extract actual named technologies,
years of experience, and concrete requirements as they're written, not
vague paraphrases like "strong technical skills." Never invent a
requirement that isn't stated — an empty string or "unspecified" is the
correct answer when the posting genuinely doesn't say.

Respond with ONLY a JSON object matching this exact shape, no other
text, no markdown code fences:
{
  "job_title": "<title or 'unspecified'>",
  "seniority_level": "entry" | "mid" | "senior" | "staff" | "unspecified",
  "required_experience_years": "<e.g. '5+ years', or empty string>",
  "education_requirement": "<degree requirement, or empty string>",
  "required_skills": ["<skill 1>", "<skill 2>"],
  "nice_to_have": ["<skill 1>", "<skill 2>"],
  "role_focus": "<short phrase>"
}
"""


def parse_jd(jd_text: str) -> JDRequirements:
    # 1000 was too tight for a JD with many distinct requirements —
    # required_skills/nice_to_have can get long. Bumped preemptively
    # after the same truncation pattern hit fit_scorer, draft_bullets,
    # and check_grounding on real, richer inputs.
    response_text = call_llm(SYSTEM_PROMPT, jd_text, max_tokens=1500, context="parse_jd")

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError as e:
        likely_truncated = not cleaned.rstrip().endswith("}")
        hint = (
            " This looks like TRUNCATION (response doesn't end with '}') "
            "— the max_tokens budget may need to be higher."
            if likely_truncated else ""
        )
        raise ValueError(
            f"Model didn't return valid JSON.{hint} Raw response:\n{response_text}"
        ) from e

    return JDRequirements(**raw)  # Pydantic validates the shape here


# if __name__ == "__main__":
#     sample_jd = """
#     Senior Backend Engineer

#     We're looking for a senior backend engineer to join our platform
#     team and help mentor junior engineers.

#     Requirements:
#     - 5+ years of experience with Python or a similar language
#     - Experience designing and building REST APIs at scale
#     - Familiarity with relational databases (Postgres preferred)
#     - Bachelor's degree in Computer Science or equivalent experience
#     - Comfortable working in a fast-paced startup environment

#     Nice to have:
#     - Experience with Docker/containerization
#     - Exposure to AWS or another cloud provider
#     - Prior experience with React or frontend work
#     """

#     result = parse_jd(sample_jd)
#     print(result.model_dump_json(indent=2))