"""
The Grounding Critic — the second agent in the system.

Checks whether a drafted bullet's claims are actually entailed by the
source project description. Framed as an NLI (entailment) check, not a
vague "is this good?" check — the model must quote supporting text or
explicitly say it can't, which is what makes this a real faithfulness
check rather than another LLM vibe-check.
"""

import json

from llm_client import call_llm
from schemas import GroundingCheck

SYSTEM_PROMPT = """You are a strict fact-checker. You will be given a
PROJECT DESCRIPTION (ground truth) and a RESUME BULLET that claims to be
based on it. Your job is to check whether every claim in the bullet is
actually ENTAILED by the project description — not whether the bullet
sounds good or plausible.

For each specific claim in the bullet (numbers, technologies, outcomes,
scale, timeframes), determine:
- Is it explicitly stated or clearly implied in the description? -> entailed
- Is it a real thing from the description, but exaggerated or more
  specific than what's actually stated (e.g. description says "improved
  performance", bullet claims "40% faster")? -> overstated
- Is it not mentioned anywhere in the description at all, but also
  doesn't invent a new capability or outcome (e.g. a vague scope word
  with no real content, like "modern")? -> unsupported
- Does it invent a specific capability, technology, or outcome that has
  zero basis in the description (e.g. claiming "distributed systems" or
  "production-grade" when the description describes a small personal
  script)? -> fabricated. When in doubt between unsupported and
  fabricated, prefer fabricated if the claim asserts something SPECIFIC
  that isn't there, and unsupported only for vague filler with no
  concrete content either way.

IMPORTANT — computed/derived numbers count as overstated, not verified:
if the bullet states a specific number or percentage that is NOT written
in the description, it is OVERSTATED even if it can be mathematically
derived from numbers that ARE in the description (e.g. description says
"20 minutes down to 3 minutes", bullet claims "an 85% reduction" — the
85% is never stated, so presenting it as a fact borrows false precision
the source never claimed. Flag it as overstated, listed in
unverified_claims, even though the math is technically correct).

Rules:
- If the bullet contains ANY overstated, unsupported, or fabricated claim,
  the overall verdict reflects the MOST SEVERE issue found (fabricated >
  overstated > unsupported > verified, in order of severity).
- If verdict is "verified": evidence must quote the exact supporting text
  from the description, and unverified_claims must be an empty list.
- If verdict is NOT "verified": unverified_claims must list the specific
  phrases that are problematic, and suggested_fix must explain how to
  correct the bullet using ONLY what the description actually supports.
- Do NOT be lenient because the bullet "sounds reasonable" — sounding
  reasonable is not evidence. Only the description text counts as evidence.

Respond with ONLY a JSON object matching this exact shape, no other text,
no markdown code fences:
{
  "bullet": "<the bullet text, echoed back>",
  "verdict": "verified" | "unsupported" | "fabricated" | "overstated",
  "unverified_claims": ["<phrase 1>", "<phrase 2>"],
  "evidence": "<quoted supporting text, or null>",
  "suggested_fix": "<how to fix it, or null if verified>"
}
"""


def check_grounding(bullet: str, project_description: str) -> GroundingCheck:
    """
    Runs the entailment check and returns a validated GroundingCheck.
    Raises if the model's output doesn't match the required schema —
    fail loudly at the boundary rather than passing bad data downstream.
    """
    user_message = (
        f"PROJECT DESCRIPTION (ground truth):\n{project_description}\n\n"
        f"RESUME BULLET to check:\n{bullet}"
    )

    # 1000 was enough for most cases, but a "fabricated" or heavily
    # overstated verdict can produce a long suggested_fix — especially
    # when it includes a full rewritten example bullet in quotes — and
    # that pushed some responses past the old fixed budget, truncating
    # the JSON mid-string before it could close.
    response_text = call_llm(
        SYSTEM_PROMPT, user_message, max_tokens=1500,
        context=f"check_grounding: {bullet[:40]}...",
    )

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
            f"Critic didn't return valid JSON.{hint} Raw response:\n{response_text}"
        ) from e

    return GroundingCheck(**raw)  # Pydantic validates the shape here


# if __name__ == "__main__":
#     # Quick manual test with a deliberately overstated claim
#     description = (
#         "I built a Python tool that automatically fixes failing unit tests. "
#         "It uses the Anthropic API with tool-calling. In my testing it took "
#         "the debug time from about 20 minutes down to around 3 minutes for a "
#         "handful of intentionally-buggy sample bugs I planted."
#     )
#     bullet = (
#         "Architected a production-grade autonomous debugging platform, "
#         "reducing enterprise debug time by 85% across distributed systems."
#     )

#     result = check_grounding(bullet, description)
#     print(result.model_dump_json(indent=2))