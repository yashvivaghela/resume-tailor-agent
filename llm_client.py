"""
Thin wrapper around the Groq client so the rest of the app doesn't need
to know which provider we're using. Swap providers later by only editing
this file.
"""

import os
import time
from groq import Groq
from groq import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from dotenv import load_dotenv

load_dotenv()  # reads .env file in project root, if present

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 5  # doubles each retry: 5s, 10s, 20s


class LLMRateLimitError(Exception):
    """Raised only after retries are exhausted, so callers (and the
    person watching the terminal) get one clear, friendly message
    instead of a raw SDK traceback."""
    pass


class LLMEmptyResponseError(Exception):
    """Raised only after retries are exhausted for a persistently empty
    response — distinct from LLMRateLimitError so the message doesn't
    misleadingly blame rate limiting for what's actually a different
    (rarer, usually transient) failure mode."""
    pass


def call_llm(system_prompt: str, user_message: str, max_tokens: int = 2000, context: str = "unknown") -> str:
    """
    Plain single-turn LLM call — no tools, no loop. Used for one-shot
    extraction/judgment tasks (parse_jd, parse_projects_doc, score_fit,
    check_grounding) that don't need multi-step decision-making.

    context: a short label identifying WHICH call site this is (e.g.
    "assign_projects", "check_grounding: proj_001"). Every caller should
    pass this. It shows up in retry/error messages so a failure says
    exactly where it happened, instead of just "the model failed" —
    without it, diagnosing which of the ~7 different call sites broke
    means guessing from a generic message.

    Automatically retries on rate limits, transient connection/server
    errors, AND a completely empty response (an occasional API glitch,
    not a content problem — the model producing zero tokens is not the
    same failure as the model producing a bad JSON string). These are
    all TECHNICAL failures, deliberately separate from the "reword this
    bullet" retry logic elsewhere, which handles bad CONTENT, not a
    failed/empty API call. Every function in this project funnels
    through here, so this protection applies everywhere with no other
    file needing to know about it.
    """
    print(f"[llm_client] -> calling model for: {context}")
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            content = response.choices[0].message.content

            if not content or not content.strip():
                # Empty response — same treatment as a technical failure,
                # NOT handed back to the caller to fail on json.loads("").
                last_error = "API returned an empty response"
                if attempt < MAX_RETRIES - 1:
                    wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
                    print(
                        f"[llm_client] [{context}] Empty response from model "
                        f"(attempt {attempt + 1}/{MAX_RETRIES}). Waiting "
                        f"{wait}s before retrying..."
                    )
                    time.sleep(wait)
                continue

            print(f"[llm_client] <- done: {context}")
            return content

        except RateLimitError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
                print(
                    f"[llm_client] [{context}] Rate limit hit (attempt "
                    f"{attempt + 1}/{MAX_RETRIES}). Waiting {wait}s before "
                    f"retrying..."
                )
                time.sleep(wait)
            continue

        except (APIConnectionError, APITimeoutError, InternalServerError) as e:
            # Transient network/server issues — worth a couple retries
            # too, same backoff, just a different message since it's
            # not specifically a rate limit.
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
                print(
                    f"[llm_client] [{context}] Temporary connection issue "
                    f"(attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__}. "
                    f"Waiting {wait}s before retrying..."
                )
                time.sleep(wait)
            continue

    # Retries exhausted — raise ONE clear, friendly error instead of
    # whatever raw SDK exception/traceback would otherwise surface.
    if last_error == "API returned an empty response":
        raise LLMEmptyResponseError(
            f"[{context}] The model returned an empty response {MAX_RETRIES} "
            f"times in a row. This is usually a transient API issue — try "
            f"again in a moment."
        )

    raise LLMRateLimitError(
        f"[{context}] You've hit a rate limit on the model after "
        f"{MAX_RETRIES} retries. Try again in a few minutes. "
        f"(underlying error: {last_error})"
    ) from last_error