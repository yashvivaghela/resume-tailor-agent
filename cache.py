"""
Caching layer for parsed project data.

The rule: parse_projects_doc() (an LLM call) only ever runs again if the
uploaded file's CONTENT hash differs from what's cached. Re-uploading the
same file, or running the app again with a different JD, never re-parses.
"""

import json
from pathlib import Path

CACHE_PATH = Path(__file__).parent / "data" / "projects_cache.json"


def load_cache() -> dict | None:
    """
    Returns the cache dict, or None if there's effectively no usable
    cache — whether that's because the file doesn't exist, is empty
    (0 bytes), or contains corrupted/malformed JSON. In every one of
    these cases, the caller should treat it as "no cache" and re-parse,
    rather than crashing.
    """
    if not CACHE_PATH.exists():
        return None

    try:
        with open(CACHE_PATH, "r") as f:
            content = f.read().strip()
            if not content:
                # File exists but is empty — treat as no cache
                return None
            return json.loads(content)
    except json.JSONDecodeError:
        # File exists but contains invalid/corrupted JSON — treat as
        # no cache rather than crashing. Re-parsing will overwrite it
        # with a valid cache on the next save.
        print(f"[cache] Warning: {CACHE_PATH} is corrupted, ignoring and re-parsing.")
        return None


def save_cache(file_hash: str, projects: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump({"file_hash": file_hash, "projects": projects}, f, indent=2)


def get_cached_projects_if_unchanged(current_hash: str) -> dict | None:
    """Returns cached projects (a dict keyed by project id) if the hash
    matches, else None (meaning: caller needs to re-parse)."""
    cache = load_cache()
    if cache is None:
        return None
    if cache.get("file_hash") == current_hash:
        return cache.get("projects")
    return None