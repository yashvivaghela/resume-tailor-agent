"""
list_projects / get_project_details — plain read functions over the
Phase 0 cache. No LLM calls here at all — this is just a query layer,
same category as parse_jd having no caching (opposite reason: here
there's nothing to compute, just a dict lookup).
"""

from cache import load_cache


def list_projects() -> list[dict]:
    """
    Returns a lightweight summary of every cached entry (experience AND
    projects) — just enough to let the Executor decide what's worth
    looking at in more detail, without loading full descriptions for
    everything up front.
    """
    cache = load_cache()
    if cache is None:
        raise RuntimeError(
            "No projects/experience parsed yet — run parse_projects_doc() first."
        )

    return [
        {
            "id": entry_id,
            "type": entry["type"],
            "title": entry["title"],
            "one_liner": entry["one_liner"],
        }
        for entry_id, entry in cache["projects"].items()
    ]


def get_project_details(entry_id: str) -> dict:
    """
    Returns the FULL record for one entry — used once the Executor has
    decided a specific project/experience entry is relevant and needs
    the real description, skills, and metrics to draft from.
    """
    cache = load_cache()
    if cache is None:
        raise RuntimeError(
            "No projects/experience parsed yet — run parse_projects_doc() first."
        )

    projects = cache["projects"]
    if entry_id not in projects:
        raise ValueError(
            f"No entry found with id '{entry_id}'. "
            f"Available ids: {list(projects.keys())}"
        )

    return projects[entry_id]


# if __name__ == "__main__":
#     print("--- list_projects() ---")
#     for entry in list_projects():
#         print(f"  [{entry['type']}] {entry['id']}: {entry['one_liner']}")

#     print()
#     print("--- get_project_details('proj_001') ---")
#     import json
#     print(json.dumps(get_project_details("proj_001"), indent=2))