"""
logger.py — structured, append-only logging for every agent decision
in a run. One JSONL file per run, human-readable, greppable, no
database needed at this scale.

This is what makes "why did it do that?" answerable after the fact,
not just while watching it live in a terminal — and it's the raw
material Phase 3.7's eval harness will count against (e.g. "62% of
bullets verified on first pass").
"""

import json
import time
import uuid
from pathlib import Path

LOG_DIR = Path(__file__).parent / "data" / "logs"


def new_run_id() -> str:
    """A short, sortable-enough id for one full executor run."""
    return f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"


def log_event(run_id: str, event_type: str, data: dict) -> None:
    """
    event_type: one of "assignment_decision", "draft_attempt",
    "grounding_check", "retry", "final_outcome" (not enforced as an
    enum — free-form string, kept loose deliberately so new event
    types can be added without touching this file).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{run_id}.jsonl"

    with open(log_path, "a") as f:
        f.write(json.dumps({
            "timestamp": time.time(),
            "event_type": event_type,
            **data,
        }) + "\n")


def read_run_log(run_id: str) -> list[dict]:
    """Reads a run's full log back as a list of event dicts, in order.
    Useful for the eval harness and for a future 'show reasoning trace'
    UI feature."""
    log_path = LOG_DIR / f"{run_id}.jsonl"
    if not log_path.exists():
        return []
    with open(log_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


# if __name__ == "__main__":
#     # Quick sanity check — no LLM calls needed, just verify the
#     # mechanics of writing and reading a log work correctly
#     test_run_id = new_run_id()
#     log_event(test_run_id, "test_event", {"foo": "bar"})
#     log_event(test_run_id, "test_event", {"foo": "baz"})

#     events = read_run_log(test_run_id)
#     print(f"Run id: {test_run_id}")
#     print(f"Logged {len(events)} events:")
#     for e in events:
#         print(f"  [{e['event_type']}] {e}")

#     # Clean up the test log so it doesn't clutter real run history
#     (LOG_DIR / f"{test_run_id}.jsonl").unlink()
#     print("\nTest log cleaned up.")