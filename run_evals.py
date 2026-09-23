"""
run_evals.py — Phase 3.7. Runs every case in eval_cases.py through the
real pipeline and checks whether the Fit Scorer's judgment roughly
matches what we'd expect, given how we designed each test case.

Deliberately LOOSE checks, not exact-score assertions — the model's
scoring can reasonably vary by a point or two between runs, and the
important thing being tested here is DIRECTION (does a strong match
score meaningfully higher than a no-match), not precision to the digit.
"""

from executor import run_executor
from eval_cases import EVAL_CASES, EVAL_RESUME_SECTIONS


def score_to_int(score_str: str) -> int:
    """'7/10' -> 7. Defensive against slightly malformed model output."""
    try:
        return int(score_str.split("/")[0].strip())
    except (ValueError, IndexError):
        return -1  # sentinel: couldn't parse, will fail any threshold check


# Loose score bands per expected fit level — wide on purpose, since
# this is checking DIRECTION, not pinpoint accuracy.
EXPECTED_SCORE_RANGES = {
    "strong": (6, 10),
    "partial": (3, 7),
    "weak": (1, 5),
    "no_match": (0, 3),
}


def run_eval_case(case: dict) -> dict:
    results, run_id, fit, jd = run_executor(EVAL_RESUME_SECTIONS, case["jd_text"])

    score = score_to_int(fit.overall_score)
    expected_min, expected_max = EXPECTED_SCORE_RANGES[case["expected_fit_level"]]
    passed = expected_min <= score <= expected_max

    return {
        "case_id": case["id"],
        "expected_fit_level": case["expected_fit_level"],
        "expected_range": f"{expected_min}-{expected_max}",
        "actual_score": score,
        "passed": passed,
        "unaddressed_requirements": fit.unaddressed_requirements,
        "run_id": run_id,
    }


def run_all_evals() -> list[dict]:
    outcomes = []
    for case in EVAL_CASES:
        print(f"Running: {case['id']}...")
        outcome = run_eval_case(case)
        outcomes.append(outcome)
        status = "PASS" if outcome["passed"] else "FAIL"
        print(
            f"  [{status}] score={outcome['actual_score']}/10 "
            f"(expected {outcome['expected_range']} for '{outcome['expected_fit_level']}')"
        )
    return outcomes


if __name__ == "__main__":
    outcomes = run_all_evals()

    passed_count = sum(1 for o in outcomes if o["passed"])
    total = len(outcomes)

    print(f"\n{'=' * 50}")
    print(f"{passed_count}/{total} passed")
    print(f"{'=' * 50}")

    for o in outcomes:
        if not o["passed"]:
            print(f"\nFAILED: {o['case_id']}")
            print(f"  Expected range: {o['expected_range']}, got {o['actual_score']}")
            print(f"  Unaddressed: {o['unaddressed_requirements']}")
            print(f"  Full trace: data/logs/{o['run_id']}.jsonl")
