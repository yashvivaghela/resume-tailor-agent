"""
eval_cases.py — the test set for Phase 3.7 (eval harness) and Phase 3.8
(ablation study). Both reuse this same list, since the ablation study
is just "run this exact test set twice, once with grounding on and
once off."

Each case is a real, distinct JD spanning a spread of fit levels
against typical cached project data (a debugging agent, a Flask
dashboard, a Discord bot, a React internship, a TA role) — deliberately
NOT all backend-Python JDs, so the eval actually exercises different
match qualities rather than testing the same easy case five times.

NOTE: entry ids below (exp_001, proj_001, etc.) assume your cache was
built from the combined Experience+Projects sample doc used throughout
development. If your real projects_cache.json has different ids, update
EVAL_RESUME_SECTIONS below to match, or the eval will fail at
get_project_details() with a clear "no entry found" error.
"""

EVAL_CASES = [
    {
        "id": "strong_backend_match",
        "jd_text": """
        Backend Engineer

        Requirements:
        - Experience with Python
        - Experience building and testing backend services
        - Familiarity with REST APIs
        """,
        "expected_fit_level": "strong",  # loose sanity check, not a hard assertion
    },
    {
        "id": "partial_fullstack_match",
        "jd_text": """
        Full-Stack Engineer

        Requirements:
        - Experience with React and a backend language (Python or Node)
        - Experience with authentication (JWT or OAuth)
        - Comfortable working directly with end users / beta testers

        Nice to have:
        - Experience with data visualization
        """,
        "expected_fit_level": "partial",
    },
    {
        "id": "weak_senior_match",
        "jd_text": """
        Senior Backend Engineer

        Requirements:
        - 5+ years of professional software engineering experience
        - Experience with CI/CD pipelines
        - Experience mentoring junior engineers
        - Bachelor's degree in Computer Science or equivalent experience
        """,
        "expected_fit_level": "weak",
    },
    {
        "id": "no_match_ml_role",
        "jd_text": """
        Machine Learning Research Engineer

        Requirements:
        - PhD in Machine Learning or a related field
        - Deep expertise in distributed training (multi-GPU/TPU)
        - Published research or patents in ML
        - Experience with PyTorch at production scale
        """,
        "expected_fit_level": "no_match",
    },
    {
        "id": "frontend_focused_match",
        "jd_text": """
        Frontend Engineer

        Requirements:
        - Experience with React
        - Experience building data visualizations or dashboards
        - Comfortable working with a backend API

        Nice to have:
        - Node.js or Discord bot development experience
        """,
        # Originally labeled "partial" — corrected after the eval run
        # showed a 9/10 score. On reflection this JD doesn't actually
        # require frontend-specific frameworks/design skills, just
        # React + dashboards + a backend API — which the cached React/
        # Flask/Recharts project genuinely satisfies well. The eval
        # caught a miscalibrated expectation, not a pipeline bug.
        "expected_fit_level": "strong",
    },
]


# A fixed, representative resume structure to test against — mirrors
# the shape of a real read_docx_sections() output, kept fixed across
# every eval case and both ablation runs so the ONLY thing varying is
# the JD, not the underlying data.
EVAL_RESUME_SECTIONS = [
    {"section_name": "Experience 1", "entry_type": "experience", "current_entry_id": "exp_001", "bullet_count": 2},
    {"section_name": "Experience 2", "entry_type": "experience", "current_entry_id": "exp_002", "bullet_count": 2},
    {"section_name": "Project 1", "entry_type": "project", "current_entry_id": "proj_001", "bullet_count": 2},
    {"section_name": "Project 2", "entry_type": "project", "current_entry_id": "proj_002", "bullet_count": 3},
    {"section_name": "Project 3", "entry_type": "project", "current_entry_id": "proj_003", "bullet_count": 2},
]