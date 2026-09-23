"""
app.py — Streamlit UI for the grounded resume tailoring system.

This is a thin UI layer over the existing pipeline — no new agent
logic lives here. Every function called below (parse_projects_doc,
run_executor_on_docx, read_run_log) already exists and was already
tested independently; this file just wires them to a browser.

KNOWN SCOPE LIMIT: JD input is pasted text only. fetch_job_posting(url)
was planned in Phase 1 but never actually built, so URL input isn't
supported yet — noted here rather than silently missing.
"""

import tempfile
from pathlib import Path

import streamlit as st

from parse_projects import parse_projects_doc
from executor import run_executor_on_docx
from cache import load_cache
from logger import read_run_log

st.set_page_config(page_title="Grounded Resume Tailor", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .step-track { display:flex; align-items:center; margin-bottom:1.75rem; }
    .step-circle {
        width:28px; height:28px; border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        font-size:13px; font-weight:600; flex-shrink:0;
    }
    .step-done { background:#1d9e75; color:#ffffff; }
    .step-current { background:#378add; color:#ffffff; }
    .step-pending { background:transparent; border:1px solid #888780; color:#888780; }
    .step-label { font-size:14px; margin-left:10px; }
    .step-line { flex:1; height:1px; background:#888780; margin:0 12px; }
    .upload-card {
        border:1px solid #888780; border-radius:12px; padding:14px 16px;
        display:flex; align-items:center; gap:10px;
    }
    .upload-card-done { border-color:#1d9e75; background:rgba(29,158,117,0.08); }
    .checklist-box { border-radius:8px; padding:12px 16px; background:rgba(136,135,128,0.08); margin-bottom:1rem; }
    .checklist-row { display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Grounded Resume Tailor")
st.caption(
    "A multi-agent system that tailors your resume to a job description — "
    "every claim is independently verified against your real project data "
    "before it's accepted, and the system tells you honestly when it can't "
    "back something up, rather than guessing."
)

# ---------------------------------------------------------------------
# Step 1 — projects/experience doc (only needed once; cached after that)
# ---------------------------------------------------------------------
existing_cache = load_cache()

# ---------------------------------------------------------------------
# Step tracker — shows progress across the three inputs. Purely visual;
# the actual gating logic (`can_run`) below is unchanged.
# ---------------------------------------------------------------------
resume_file_present = st.session_state.get("resume_upload") is not None
jd_present = bool(st.session_state.get("jd_text_area", "").strip())

step1_class = "step-done" if existing_cache is not None else "step-current"
step1_icon = "✓" if existing_cache is not None else "1"
step2_class = "step-done" if resume_file_present else ("step-current" if existing_cache is not None else "step-pending")
step2_icon = "✓" if resume_file_present else "2"
step3_class = "step-done" if jd_present else ("step-current" if resume_file_present else "step-pending")
step3_icon = "✓" if jd_present else "3"

st.markdown(
    f"""
    <div class="step-track">
        <div class="step-circle {step1_class}">{step1_icon}</div>
        <span class="step-label">Projects doc</span>
        <div class="step-line"></div>
        <div class="step-circle {step2_class}">{step2_icon}</div>
        <span class="step-label">Resume</span>
        <div class="step-line"></div>
        <div class="step-circle {step3_class}">{step3_icon}</div>
        <span class="step-label">Job description</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander(
    "1. Upload your projects + experience doc",
    expanded=(existing_cache is None),
):
    if existing_cache is not None:
        st.success(
            f"Using previously uploaded data — "
            f"{len(existing_cache['projects'])} entries loaded. "
            f"Upload a new file below only if you want to replace it."
        )

    projects_file = st.file_uploader(
        "PDF, DOCX, or TXT — a doc listing your Experience and Projects "
        "(see README for the expected format)",
        type=["pdf", "docx", "txt"],
        key="projects_upload",
    )

    if projects_file is not None:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(projects_file.name).suffix
        ) as tmp:
            tmp.write(projects_file.getvalue())
            tmp_path = tmp.name

        with st.spinner("Reading your projects doc..."):
            try:
                projects = parse_projects_doc(tmp_path)
                st.success(f"Loaded {len(projects)} entries from your doc.")
            except Exception as e:
                st.error(f"Couldn't parse that file: {e}")

# ---------------------------------------------------------------------
# Step 2 — resume + job description
# ---------------------------------------------------------------------
res_col, spacer_col = st.columns([1, 1])
with res_col:
    st.subheader("2. Upload your resume")
    resume_file = st.file_uploader("Resume (.docx only)", type=["docx"], key="resume_upload")
    if resume_file is not None:
        size_kb = len(resume_file.getvalue()) / 1024
        st.markdown(
            f"""
            <div class="upload-card upload-card-done">
                <span>📄</span>
                <div>
                    <div style="font-size:13px;">{resume_file.name}</div>
                    <div style="font-size:12px; opacity:0.7;">{size_kb:.1f} KB</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.subheader("3. Paste the job description")
jd_text = st.text_area(
    "Job description text",
    height=200,
    placeholder="Paste the full job posting text here...",
    key="jd_text_area",
)
word_count = len(jd_text.split())
cap_col1, cap_col2 = st.columns([3, 1])
with cap_col1:
    st.caption("URL input isn't supported yet — paste the job description text directly.")
with cap_col2:
    st.caption(f"{word_count} words")

can_run = resume_file is not None and jd_text.strip() and load_cache() is not None

missing = []
if load_cache() is None:
    missing.append("a projects/experience doc")
if resume_file is None:
    missing.append("a resume")
if not jd_text.strip():
    missing.append("a job description")

checklist_items = [
    ("a projects/experience doc", "Projects + experience doc"),
    ("a resume", "Resume uploaded"),
    ("a job description", "Job description pasted"),
]
checklist_rows = "".join(
    f'<div class="checklist-row"><span>{"⚪" if key in missing else "✅"}</span>'
    f'<span>{label}</span></div>'
    for key, label in checklist_items
)
st.markdown(f'<div class="checklist-box">{checklist_rows}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Step 3 — run the pipeline
# ---------------------------------------------------------------------
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None

if st.button("Tailor my resume", disabled=not can_run, type="primary"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_resume:
        tmp_resume.write(resume_file.getvalue())
        resume_path = tmp_resume.name

    output_path = str(Path(tempfile.gettempdir()) / "tailored_resume.docx")

    pipeline_error = None
    with st.spinner(
        "Running the agent pipeline — this can take a minute or two "
        "(matching sections, drafting, fact-checking, scoring)..."
    ):
        try:
            results, run_id, fit = run_executor_on_docx(resume_path, jd_text, output_path)
        except Exception as e:
            # Don't call st.error()/st.stop() while still inside the
            # spinner block — doing so interrupts Streamlit mid-cleanup
            # of the spinner's UI element, leaving it visually "stuck"
            # showing as still running even though execution actually
            # halted underneath it. Capture the error, let the `with`
            # block exit NORMALLY first, then handle it outside.
            pipeline_error = str(e)

    if pipeline_error is not None:
        st.error(f"Something went wrong: {pipeline_error}")
        st.stop()

    # SAVE to session_state — this is what survives the rerun that
    # clicking the download button (or any other widget) triggers.
    # Local variables like `results`/`fit` here would otherwise vanish
    # the moment ANY widget is clicked again, since Streamlit reruns
    # the whole script from the top and this `if st.button(...)` block
    # would no longer be True on that rerun.
    with open(output_path, "rb") as f:
        docx_bytes = f.read()

    st.session_state.pipeline_results = {
        "results": results,
        "fit": fit,
        "run_id": run_id,
        "docx_bytes": docx_bytes,
    }


def _parse_score(score, max_score=10):
    """fit.overall_score has shown up as a plain number and as a string
    like "3/10" depending on the scorer's output — handle both so the
    UI doesn't crash on whichever one comes back."""
    if isinstance(score, str):
        if "/" in score:
            num, _, denom = score.partition("/")
            try:
                return float(num.strip()), float(denom.strip())
            except ValueError:
                return 0.0, max_score
        try:
            return float(score.strip()), max_score
        except ValueError:
            return 0.0, max_score
    return float(score), max_score


def _score_color(score, max_score=10):
    value, max_score = _parse_score(score, max_score)
    pct = value / max_score if max_score else 0
    if pct <= 0.4:
        return "#e24b4a"  # red
    if pct <= 0.7:
        return "#ef9f27"  # amber
    return "#63a022"  # green


def _dedupe_unaddressed(weak_areas, unaddressed_requirements):
    """Drop unaddressed items already implied by a weak-area line, so the
    two columns stop repeating the same skills (e.g. 'Django' appearing in
    both 'Django experience (only listed...)' and a bare 'Django' row)."""
    weak_text = " | ".join(a.lower() for a in weak_areas)
    filtered = []
    for item in unaddressed_requirements:
        key = item.lower().split(" (")[0].strip()
        if key and key in weak_text:
            continue
        filtered.append(item)
    return filtered


# ---------------------------------------------------------------------
# Display — reads from session_state, NOT from the button's click state.
# This is what makes it survive the rerun triggered by clicking
# Download (or the reasoning-trace expander, or anything else).
# ---------------------------------------------------------------------
if st.session_state.pipeline_results is not None:
    saved = st.session_state.pipeline_results
    results, fit, run_id, docx_bytes = (
        saved["results"], saved["fit"], saved["run_id"], saved["docx_bytes"]
    )

    st.success("Done.")

    # --- Fit Score ---
    unaddressed_filtered = _dedupe_unaddressed(fit.weak_areas, fit.unaddressed_requirements)
    score_value, score_max = _parse_score(fit.overall_score)
    score_display = int(score_value) if score_value == int(score_value) else score_value
    color = _score_color(fit.overall_score)

    with st.container(border=True):
        score_col, note_col, dl_col = st.columns([1, 3, 1.3])
        with score_col:
            st.markdown(
                f"""
                <p style="font-size:12px; color:#888780; margin:0 0 4px; text-transform:uppercase; letter-spacing:0.04em;">Overall fit</p>
                <p style="font-size:40px; font-weight:700; margin:0; color:{color}; line-height:1;">{score_display}<span style="font-size:16px; opacity:0.55;">/{int(score_max)}</span></p>
                """,
                unsafe_allow_html=True,
            )
        with note_col:
            st.markdown(
                f'<p style="margin-top:10px; font-size:14px; color:#888780; line-height:1.5;">{fit.honest_note}</p>',
                unsafe_allow_html=True,
            )
        with dl_col:
            st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
            st.download_button(
                "⬇ Download resume",
                docx_bytes,
                file_name="tailored_resume.docx",
                type="primary",
                use_container_width=True,
            )

        st.divider()

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            st.markdown(
                f'<p style="font-size:13px; font-weight:600; color:#1d9e75; margin:0 0 10px;">✅ Strong ({len(fit.strong_areas)})</p>',
                unsafe_allow_html=True,
            )
            tags = "".join(
                f'<span style="display:inline-block; font-size:12px; padding:5px 10px; margin:0 6px 6px 0; '
                f'background:rgba(29,158,117,0.15); color:#1d9e75; border-radius:999px;">{a}</span>'
                for a in fit.strong_areas
            )
            st.markdown(f'<div>{tags}</div>', unsafe_allow_html=True)
        with fcol2:
            st.markdown(
                f'<p style="font-size:13px; font-weight:600; color:#ef9f27; margin:0 0 10px;">⚠️ Partially covered ({len(fit.weak_areas)})</p>',
                unsafe_allow_html=True,
            )
            rows = "".join(
                f'<span style="display:inline-block; font-size:12px; padding:5px 10px; margin:0 6px 6px 0; '
                f'background:rgba(239,159,39,0.15); color:#ef9f27; border-radius:999px;">{a}</span>'
                for a in fit.weak_areas
            )
            st.markdown(f'<div>{rows}</div>', unsafe_allow_html=True)
        with fcol3:
            st.markdown(
                f'<p style="font-size:13px; font-weight:600; color:#e24b4a; margin:0 0 10px;">❌ Missing ({len(unaddressed_filtered)})</p>',
                unsafe_allow_html=True,
            )
            tags = "".join(
                f'<span style="display:inline-block; font-size:12px; padding:5px 10px; margin:0 6px 6px 0; '
                f'background:rgba(226,75,74,0.15); color:#e24b4a; border-radius:999px;">{a}</span>'
                for a in unaddressed_filtered
            )
            st.markdown(f'<div>{tags}</div>', unsafe_allow_html=True)

    # --- Per-section results ---
    st.subheader("Section-by-section changes")
    replaced_count = sum(1 for r in results if r["was_replaced"])
    st.caption(f"{replaced_count} of {len(results)} sections replaced")

    for r in results:
        badge = "REPLACED" if r["was_replaced"] else "kept"
        label = f"{r['section_name']} — {badge}"
        # Only auto-expand the sections that actually changed; "kept"
        # sections are just confirmation and don't need to open by default.
        with st.expander(label, expanded=r["was_replaced"]):
            st.caption(r["assignment_reason"])
            for b in r["final_bullets"]:
                st.write(f"• {b}")
            for g in r["gaps"]:
                st.warning(f"Not addressed — {g['requirement']}: {g['reason']}")

    # --- Reasoning trace ---
    with st.expander("Show full reasoning trace (every agent decision, in order)"):
        events = read_run_log(run_id)
        st.json(events)