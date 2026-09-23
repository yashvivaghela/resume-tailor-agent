"""
executor.py — the main agentic loop. Ties together everything built so
far: Project Assignment (Phase 1.5), drafting (Phase 2), and grounding
verification (Phase 2), into the actual retry/reword/flag-gap behavior
diagrammed early in Phase 2's design.

This is the piece that turns individually-tested functions into an
actual agent: it observes each grounding verdict and decides what to do
next, rather than following one fixed path.
"""

from parse_jd import parse_jd
from assign_projects import assign_projects_to_sections
from project_queries import get_project_details, list_projects
from draft_bullet import draft_bullets
from check_grounding import check_grounding
from schemas import GroundingVerdict
from logger import new_run_id, log_event
from docx_sections import read_docx_sections
from match_sections_to_entries import match_sections_to_entries
from docx_writer import write_tailored_docx
from style_profile import detect_title_line_style, detect_bullet_bold_style
from fit_scorer import score_fit, format_resume_text
from recovery import attempt_recovery
from schemas import FitScore, JDRequirements
import re as _re

_BOLD_MARKER_RE_FOR_CHECK = _re.compile(r"\*\*(.+?)\*\*")


def _for_grounding_check(bullet: str) -> str:
    """Strips **bold** markers before check_grounding sees the bullet —
    they're a cosmetic formatting signal, not a factual claim, and
    shouldn't affect whether the underlying claim is verified."""
    return _BOLD_MARKER_RE_FOR_CHECK.sub(lambda m: m.group(1), bullet)

MAX_GROUNDING_RETRIES = 2


def _fallback_bullet_for_entry(entry: dict) -> str:
    """
    Last-resort content for a bullet slot that has no verified draft to
    show for it (every attempt was rejected by check_grounding, and
    there was nothing better to fall back to). Used ONLY for a
    was_replaced=True section, where leaving the slot blank in `edits`
    is the alternative — and a visibly empty bullet in the final docx
    is worse than a modest, unglamorous, but 100% truthful one.

    Deliberately NOT another LLM call: recovery is already the
    "something else already failed" path (this fires after
    MAX_GROUNDING_RETRIES drafts were all rejected), and another model
    call here just risks failing the same way (as seen with recovery's
    own empty-response failures) — this must always succeed. It's built
    only from `entry["skills"]`, which is already-verified real
    metadata from the source project doc, so it's structurally
    incapable of fabricating anything.
    """
    skills = entry.get("skills") or []
    if skills:
        return f"Built {entry['title']}, applying {', '.join(skills[:4])}."
    return f"Built {entry['title']}."


def run_section(
    run_id: str,
    section_name: str,
    entry: dict,
    target_requirements: list[str],
    bullet_count: int,
    was_replaced: bool = False,
    grounding_enabled: bool = True,
    use_inline_bold: bool = False,
) -> dict:
    """
    Runs the full draft -> check -> retry loop for ONE resume section's
    worth of bullets. Returns:
      {
        "slots": [
          {"status": "final", "bullet": str, "requirement": str} |
          {"status": "gap", "requirement": str, "reason": str},
          ...  # one entry per draft, in ORIGINAL order — this is what
               # correctly maps back to bullet_indices[i] later, since
               # a middle draft can fail while later ones succeed
        ],
        "final_bullets": [str, ...],   # convenience view, order preserved
        "gaps": [{"requirement": str, "reason": str}, ...]  # convenience view
      }

    was_replaced: True when this section's PROJECT is being swapped for
    a different one (not just re-tailoring the same project). This
    matters because an unfilled slot means something different in each
    case: if the project isn't changing, an untouched slot is still
    honestly describing the same project. If the project IS changing,
    an unfilled slot would leave the OLD project's stale text behind
    under the NEW project's title — so drafting is told to fill every
    slot with general-but-true content about the new project instead.

    grounding_enabled: exists ONLY for the Phase 3.8 ablation study —
    when False, every drafted bullet is accepted unconditionally with
    NO check_grounding call at all (no verification, no retry, no
    gaps). Never use False in the real pipeline; it exists purely so
    the ablation script can measure what fabrication/overstatement
    rate the grounding check actually prevents, by comparing this
    mode's raw output against the normal mode's.

    IMPORTANT: "slots" is the one downstream code (docx editing) must
    use for positional mapping. "final_bullets"/"gaps" are flattened
    views for callers that only care about content (e.g. the Fit
    Scorer), NOT for figuring out which paragraph index gets what —
    using those for positional mapping was a real bug (a gap in the
    middle would silently shift every later bullet into the wrong slot).
    """
    drafts = draft_bullets(
        entry, target_requirements, max_count=bullet_count,
        require_exact_count=was_replaced, use_inline_bold=use_inline_bold,
    )
    log_event(run_id, "draft_attempt", {
        "section_name": section_name,
        "entry_id": entry["title"],
        "requested_max_count": bullet_count,
        "actual_count": len(drafts),
        "drafts": drafts,
    })

    slots = []

    for draft in drafts:
        bullet = draft["bullet"]
        requirement = draft["targets_requirement"]

        if not grounding_enabled:
            # ABLATION MODE ONLY: accept the raw draft unconditionally,
            # no check_grounding call at all. This is what lets the
            # ablation script measure what the grounding check actually
            # catches — by comparing THIS unchecked output against the
            # normal mode's, using an independent check_grounding pass
            # afterward purely for scoring, never for gating.
            slots.append({"status": "final", "bullet": bullet, "requirement": requirement})
            log_event(run_id, "grounding_skipped_ablation_mode", {
                "section_name": section_name,
                "requirement": requirement,
                "bullet": bullet,
            })
            continue

        check = check_grounding(_for_grounding_check(bullet), entry["description"])
        attempts = 0
        log_event(run_id, "grounding_check", {
            "section_name": section_name,
            "requirement": requirement,
            "attempt": attempts,
            "bullet": bullet,
            "verdict": check.verdict.value,
            "unverified_claims": check.unverified_claims,
        })

        failing_verdicts = {
            GroundingVerdict.OVERSTATED,
            GroundingVerdict.UNSUPPORTED,
            GroundingVerdict.FABRICATED,
        }
        while check.verdict in failing_verdicts and attempts < MAX_GROUNDING_RETRIES:
            attempts += 1
            log_event(run_id, "retry", {
                "section_name": section_name,
                "requirement": requirement,
                "attempt": attempts,
                "previous_verdict": check.verdict.value,
                "guidance_used": check.suggested_fix,
            })

            redrafted = draft_bullets(
                entry, [requirement], max_count=1, guidance=check.suggested_fix,
                use_inline_bold=use_inline_bold,
            )
            bullet = redrafted[0]["bullet"]
            check = check_grounding(_for_grounding_check(bullet), entry["description"])
            log_event(run_id, "grounding_check", {
                "section_name": section_name,
                "requirement": requirement,
                "attempt": attempts,
                "bullet": bullet,
                "verdict": check.verdict.value,
                "unverified_claims": check.unverified_claims,
            })

        if check.verdict == GroundingVerdict.VERIFIED:
            slots.append({"status": "final", "bullet": bullet, "requirement": requirement})
        else:
            slots.append({
                "status": "gap",
                "requirement": requirement,
                "reason": check.suggested_fix or "Not supported by project data.",
            })

    final_bullets = [s["bullet"] for s in slots if s["status"] == "final"]
    gaps = [
        {"requirement": s["requirement"], "reason": s["reason"]}
        for s in slots if s["status"] == "gap"
    ]

    return {"slots": slots, "final_bullets": final_bullets, "gaps": gaps}


def run_executor(
    resume_sections: list[dict],
    jd_text: str,
    grounding_enabled: bool = True,
    untailored_text: str = "",
    use_inline_bold: bool = False,
) -> tuple[list[dict], str, "FitScore", "JDRequirements"]:
    """
    The full pipeline, start to finish, given a fake/real resume
    structure and a raw JD. Returns (results, run_id) — the run_id lets
    the caller pull up the full reasoning trace via logger.read_run_log().

    grounding_enabled: passed straight through to run_section() for
    every section — see its docstring. Only ever False in the Phase
    3.8 ablation script.

    untailored_text: everything in the source resume that isn't an
    Experience/Project section (Skills, Education, ...) — passed
    straight through to the Fit Scorer so it judges the WHOLE resume,
    not just the sections this pipeline tailors. Empty for the fake-
    resume test path, since there's no real docx behind it.
    """
    run_id = new_run_id()
    print(f"\n[executor] === Starting run {run_id} ===")

    print("[executor] Stage: parsing job description...")
    jd = parse_jd(jd_text)

    print(f"[executor] Stage: assigning {len(resume_sections)} section(s)...")
    assignments = assign_projects_to_sections(resume_sections, jd)

    for a in assignments:
        log_event(run_id, "assignment_decision", {
            "section_name": a.section_name,
            "assigned_entry_id": a.assigned_entry_id,
            "was_replaced": a.was_replaced,
            "reason": a.reason,
        })

    all_requirements = jd.required_skills + jd.nice_to_have

    results = []
    for i, assignment in enumerate(assignments, 1):
        print(
            f"[executor] Stage: drafting+grounding section {i}/{len(assignments)} "
            f"('{assignment.section_name}' -> {assignment.assigned_entry_id})..."
        )
        entry = get_project_details(assignment.assigned_entry_id)
        bullet_count = next(
            s["bullet_count"] for s in resume_sections
            if s["section_name"] == assignment.section_name
        )

        section_result = run_section(
            run_id, assignment.section_name, entry, all_requirements, bullet_count,
            was_replaced=assignment.was_replaced,
            grounding_enabled=grounding_enabled,
            use_inline_bold=use_inline_bold,
        )
        print(
            f"[executor]   -> {len(section_result['final_bullets'])} verified, "
            f"{len(section_result['gaps'])} gap(s)"
        )

        log_event(run_id, "final_outcome", {
            "section_name": assignment.section_name,
            **section_result,
        })

        results.append({
            "section_name": assignment.section_name,
            "assigned_entry_id": assignment.assigned_entry_id,
            "was_replaced": assignment.was_replaced,
            "assignment_reason": assignment.reason,
            **section_result,
        })

    # Fit Scorer — the 3rd agent. Runs once, on the FINISHED resume as
    # a whole, after every section's drafting/grounding is done. This
    # is deliberately a separate judgment from the per-bullet grounding
    # check: even if every bullet is individually true, the resume as
    # a whole might still be a weak fit for this specific JD.
    print("[executor] Stage: scoring overall fit...")
    resume_text = format_resume_text(results, untailored_text)
    fit = score_fit(resume_text, jd)
    log_event(run_id, "fit_score", fit.model_dump())

    print(f"[executor] === Run {run_id} complete (fit: {fit.overall_score}) ===\n")
    return results, run_id, fit, jd


def run_executor_on_docx(resume_path: str, jd_text: str, output_path: str) -> tuple[list[dict], str, "FitScore"]:
    """
    The FULL real pipeline: reads an actual resume file, matches its
    sections to cached entries, runs the same draft/check/retry loop as
    run_executor(), and writes a tailored COPY of the resume to
    output_path — never touching resume_path itself.

    Sections that don't match any cached entry are left completely
    untouched in the output (see match_sections_to_entries.py's
    docstring for the known simplification this represents).
    """
    doc_sections, untailored_text = read_docx_sections(resume_path)
    matches = match_sections_to_entries(doc_sections)

    # Learn THIS resume's own title-line formatting (tech-stack brackets,
    # whether it links projects at all) from ALL its title lines, before
    # anything gets edited — see style_profile.py's module docstring for
    # why this must never be hardcoded per-project.
    all_title_indices = [(s["title_index"], s["entry_type"]) for s in doc_sections]
    style_by_type = detect_title_line_style(resume_path, all_title_indices)
    all_bullet_indices = [i for s in doc_sections for i in s["bullet_indices"]]
    bullet_bold = detect_bullet_bold_style(resume_path, all_bullet_indices)

    resume_sections_input = []
    doc_sections_by_name = {}
    for s in doc_sections:
        entry_id = matches.get(s["section_name"])
        doc_sections_by_name[s["section_name"]] = s
        if entry_id is not None:
            resume_sections_input.append({
                "section_name": s["section_name"],
                "entry_type": s["entry_type"],
                "current_entry_id": entry_id,
                "bullet_count": s["bullet_count"],
            })
        # else: unmatched section, left out of the input entirely —
        # run_executor() never sees it, so it's never touched below.

    results, run_id, fit, jd = run_executor(
        resume_sections_input, jd_text, untailored_text=untailored_text,
        use_inline_bold=bullet_bold,
    )
    log_event(run_id, "style_profile_detected", {"by_type": style_by_type, "bullet_bold": bullet_bold})

    edits = {}
    title_renders = {}
    for r in results:
        doc_section = doc_sections_by_name[r["section_name"]]

        if r["was_replaced"]:
            new_entry = get_project_details(r["assigned_entry_id"])
            # Full structured render (title + whatever segments THIS
            # resume's own convention uses), not a plain text swap —
            # this is what stops a new project's title line from
            # inheriting a stale tech-stack/link run left over from
            # whatever project used to occupy this slot. See
            # docx_writer.render_title_line()'s docstring.
            title_renders[doc_section["title_index"]] = {
                "title": new_entry["title"],
                "skills": new_entry["skills"],
                "link": new_entry.get("link"),
                "style": style_by_type.get(doc_section["entry_type"], {}),
            }

        # Use "slots" (positionally correct, one entry per ORIGINAL
        # bullet_indices position) — NOT "final_bullets" (which drops
        # position info whenever a middle draft fails). See run_section's
        # docstring for why this distinction matters.
        #
        # IMPORTANT: len(slots) can be SHORTER than len(bullet_indices).
        # draft_bullets is allowed to write fewer bullets than the slot
        # count when there isn't enough distinct evidence to fill every
        # slot — this is intentional, not a bug, for a KEPT section. For
        # a REPLACED section, require_exact_count asks for the full
        # count but doesn't hard-guarantee it either. Either way, an
        # index with no corresponding slot means "nothing was even
        # attempted here" — handled the same as an explicit gap below.
        for i, bullet_index in enumerate(doc_section["bullet_indices"]):
            if i >= len(r["slots"]):
                if r["was_replaced"]:
                    edits[bullet_index] = _fallback_bullet_for_entry(
                        get_project_details(r["assigned_entry_id"])
                    )
                continue

            slot = r["slots"][i]
            if slot["status"] == "final":
                edits[bullet_index] = slot["bullet"]
            elif r["was_replaced"]:
                # CRITICAL: this section's project CHANGED. Leaving this
                # slot out of `edits` would keep the OLD project's
                # original text here — mixing two different projects'
                # content under one (new) title. Must explicitly fill
                # it with a safe fallback instead of leaving it blank —
                # a visibly empty bullet is worse than a modest, true
                # one. (draft_bullets was told to fill every slot when
                # was_replaced=True, so this should be rare — it's a
                # last-resort safety net, not the expected path.)
                edits[bullet_index] = _fallback_bullet_for_entry(
                    get_project_details(r["assigned_entry_id"])
                )
            # else: section was NOT replaced, so an untouched slot still
            # honestly describes the SAME project — safe to leave out of
            # `edits` entirely and keep the original bullet text as-is.

    write_tailored_docx(resume_path, output_path, edits, title_renders=title_renders, bullet_bold=bullet_bold)

    log_event(run_id, "docx_write", {
        "source": resume_path,
        "output": output_path,
        "paragraphs_edited": len(edits) + len(title_renders),
    })

    # --- Outer loop: fit-score-triggered recovery (Phase 3.5b) ---
    # Capped at ONE section reassignment per run. Recovery only ever
    # replaces a WHOLE section's project (never a single bullet with a
    # foreign entry's content) — see recovery.py's docstring for why.
    #
    # IMPORTANT: recovery is an OPTIONAL enhancement, not core to the
    # pipeline. By this point, every section has already been drafted,
    # grounded, and the resume has been written to output_path — a
    # complete, correct result already exists. If recovery fails for
    # ANY reason, that success must not be thrown away. Wrap the whole
    # thing so a failure here degrades to "skip recovery, keep the
    # already-good result" rather than crashing the entire run.
    if fit.unaddressed_requirements:
        try:
            used_ids = {r["assigned_entry_id"] for r in results}
            unused_entries = [c for c in list_projects() if c["id"] not in used_ids]

            if unused_entries:
                print("[executor] Stage: attempting gap recovery...")
                decision = attempt_recovery(
                    fit, results, resume_sections_input, unused_entries, jd
                )
                log_event(run_id, "recovery_decision", decision.model_dump())

                if decision.section_name is not None:
                    doc_section = doc_sections_by_name[decision.section_name]
                    new_entry = get_project_details(decision.new_entry_id)

                    new_section_result = run_section(
                        run_id, decision.section_name, new_entry,
                        jd.required_skills + jd.nice_to_have,
                        doc_section["bullet_count"], was_replaced=True,
                        use_inline_bold=bullet_bold,
                    )

                    # Update this section's entry in `results` so the final
                    # return value reflects the recovery, not the original.
                    for idx, r in enumerate(results):
                        if r["section_name"] == decision.section_name:
                            results[idx] = {
                                "section_name": decision.section_name,
                                "assigned_entry_id": decision.new_entry_id,
                                "was_replaced": True,
                                "assignment_reason": decision.reason,
                                **new_section_result,
                            }
                            break

                    # Rebuild this section's edits using the SAME safety
                    # rules as the first pass (fill any leftover gap slot
                    # with a safe fallback bullet, since the project
                    # changed again). Recovery is always a "replaced"
                    # section, so ANY unfilled index — whether a real gap
                    # slot or simply missing because slots came up
                    # shorter than bullet_indices — gets the same
                    # fallback, never left blank or with stale content
                    # from before the recovery.
                    title_renders[doc_section["title_index"]] = {
                        "title": new_entry["title"],
                        "skills": new_entry["skills"],
                        "link": new_entry.get("link"),
                        "style": style_by_type.get(doc_section["entry_type"], {}),
                    }
                    for i, bullet_index in enumerate(doc_section["bullet_indices"]):
                        if i >= len(new_section_result["slots"]):
                            edits[bullet_index] = _fallback_bullet_for_entry(new_entry)
                            continue
                        slot = new_section_result["slots"][i]
                        edits[bullet_index] = (
                            slot["bullet"] if slot["status"] == "final"
                            else _fallback_bullet_for_entry(new_entry)
                        )

                    write_tailored_docx(resume_path, output_path, edits, title_renders=title_renders, bullet_bold=bullet_bold)
                    log_event(run_id, "docx_write_after_recovery", {
                        "output": output_path,
                        "recovered_section": decision.section_name,
                    })

                    # Re-score against the NOW-updated resume so the
                    # returned fit accurately reflects what's actually in
                    # the final file, not the pre-recovery version. Must
                    # carry untailored_text through here too, or recovery
                    # would silently regress to scoring off an incomplete
                    # resume even though the first pass got it right.
                    resume_text = format_resume_text(results, untailored_text)
                    fit = score_fit(resume_text, jd)
                    log_event(run_id, "fit_score_after_recovery", fit.model_dump())

        except Exception as e:
            print(f"[executor] Recovery step failed, skipping it: {e}")
            log_event(run_id, "recovery_failed_skipped", {"error": str(e)})
            # Deliberately no re-raise — results/fit/output_path from
            # BEFORE this block are already valid and already written
            # to disk. A failed recovery attempt should never erase that.

    return results, run_id, fit


# if __name__ == "__main__":
#     fake_resume_sections = [
#         {"section_name": "Project 1", "entry_type": "project", "current_entry_id": "proj_003", "bullet_count": 2},
#         {"section_name": "Project 2", "entry_type": "project", "current_entry_id": "proj_002", "bullet_count": 3},
#     ]

#     sample_jd = """
#     Senior Backend Engineer

#     Requirements:
#     - 5+ years of experience with Python
#     - Experience designing and building REST APIs at scale
#     - Familiarity with automated testing and CI/CD practices
#     - Bachelor's degree in Computer Science or equivalent experience
#     """

#     print("=" * 60)
#     print("TEST 1: run_executor() with fake resume structure")
#     print("=" * 60)
#     results, run_id, fit, jd = run_executor(fake_resume_sections, sample_jd)

#     print(f"Run id: {run_id} (full trace saved to data/logs/{run_id}.jsonl)")

#     for r in results:
#         print(f"\n=== {r['section_name']} ===")
#         print(f"Entry: {r['assigned_entry_id']} (replaced: {r['was_replaced']})")
#         print(f"Reason: {r['assignment_reason']}")
#         print(f"\nFinal bullets:")
#         for b in r["final_bullets"]:
#             print(f"  - {b}")
#         if r["gaps"]:
#             print(f"\nHonest gaps:")
#             for g in r["gaps"]:
#                 print(f"  - [{g['requirement']}]: {g['reason']}")

#     print(f"\n--- Fit Score ---")
#     print(f"Overall: {fit.overall_score}")
#     print(f"Strong areas: {fit.strong_areas}")
#     print(f"Weak areas: {fit.weak_areas}")
#     print(f"Unaddressed: {fit.unaddressed_requirements}")
#     print(f"Honest note: {fit.honest_note}")

#     print("\n\n" + "=" * 60)
#     print("TEST 2: run_executor_on_docx() — the REAL end-to-end test")
#     print("=" * 60)
#     results2, run_id2, fit2 = run_executor_on_docx(
#         resume_path="data/test_resume_A_headings.docx",
#         jd_text=sample_jd,
#         output_path="data/test_resume_A_TAILORED_FULL.docx",
#     )
#     print(f"Run id: {run_id2}")
#     print(f"Tailored resume saved to: data/test_resume_A_TAILORED_FULL.docx")
#     for r in results2:
#         print(f"\n=== {r['section_name']} ===")
#         print(f"Entry: {r['assigned_entry_id']} (replaced: {r['was_replaced']})")
#         for b in r["final_bullets"]:
#             print(f"  - {b}")
#         if r["gaps"]:
#             for g in r["gaps"]:
#                 print(f"  - [GAP, original bullet kept] [{g['requirement']}]: {g['reason']}")

#     print(f"\n--- Fit Score ---")
#     print(f"Overall: {fit2.overall_score}")
#     print(f"Strong areas: {fit2.strong_areas}")
#     print(f"Weak areas: {fit2.weak_areas}")
#     print(f"Unaddressed: {fit2.unaddressed_requirements}")
#     print(f"Honest note: {fit2.honest_note}")

#     # Recovery's outcome only ever gets logged, never printed above —
#     # pull it back out of the run's own log so it's actually visible
#     # here, instead of silently happening (or not happening) off-screen.
#     from logger import read_run_log
#     events = read_run_log(run_id2)
#     recovery_events = [e for e in events if e["event_type"].startswith("recovery") or e["event_type"] == "fit_score_after_recovery"]

#     print(f"\n--- Recovery ---")
#     if not recovery_events:
#         print("Recovery was never triggered (fit.unaddressed_requirements was empty).")
#     else:
#         for e in recovery_events:
#             if e["event_type"] == "recovery_decision":
#                 if e.get("section_name"):
#                     print(f"Recovery PROPOSED: reassign '{e['section_name']}' -> {e['new_entry_id']}")
#                     print(f"Reason: {e['reason']}")
#                 else:
#                     print(f"Recovery attempted, but found NO good match. Reason: {e['reason']}")
#             elif e["event_type"] == "fit_score_after_recovery":
#                 print(f"Fit score after recovery attempt: {e['overall_score']}")
#                 print("(fit2 above already reflects this post-recovery score, since "
#                       "run_executor_on_docx returns the FINAL fit, not the original)")