# resume-tailor-agent

A multi-agent system that tailors your resume to a job description while keeping every claim grounded in your actual experience.

The system doesn't just rewrite your resume with an LLM. It uses separate agents for **planning, drafting, independent verification, and resume-level evaluation**, with feedback from one stage determining what happens next.

Every generated claim is independently checked against your real project data before it is accepted, and the system tells you honestly when it cannot support a requirement rather than guessing.

## What it does

Give it your projects/experience document and a resume (`.docx`). For any job description, the system:

- Parses the job description and identifies its requirements.
- Decides which projects or experiences best fit each resume section.
- Swaps out weak project matches when a better unused project exists.
- Drafts new bullets targeting the job description.
- Independently verifies every generated bullet against the source project data.
- Revises and re-checks bullets when they are overstated or unsupported.
- Preserves honest gaps when a claim cannot be grounded in the available evidence.
- Scores the completed resume against the job description as a whole.
- Can trigger one recovery pass if important requirements remain unaddressed, replacing a weak section with a better-fit unused project and rebuilding it.

## How it works

```text
              Job Description
                     +
          Resume + Project Data
                     │
                     ▼
          ┌─────────────────────┐
          │   Parse JD/Resume   │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Assign Projects    │
          │  to Resume Sections │
          └──────────┬──────────┘
                     │
                     ▼
              ┌────────────┐
              │  Executor  │
              │   Drafts   │
              │   Bullets  │
              └─────┬──────┘
                    │
                    ▼
          ┌────────────────────┐
          │  Grounding Critic  │
          │  Independently     │
          │  verifies each     │
          │  generated bullet  │
          └─────────┬──────────┘
                    │
             ┌──────┴──────┐
             │             │
          VERIFIED     NOT VERIFIED
             │             │
             ▼             ▼
        Accept bullet   Use critic's
             │          feedback to
             │          revise bullet
             │             │
             │             ▼
             │       Re-check bullet
             │             │
             │      ┌──────┴──────┐
             │      │             │
             │   VERIFIED     Still fails
             │      │             │
             │      │             ▼
             │      │        Honest gap
             │      │             │
             └──────┴─────────────┘
                           │
                           ▼
                Complete all sections
                           │
                           ▼
                 ┌─────────────────┐
                 │  Assemble the   │
                 │  tailored resume│
                 └────────┬────────┘
                          │
                          ▼
                    ┌───────────┐
                    │ Fit Scorer│
                    │ Whole     │
                    │ Resume    │
                    └─────┬─────┘
                          │
                 Unaddressed requirements?
                     /             \
                   No              Yes
                   │                │
                   ▼                ▼
                  Done       ┌────────────────┐
                             │ Recovery       │
                             │ Decision       │
                             │ Find a better  │
                             │ unused project │
                             └───────┬────────┘
                                     │
                             Better match exists?
                                /          \
                              No            Yes
                              │              │
                              ▼              ▼
                         Report gap    Reassign section
                                            │
                                            ▼
                                     Draft + Ground again
                                            │
                                            ▼
                                      Rewrite resume
                                            │
                                            ▼
                                         Re-score
                                     (max 1 recovery)
```

## The agents

### 1. Executor

The Executor handles the main planning and drafting workflow. It determines which project or experience entry should fill each resume section and generates candidate bullets targeting the job description's requirements.

It then coordinates the verification loop: when a generated bullet fails grounding, the Executor uses the critic's feedback to request a revised version.

### 2. Grounding Critic

The Grounding Critic independently fact-checks every generated bullet against the source project description.

This is an entailment/faithfulness check rather than a "does this sound good?" check.

Each bullet receives one of four verdicts:

- `verified` — the claim is supported and can be accepted.
- `overstated` — the claim has a real basis but exaggerates what was actually done.
- `unsupported` — the source project data does not provide enough evidence for the claim.
- `fabricated` — there is no real basis for the claim.

For `overstated`, `unsupported`, or `fabricated` bullets, the critic provides revision guidance. The Executor then redrafts the bullet and sends it through grounding again, with a maximum of two retries.

If the bullet still cannot be verified, it becomes an honest gap rather than being forced into the resume.

### 3. Fit Scorer

The Fit Scorer evaluates the **finished resume as a whole** against the job description.

This is intentionally different from the Grounding Critic: a resume can contain only truthful bullets and still be a poor match for a particular job.

If the Fit Scorer identifies important unaddressed requirements, the system can trigger one recovery attempt. The recovery logic evaluates unused projects and can reassign an entire resume section to a better-fit project.

That section is then drafted and grounded again before the resume is rewritten and re-scored.

## What makes it agentic

This isn't a single LLM call that rewrites a resume, or a fixed sequence where every model is called once. The output of one agent determines what the system does next. The Executor drafts a bullet, the Grounding Critic evaluates it, and its verdict and revision guidance determine whether the bullet is accepted, revised and checked again, or recorded as an honest gap. After all sections are completed, the Fit Scorer evaluates the entire resume; if it finds important unaddressed requirements, that result can trigger a recovery decision to replace a weak project with a better unused match, rebuild the section, and score the resume again. These decision loops are bounded grounding gets at most two retries and fit-based recovery happens at most once — so the system has autonomous feedback-driven behavior without uncontrolled looping.


## Setup

```bash
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env   # add your GROQ_API_KEY

streamlit run app.py
```
