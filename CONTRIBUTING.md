# Working together on Swisher

Read `docs/SWISHER_PARTNER_BRIEF_HE.md` first. The Python application is a local
pilot, not a deployed multi-team service. Camera recording and coaching accuracy
still require field acceptance with the team.

## Setup (currently verified on Apple Silicon)

Use Python 3.11 or 3.12. From the repository directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip check
python -m pytest -q
python -m streamlit run app.py
```

First inference downloads official model weights. Record one short clip before
filming a full session. Linux/Windows and each partner's camera/browser need their
own acceptance checks; a passing unit test suite does not validate vision accuracy.

## Changes

- Work on a task branch. Open a pull request; do not force-push shared branches.
- State the player/coach value, one-year feasibility and verification method.
- Separate observations, model estimates and coach-approved feedback.
- Add regression tests for bugs. Report mocks and untested hardware explicitly.
- Link schema/model changes to an evaluation and a migration plan.
- Ask another partner to review measurement or feedback changes before merging.

## Never commit

Player recordings, consent forms, identifying manifests, model weights, access
tokens or private keys. `.gitignore` is a guardrail, not a privacy audit. Check the
staged diff before every commit. Team dataset access is separate from code access.

## Local verification

```bash
python -m pytest -q
python -m ruff check app.py src tests pages scripts/benchmark_pose.py
git diff --check
git diff --cached --stat
```

The optional PDF builder needs reportlab and python-bidi, not runtime dependencies.
The PDF is a snapshot; change its Markdown source and regenerate it when decisions
change. Historical model cards do not supersede the current partner brief.
