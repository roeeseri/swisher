# Swisher — private team prototype

Canonical repository: https://github.com/roeeseri/swisher

Team documents: https://drive.google.com/drive/folders/1Obp0c-fbrVR5MUCPlmz2-kpAsoyX6jSA

Code backup: https://drive.google.com/file/d/1NZbIV9aidaflLY5lrlBZT8H10zTWkuYt/view

Local basketball video review and coaching prototype. Observations and feedback require coach review. Software tests do not establish coaching accuracy or player improvement.

## Run

Use Python 3.11 or 3.12:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m swisher.studio.server --port 8770 --root /absolute/path/to/private-swisher-data
```

Open http://127.0.0.1:8770/coach for the player flow and http://127.0.0.1:8770/ for coach review. Register an anonymous player and consent before uploading a clip. Model weights download separately on first inference; network access is required. No API keys are needed. `.env.example` documents that dotenv is not automatically loaded.

Alternative interface: `python -m streamlit run app.py`.

## Structure

- `src/swisher/`: current analysis, storage, feedback and local Studio.
- `tests/`: automated tests, synthetic media and mocked inference.
- `app.py`, `pages/`, `components/`: Streamlit interface.
- `scripts/`: launch, evaluation and reporting tools.
- `docs/`: MVP plan and historical technical documents.
- `archive/`: older source variants for reference; excluded from runtime and current tests.

## Verify

```sh
python -m pytest -q
python -m ruff check app.py src tests pages scripts
```

Read [RUN_MVP.md](RUN_MVP.md), [CONTRIBUTING.md](CONTRIBUTING.md), and [recovery notes](docs/RECOVERY.md). Historical documents may describe superseded dependencies or previous validation; use pyproject.toml for current requirements.

## Team workflow

Use GitHub branches and pull requests for code. Use Google Drive for documents and dated ZIP backups. Invite named collaborators individually; keep access restricted.

Never commit recordings, player information, consent forms, measurements, credentials, `.env`, model weights or runtime workspaces. Store private runtime data outside this repository. This is a local prototype; do not expose the server to the public internet.
