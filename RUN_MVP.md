# Run the Swisher MVP

Python 3.11–3.12 required. From this directory:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m swisher.studio.server --port 8770
```

Open http://127.0.0.1:8770/coach. Create an anonymous player and record consent,
upload a fixed-camera clip, then open coach review (`/`) to approve/correct events
and feedback. Refresh the player profile and play approved feedback. Start a drill
and count shots manually. Optional microphone commands depend on browser support.
The first real analysis needs pretrained RF-DETR and MediaPipe model assets;
downloads require network. Missing assets/inference failures do not create fake data.

```sh
.venv/bin/python -m pytest -q
```

See `docs/MVP_12_WEEKS.md` for architecture, validation targets and limitations.
Use a dedicated private `--root` directory. Local operator use only.
