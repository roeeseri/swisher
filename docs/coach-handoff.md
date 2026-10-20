# Private coach handoff

Build a self-contained review pack from a completed Studio workspace:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_coach_pack.py \
  --output artifacts/coach_delivery_2026-10-20/Swisher_Coach_Review
```

The output ZIP contains an `index.html`, short Hebrew instructions, 207 H.264 shot clips and unmarked release/motion stills. It does not contain full original recordings, credentials, certificates, source code or local source paths. The count follows the workspace and is not hard-coded by the exporter. The existing proposed notes remain unverified and clearly require coach approval.

Coaches extract the entire ZIP, open `index.html` in a desktop browser, choose a coach identifier, review shots and export a JSON feedback file. Offline review stores progress in that browser, with a memory fallback and a warning when persistent storage is unavailable. A clean browser, private browsing, moving the pack or clearing storage may lose local progress; coaches should export after each work block. Clip/frame boundary editing is deliberately unavailable in this offline package because the full source recordings are not included. Coaches can correct notes/outcomes and reject false proposals.

The export is registered under `data/studio/coach_packages.json`. Retain this registry together with the master workspace. When a coach returns JSON, run the local Studio and import through its authenticated loopback API:

```bash
PYTHONPATH=src .venv/bin/python scripts/import_coach_feedback.py /absolute/path/Swisher_feedback_coach.json
```

Import validates the package registration, source hashes, immutable candidate identity and coach decisions. It appends review history, preserves original model suggestions and skips already imported review IDs. Failed imports restore live shot objects in place so the running worker does not lose its references. Labels are not silently overwritten or used to retrain weights.

The handoff ZIP contains footage of minors and is intended only for authorized project coaches. No upload or message is sent by these scripts. Runtime packages, returned labels, raw media and owner snapshots remain ignored by Git.

Validation: the standalone JavaScript adapter was exercised independently for state restoration, a coach correction, stale-edit rejection and export identity. Native `file://` navigation could not be verified by the assistant's browser tool because its URL policy permits HTTP(S) only. Opening the extracted HTML manually in the coach's chosen desktop browser remains the first recipient acceptance check. The normal desktop app's HTTP interface has separate browser acceptance coverage.
