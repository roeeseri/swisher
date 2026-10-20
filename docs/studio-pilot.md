# Swisher Studio — capture and coach review

## Run locally

```bash
PYTHONPATH=src .venv/bin/python -m swisher.studio.server --port 8770
```

Open http://127.0.0.1:8770. On macOS, `scripts/start_studio.command` launches the same app.
This complements the existing Streamlit research workspace; it does not replace its historical sessions.

## Phone captures, computer processes

```bash
PYTHONPATH=src .venv/bin/python -m swisher.studio.server --port 8770 --phone
```

Or use `scripts/start_phone_studio.command`. Enabling this mode opens a paired HTTPS endpoint on the local network (8772) and a public-certificate-only setup page (8773). Both devices must share a trusted network. Do not forward these ports through a router. The computer dashboard remains on loopback.

The desktop **חיבור טלפון** button shows the actual addresses and an eight-digit single-use pairing code, valid for ten minutes. Owner-installed local CA trust is required on the phone for browser camera capture; nothing is installed into any trust store automatically. An iPhone configuration profile and Android CA certificate can be downloaded from the setup page. A root CA adds trust in certificates signed by this computer: only install on an owner-controlled test phone and remove after the pilot. Never bypass a browser certificate warning. Keys remain in ignored `data/studio/certificates/`, permission 0600. Server certificates last 14 days; the local CA lasts 30 days. This is a pilot connection, not production authentication or cloud deployment.

Phones pair through HTTPS. Unpaired clients see only the pairing page, not videos or the desktop token. Pairing is rate limited, single use, and phone sessions expire after 24 hours or server restart. Restart invalidates codes/tokens. The setup HTTP server exposes only the public certificate/profile, never source videos or keys. A successful TLS/loopback test does not establish acceptance on the user's physical phone.

## Capture workflow

1. Fill in a pseudonymous player identifier and camera view; confirm permission to record and use the footage.
2. Open the camera, frame the entire shooter, and drag a selection around the shooter when other people are visible.
3. Start capture. Independent 30-second recordings upload automatically while capture continues. No audio is recorded. Uploaded segments enter a persistent serial analysis queue.
4. Stop capture, leave the page open until pending uploads finish, and review shots on the computer.

Pending/failed uploads remain in the current browser tab with retry/download options. They are not an offline durable store. Closing/reloading the tab may lose unuploaded segments; the page warns. Keep the phone screen active. Up to eight unacknowledged segments are retained before recording is stopped to avoid uncontrolled memory growth. Segment boundaries have a short recorder restart gap; a shot spanning boundaries can be truncated or missed, and proposals within one second of a boundary are flagged. There is no cross-segment stitching or deduplication yet. This is live ingestion with queued processing, not a verified real-time shooting detector. Device camera permissions, iOS/Android recorder behavior and network interruptions require physical-device acceptance.

Existing video files (MP4/MOV/WebM/M4V) up to 200MB and ten minutes can also be uploaded. Select the shooter on the preview before upload when the scene contains other players/posters. WebM is normalized while retaining its source, with an explicit ten-minute limit in Studio. The legacy recorder keeps its original two-minute limit.

## Coach review workflow

- One shot at a time, with an H.264 clip, a source release frame, slow playback and the original suggestion.
- When no release is detected, a motion frame is explicitly labeled as such; no release is invented.
- Approve an available suggestion, replace it with a coach comment, mark uncertainty, or reject a non-shot.
- Optionally label outcome, adjust clip bounds and release frame, or add a missed shot manually in source seconds.
- Changing clip bounds schedules regeneration; the model proposal is retained unchanged.
- Coach identifier, timestamps, all review versions and model suggestions are saved separately. Stale edits are rejected.
- Existing clips are unassigned until the user explicitly maps them to player identifiers via the activity screen.
- Export produces a JSON annotation dataset. Saving labels does **not** retrain model weights automatically.

## Model scope

The existing MediaPipe/RF-DETR pipeline generates shot/release proposals. The initial textual suggestion is a conservative rule derived from visible 2D knee measurements, available only with a release proposal and at least 70% valid pose coverage. It is a prompt for the coach to examine the clip, not a trained technique diagnosis, calibrated confidence or evidence of what caused a miss. Missing/low-quality data produce an abstention. No autonomous coaching or make/miss claim has been added.

## Existing pilot data

```bash
PYTHONPATH=src .venv/bin/python -m swisher.studio.server \
  --import-pilot artifacts/club_pilot/updated
```

Import reuses existing reports without rerunning vision, extracts 207 unverified candidate clips, and preserves their source frames. It does not infer player identities. Import is idempotent by report path. Runtime data, certificates, video clips, release images and coach labels stay under ignored `data/studio/`, not Git.

## Acceptance evidence

- Tests cover review history, validation, concurrent stale-edit protection, abstention, recovery after restart, H.264 clip frame count, API authorization, consent gating, manual shot creation, dataset export, pairing token separation and expiry.
- A real uploaded pilot excerpt with an explicit shooter region completed the actual vision pipeline, generated a clip and release proposal at local frame 34 (source frame 92), and accepted a replacement coach note through the browser in an isolated test workspace. This is an integration smoke test, not detection accuracy validation.
- Desktop browser verified H.264 playback (960×540), slow playback and release image display. Physical phone capture and uninterrupted multi-segment recording are not verified yet.

## Tracking in review media

Shot clips render the existing per-frame pose trace as cyan body edges and yellow joints; green rings mark ball observations. The release image uses the same exact-frame observations and a shooter crop when a tracked region exists. Hidden joints and frames without a ball observation are left unmarked. This does not add a verified continuous ball tracker or interpolate missing observations. Model proposals and raw source videos remain unchanged. The first rollout regenerates the 207 previews with these markings; no vision rerun is required.
