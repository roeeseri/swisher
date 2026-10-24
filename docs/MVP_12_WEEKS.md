# Swisher investor vertical slice — 23 October 2026

## Architecture and actual delivery

Existing Swisher Python repository reused, not rewritten. Run the local Studio server;
`/coach` is the player training flow and `/` is the coach review workspace.

Video upload → permission gate → bounded background queue → RF-DETR person/ball
observations + MediaPipe pose and segmentation → frame-level provenance → proposed
shots → coach correction of boundaries/release/outcomes/cues → descriptive player
baseline → bounded spoken feedback and drill commands.

The local JSON store contains pseudonymous player IDs, permission attestations,
video jobs and versioned reviews. A player baseline uses only the latest accepted
review, recomputes measurements using corrected boundaries and release frame,
and groups by capture/job, camera view and visible body side. Five observations
make a descriptive baseline available; this threshold is a product choice,
not evidence of statistical or professional validity. Baselines are not pooled
across captures until camera calibration/repeatability is established.

Implemented in this slice: local upload and real inference pipeline inherited
from source; segmentation comparison integrated into frame traces; player permission
records; descriptive medians/MAD with source shot IDs; manual five-shot drill;
browser speech synthesis and opt-in speech command recognition; coach-approved
cue playback; deletion of the dedicated local workspace while idle.
No mock video results are injected. Synthetic fixtures test logic only.

Not validated: shot detector precision/recall, 2D metric error, segmentation
accuracy, coaching effectiveness, real-time latency, browser audio on this machine.
Not implemented: live shot-driven voice, conversational LLM, personalized trained
model, multiuser permissions, remote portal or production minor onboarding.
The current coach identity is an operator-entered label, not authenticated identity.

## Skeleton versus body envelope

Every analyzed crop requests a segmentation mask from the pretrained pose task.
`body_envelope` in pose.jsonl records foreground extent/fraction, landmark extent
and visible landmark inclusion. The mask is thresholded at 0.5; boundary certainty
is an uncalibrated diagnostic, not correctness probability. Both outputs share
a backbone, so agreement cannot establish independent accuracy. No correction
is derived from the mask yet; no body-shape diagnosis is made.

Study: manually trace masks on 200 consented frames stratified by angle, loose
clothing, occlusion, lighting and body size. Compare skeleton-only tracking vs
segmentation-assisted tracking using identity switches, valid-frame coverage,
manual-mask IoU, silhouette boundary F1 and downstream angle errors. Report
paired results and bootstrap intervals by player, not randomly mixed frames.
Do not claim segmentation improves coaching before this comparison passes.

## Twelve-week delivery gates (proposed targets, not measured results)

| Weeks | Delivery | Acceptance evidence |
|---|---|---|
| 1–2 | Local slice, privacy procedure, fixed-camera capture guide | Upload→review→profile→voice smoke; no unreviewed diagnosis |
| 3–4 | Coach-labelled evaluation set, body-envelope comparison | 200 reviewed shots + 200 traced frames; split held-out players |
| 5–6 | Tune segmentation and shot proposals | Event precision ≥90%, recall ≥85%, event match ±0.5s; coach labels |
| 7–8 | Calibrate repeatable 2D measurements, personal consistency | Median angle error ≤8° vs dual-rater 2D labels; coverage and abstention reported |
| 9–10 | Streaming prototype + voice timing | Measure p50/p95 shot-to-cue latency; target p95 <3s on declared hardware |
| 11–12 | Coach-supervised demo, before/after pilot | No unsupported corrective cues in 100 audited turns; show failures and denominator |

All targets require real measurement. Use player-separated evaluation, two coach
raters with adjudication, shot-event TP/FP/FN, release-frame MAE, angle MAE,
mask IoU, abstention/coverage by subgroup, cue approval rate, and latency p50/p95.
Track 20–30 baseline shots → one coach-approved cue/drill → 20–30 follow-up shots.
A short pilot is feasibility evidence; it cannot establish causal improvement.

Small-team/year gate: fixed-camera review and descriptive feedback are bounded.
Player-benefit gate: one actionable coach-approved cue per set, with reviewed
follow-up. Proof gate: predeclared held-out metrics and coach agreement before
expanding into automatic live coaching or a group portal.

## Privacy and operation

Default binding is loopback only. Do not enable LAN phone mode with minors until
access controls, guardian workflow and transport are reviewed. No video or child
identifying details were copied into this checkout. All subjects in uploads must
have permission; minors require a recorded guardian reference. This is an operator
attestation, not legal consent verification. Use nicknames and single-player clips.

Pilot retention policy: operator deletes data at pilot end or within 30 days,
whichever is earlier; automatic expiry is not implemented. The local erase action
removes jobs, uploads, derived clips, reviews, cached exports and profiles within
the dedicated workspace. External files, exported copies and backups need separate
erasure. Do not import external pilot sources for a self-contained deletion demo.
Browser speech recognition may send audio to the browser vendor: leave microphone
off unless that processing is approved. Typed commands and local speech output
remain available. No transcript/audio is stored by the training API.

Production gate: authenticated coach/guardian roles, scoped access, encryption,
formal consent/revocation, expiry/deletion jobs, isolated media processing, audit
and a reviewed deployment threat model. This local demo is not a hosted child app.
