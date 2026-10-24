# Swisher Vision Model Card v0.2

## Objective

Provide a transparent first measurement layer for fixed-camera basketball videos.

## Model

RF-DETR Nano detects the player and ball. MediaPipe Pose Heavy runs on a padded,
temporally smoothed player crop. Swisher adds quality abstention, movement-window
estimation, robust 2D joint geometry, result contracts and provenance.

Grounding DINO Tiny proposes a rim location once per fixed-camera clip. A strict
confidence and geometry gate rejects weak proposals. Release is estimated only
when ball-to-wrist separation grows while the ball moves upward. Make or miss is
reported only for a clear downward crossing of the rim plane; otherwise the
system returns `unknown`.

## Appropriate use

- Short, fixed-camera videos.
- One primary player.
- Entire body visible.
- Product and data-pipeline experiments.

## Inappropriate use

- Clinical or professional biomechanical diagnosis.
- Claims about why a shot missed.
- Accurate 3D measurements.
- Automatic coaching without human validation.

## Validation gates before coaching

- Compare joint angles against manually reviewed reference frames.
- Validate event windows against coach-labelled shot start/release/end.
- Measure abstention and failure rates by camera view and lighting.
- Do not enable advice until a coach agrees with recommendations on a held-out set.
