# Swisher Golden Dataset v1

The production accuracy work starts with coach-verified ground truth. Each clip
must have an immutable source video and a versioned JSON annotation containing:

- anonymous player and session identifiers;
- shot start, release and shot end frames;
- made, missed or unknown outcome;
- player, ball and rim boxes for detector training;
- one primary coach diagnosis, cue and confidence;
- camera view, resolution, FPS and consent status.

## Acceptance targets

Release is evaluated as absolute frame error at 60 FPS. The Year-1 gate is a
median error of at most two frames and a 95th percentile error of at most five
frames on a player-disjoint test set.

Detector gates on the same held-out set:

- primary player recall >= 98%;
- visible-ball recall >= 85%;
- rim recall >= 95%;
- false-positive rate low enough that incorrect coaching occurs in < 1% of
  accepted sessions.

Outcome accuracy is reported only on clips where ball and rim trajectories pass
their quality gates. Coverage and accuracy must be published together.

## Split rule

All clips from the same player and original recording session belong to exactly
one of train, validation or test. This prevents near-duplicate leakage.
