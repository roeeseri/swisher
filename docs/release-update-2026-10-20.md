# Release and camera tracking experiment

The current change adds shooter-scale ball detections, a visible-wrist separation
rule, and anchor-to-frame camera motion estimation. It does not train new weights
or establish shooting-technique accuracy. All shot windows and release intervals
remain proposals requiring review.

## Observed failures

On IMG_1083, a dense full-frame scan still produced no releases. Logged detections
missed the ball around hand contact, and the former release rule rejected frames
when either wrist was obscured. In IMG_1075, camera movement shifted the shooter
out of the original region and a different player could enter it.

## Changes

- Keep full-frame person detection at the configured stride. Run the same ball
  detector on the shooter crop at source frame rate while a visible hand is high.
- Associate ball candidates with visible hands; multiple balls elsewhere in the
  image no longer discard the entire frame.
- Require a nearby ball followed by upward separation and local temporal support.
  Normalize distances by the larger of shoulder width and half the visible torso
  length, avoiding scale collapse when projected shoulders align.
- A single visible hand may support a proposed separation interval. Shooting
  handedness remains unknown in that case and hand-specific release metrics are
  withheld.
- Map the selected region with robust background feature matches from the first
  frame. Failed mappings abstain rather than reusing a stale region.
- Render review-required event markers separately from the original overlay. The
  smaller preview preserves source frame count, frame rate, and event numbering.

## Evaluation protocol and limits

Reference intervals are provisional assistant visual annotations guided by old
proposals. They are not exhaustive, independent full-video labels and are not
coach approved. IMG_1083 was used for development. IMG_1078 and IMG_1068 were used
as video holdouts, with nine reviewed positive intervals each and reviewed
negative examples. Do not treat these as held-out player identities.

Matching uses one-to-one interval overlap with the visual reference. Predictions
outside reviewed windows remain unverified rather than automatically becoming
false positives. The bundled pipeline changes were evaluated together; this is
not an ablation isolating each change's effect.

The reproducible comparison is written to
`artifacts/club_pilot/updated/evaluation/comparison.json`. Baseline reports remain
in `artifacts/club_pilot/`; updated reports are in its `updated/` subdirectory.
The original videos are read from the user's data folder and remain unchanged.

This work does not validate make/miss, causal explanations of misses, real-world
coaching effectiveness, or generalization to new courts and players.
