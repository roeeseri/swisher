# Feedback audit and correction — 2026-10-20

## Finding

The Studio's previous `suggestion()` returned one hard-coded knee cue. It was not a learned feedback prediction. The active workspace contained 54 identical available suggestions, 153 abstentions and no coach reviews.

Search covered the active and desktop Swisher checkouts, the September backup, Git history/remote branch inventory, model/checkpoint and notebook paths under Desktop, Documents, Downloads and mirrored ChatGPT projects. The broader filename search found 274 model/notebook files; none of the matching paths or relevant notebook code contained a Swisher coaching model. Unrelated models were not loaded. The basketball cache contains RF-DETR Nano; the project contains MediaPipe Heavy/Lite pose assets. No accessible trained coach-feedback checkpoint, training notebook or supervised coach-feedback labels were found. This is a bounded search finding, not a claim about inaccessible cloud storage or every disk.

## Correction

`studio/feedback.py` consumes the existing pretrained model's per-frame pose trace and proposed release frame, without rerunning vision or training weights. It produces descriptive, coach-reviewed observations with stored numerical features and source frames:

- Knee motion from one consistently visible side, rather than pooled body sides.
- Torso motion only when shoulders and hips are visible.
- A visible arm's post-release hold from contiguous frames. Missing visibility cannot be interpreted as hand drop. A censored hold is explicitly a lower bound.
- Within-video comparisons only with at least five reliable release proposals and no camera-mapping failures. References are unreviewed proposals, never assumed successful shots or a verified personal shooting baseline.
- Each feature has visibility and temporal-support gates. A missing release does not suppress independent visible knee/torso observations.

The branches and thresholds are experimental rules over model outputs. They are not learned diagnoses, calibrated probabilities, a 3D biomechanical assessment or evidence of why a shot missed. Similar measured movements may legitimately produce identical comments. No cosmetic randomization was added.

## Current result

207 unreviewed proposals updated: 106 available observations and 101 abstentions; 65 distinct texts. Focus counts: torso motion 48, knee motion 34, visible-arm hold 18, torso observation 4 and within-video knee variation 2. Counts measure output availability, not accuracy or usefulness. The knee timing wording describes the interval around the estimated release.

The original suggestions were backed up and retained in per-shot history. Existing reviewed shots are not automatically replaced. Review saves capture the suggestion actually shown to the coach. Imports from older offline packs resolve approved text against retained suggestion history, so an old approval cannot approve a different newly generated note.

The desktop display refresh key includes the feedback method/text. New uploads use the new engine; manually added shots without pose evidence abstain. The coach handoff ZIP and owner snapshots are regenerated from the updated workspace. Registered old packs remain importable.

## Validation

66 tests passed, including contiguous hand-hold measurement, visibility-gap abstention, no-release independent measurements, low coverage, relative comparison support, censored lower bounds and preservation of an old approved note after a suggestion update. These are implementation tests; coach validation remains outstanding. Browser checks confirmed that separate candidates display the new observations.
