"""Shot-specific, auditable observations from pretrained pose outputs.

This is not a learned coaching classifier. Every message carries the measured
feature, exact source frames and method; missing visibility produces abstention.
"""

from __future__ import annotations

import math
from statistics import median

import numpy as np

from swisher.geometry import angle_deg

METHOD = "pose_temporal_observations_v2"
LIMITATION = "מדידות בהיטל התמונה; אין כאן אבחון סיבה להחטאה. נדרש אישור מאמן."


def visible(row, names, threshold=0.6):
    return bool(
        row.get("valid")
        and row.get("camera_tracking_valid", True)
        and all(
            row.get("visibility_by_name", {}).get(n, 0) >= threshold
            and n in row.get("xy", {})
            and all(math.isfinite(v) for v in row["xy"][n])
            for n in names
        )
    )


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def measure(candidate, records, fps, knee_side=None):
    start, end = candidate["shot_start"], candidate["shot_end"]
    rows = [r for r in records if start <= r["index"] <= end]
    reliable = [r for r in rows if r.get("valid") and r.get("camera_tracking_valid", True)]
    result = {
        "coverage": len(reliable) / max(1, end - start + 1),
        "frames": [start, end],
        "release_frame": candidate.get("release_frame"),
    }
    minimum = max(8, round(fps * 0.25))
    if len(reliable) < minimum or result["coverage"] < 0.7:
        return result
    release = candidate.get("release_frame")
    anchor = release if release is not None else candidate.get("motion_frame", (start + end) // 2)
    phase = [
        r
        for r in rows
        if max(start, anchor - round(fps * 0.8))
        <= r["index"]
        <= min(end, anchor + round(fps * 0.2))
    ]
    if knee_side is None:
        knee_side = max(
            ("l", "r"),
            key=lambda side: sum(visible(r, [side + "h", side + "k", side + "a"]) for r in phase),
        )
    names = [knee_side + "h", knee_side + "k", knee_side + "a"]
    knees = []
    for r in phase:
        if visible(r, names):
            value = angle_deg(*(np.asarray(r["xy"][n]) for n in names))
            if value is not None and 20 <= value <= 180:
                knees.append(value)
    if len(knees) >= minimum and len(knees) / max(1, len(phase)) >= 0.75:
        result.update(
            knee_side=knee_side,
            knee_min=percentile(knees, 0.05),
            knee_range=percentile(knees, 0.95) - percentile(knees, 0.05),
        )
    leans = []
    for r in phase:
        if visible(r, ["ls", "rs", "lh", "rh"]):
            xy = r["xy"]
            sx = (xy["ls"][0] + xy["rs"][0]) / 2
            sy = (xy["ls"][1] + xy["rs"][1]) / 2
            hx = (xy["lh"][0] + xy["rh"][0]) / 2
            hy = (xy["lh"][1] + xy["rh"][1]) / 2
            if hy - sy >= 20:
                leans.append(math.degrees(math.atan2(sx - hx, hy - sy)))
    if len(leans) >= minimum and len(leans) / max(1, len(phase)) >= 0.75:
        result["trunk_range"] = percentile(leans, 0.95) - percentile(leans, 0.05)
    if release is None or end < release + round(fps * 0.5):
        return result
    # Choose one consistently visible arm, not an assumed shooting handedness.
    by_index = {r["index"]: r for r in rows}
    eligible = []
    for side in ("l", "r"):
        names = [side + "s", side + "e", side + "w"]
        initial = [by_index.get(release + i, {}) for i in range(1, 4)]
        if all(visible(r, names) for r in initial):
            eligible.append(
                (sum(sum(r["visibility_by_name"][n] for n in names) for r in initial), side)
            )
    if not eligible:
        return result
    side = max(eligible)[1]
    names = [side + "s", side + "e", side + "w"]
    count = 0
    ended = False
    support = []
    for index in range(release + 1, min(end, release + round(fps * 0.8)) + 1):
        r = by_index.get(index, {})
        if not visible(r, names):
            break
        xy = r["xy"]
        angle = angle_deg(*(np.asarray(xy[n]) for n in names))
        high = angle is not None and angle >= 145 and xy[side + "w"][1] < xy[side + "e"][1]
        support.append(index)
        if high:
            count += 1
        else:
            # A single noisy landmark must not diagnose an early hand drop.
            following = [by_index.get(index + i, {}) for i in range(3)]
            low = []
            for f in following:
                if not visible(f, names):
                    continue
                points = f["xy"]
                a = angle_deg(*(np.asarray(points[n]) for n in names))
                low.append(
                    a is not None and (a < 140 or points[side + "w"][1] >= points[side + "e"][1])
                )
            ended = len(low) == 3 and all(low)
            break
    if count >= 3 and (ended or count / fps >= 0.5):
        result.update(
            hold_seconds=count / fps,
            hold_ended=ended,
            arm_side=side,
            hold_frames=[release + 1, release + count],
        )
    return result


def propose(candidate, features, context=None):
    base = {
        "available": False,
        "text": "לא התקבל זיהוי תנועה רציף מספיק להערה ספציפית. מה המאמן רואה?",
        "method": METHOD,
        "source_models": ["MediaPipe Pose Heavy", "RF-DETR Nano"],
        "requires_coach_review": True,
        "kind": "abstention",
        "evidence": [],
        "features": features,
        "limitations": LIMITATION,
    }
    if features.get("coverage", 0) < 0.7:
        return base
    evidence = [
        f"פריימי מקור {features['frames'][0]}–{features['frames'][1]}; כיסוי גוף {features['coverage']:.0%}."
    ]

    def result(focus, text, extra):
        return {
            **base,
            "available": True,
            "kind": "measured_observation",
            "focus": focus,
            "text": text,
            "evidence": evidence + extra,
        }

    context = context or {}
    knee = features.get("knee_min")
    knee_ref = context.get("knee_min")
    if knee is not None and knee_ref is not None and knee - knee_ref >= 12:
        return result(
            "knee_consistency",
            "בזריקה הזאת נראתה כפיפה קטנה יותר של הברך לעומת רוב הזריקות שסומנו באותו סט. בדקו את רצף העלייה לפני השחרור.",
            [
                f"זווית ברך מינימלית נצפית {knee:.0f}° לעומת חציון {knee_ref:.0f}° באותו צד גוף ובאותו סרטון."
            ],
        )
    trunk = features.get("trunk_range")
    trunk_ref = context.get("trunk_range")
    if trunk is not None and trunk_ref is not None and trunk - trunk_ref >= 8:
        return result(
            "trunk_consistency",
            "בזריקה הזאת תנועת הגו הייתה גדולה יותר מרוב הזריקות שסומנו בסט. בדקו אם הסיום נשאר בשליטה.",
            [f"טווח נטיית גו {trunk:.1f}° לעומת חציון {trunk_ref:.1f}° באותו סרטון."],
        )
    hold = features.get("hold_seconds")
    if hold is not None and features.get("hold_ended") and hold < 0.3:
        return result(
            "follow_through",
            f"היד הנראית נשארה ישרה וגבוהה כ־{hold:.2f} שניות אחרי השחרור המשוער ואז ירדה. בדקו את המשך התנועה.",
            [
                f"רצף יד גלויה בפריימים {features['hold_frames']}; ירידה נתמכה בשלושה פריימים רצופים."
            ],
        )
    if trunk is not None and trunk >= 16:
        return result(
            "trunk_motion",
            f"סביב תנועת הזריקה נמדד שינוי של כ־{trunk:.0f}° בנטיית הגו בתמונה. בדקו את יציבות הגוף בזמן העלייה והסיום.",
            ["אין מכך מדידת סיבוב גוף תלת־ממדי או הוכחת טעות טכנית."],
        )
    if hold is not None:
        qualifier = "" if features.get("hold_ended") else "לפחות "
        return result(
            "follow_through_observation",
            f"אחרי השחרור המשוער היד הנראית נשארה ישרה וגבוהה {qualifier}כ־{hold:.2f} שניות. בדקו אם המשך התנועה עקבי גם בזריקות הבאות.",
            [f"פריימי המשך תנועה: {features['hold_frames']}."],
        )
    knee_range = features.get("knee_range")
    if knee is not None and knee_range is not None and knee_range >= 15:
        phase = (
            "סביב השחרור המשוער" if candidate.get("release_frame") is not None else "בקטע המסומן"
        )
        return result(
            "knee_motion",
            f"{phase} נמדד שינוי של כ־{knee_range:.0f}° בזווית הברך בתמונה. בדקו בתנועה האיטית את התיאום עם עליית היד.",
            [
                f"צד גוף קבוע: {features.get('knee_side')}; זווית מזערית נצפית {knee:.0f}°.",
                "טווח זווית אינו מדידת כוח רגליים.",
            ],
        )
    if trunk is not None:
        return result(
            "trunk_observation",
            f"בקטע המסומן נמדד שינוי של כ־{trunk:.0f}° בנטיית הגו בתמונה. המאמן צריך לבדוק אם התנועה מתאימה לזריקה.",
            ["תיאור תנועה בלבד; אין קביעה שהטכניקה תקינה או שגויה."],
        )
    return base


def generate(candidates, records, fps, camera_stable=True):
    side = max(
        ("l", "r"), key=lambda s: sum(visible(r, [s + "h", s + "k", s + "a"]) for r in records)
    )
    features = [measure(c, records, fps, side) for c in candidates]
    context = {}
    for key in ("knee_min", "trunk_range"):
        values = [
            f[key]
            for c, f in zip(candidates, features)
            if c.get("release_frame") is not None and f.get("coverage", 0) >= 0.8 and key in f
        ]
        if camera_stable and len(values) >= 5:
            context[key] = median(values)
    return {c["shot_id"]: propose(c, f, context) for c, f in zip(candidates, features)}
