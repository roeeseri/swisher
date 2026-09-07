from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from contextlib import contextmanager
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from swisher import pose_backend
from swisher.contracts import (
    AnalysisResult,
    CoachingFeedback,
    DetectionSummary,
    EventWindow,
    QualitySummary,
    ReleaseEvent,
    ShotMetrics,
    ShotOutcome,
    VideoMetadata,
)
from swisher.detection import RfDetrDetector, padded_crop, select_primary_player, smooth_box
from swisher.geometry import angle_deg, trunk_lean_deg
from swisher.proposals import propose_shots
from swisher.rim import GroundingDinoRimDetector, RimDetection

ProgressCallback = Callable[[float], None]
MIN_POSE_CONFIDENCE = 0.55
DETECTION_STRIDE = 3
RIM_SEARCH_STRIDE = 30


def analyze_video(
    input_path: Path,
    output_path: Path,
    on_progress: ProgressCallback | None = None,
    *,
    player_region: tuple[int, int, int, int] | None = None,
    thorough: bool = False,
    search_rim: bool = False,
    trace_path: Path | None = None,
) -> AnalysisResult:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("אסור לדרוס את סרטון המקור")
    started = time.monotonic()
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("לא ניתן לקרוא את קובץ הווידאו")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width < 320 or height < 240 or fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError("מטא־דאטה של הסרטון אינו תקין או שהרזולוציה נמוכה מדי")
    if frame_count > int(fps * 120):
        capture.release()
        raise ValueError("בגרסת האב־טיפוס ניתן לנתח סרטונים באורך של עד שתי דקות")

    metadata = VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=frame_count / fps,
    )

    if player_region is not None:
        x1, y1, x2, y2 = player_region
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            capture.release()
            raise ValueError("אזור השחקן חורג מגבולות התמונה")
    try:
        writer = _video_writer(output_path, fps, width, height)
    except Exception:
        capture.release()
        raise
    pose_module = pose_backend
    frames: list[dict[str, object]] = []
    detector = RfDetrDetector()
    rim_detector = GroundingDinoRimDetector()
    tracked_box: tuple[float, float, float, float] | None = None
    latest_detections = []
    sampled_frames = 0
    player_detected_frames = 0
    ball_detected_frames = 0
    ball_observations: list[dict[str, object]] = []
    brightness_samples: list[float] = []
    blur_samples: list[float] = []
    rim_detection: RimDetection | None = None
    rim_attempts = 0
    stride = 1 if thorough else DETECTION_STRIDE
    eligible = False
    ambiguous_frames = 0

    with (
        _release_resources(capture, writer),
        pose_module.Pose(
            static_image_mode=True,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose,
    ):
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (
                search_rim
                and rim_detection is None
                and index % RIM_SEARCH_STRIDE == 0
                and rim_attempts < 3
            ):
                rim_detection = rim_detector.detect(rgb)
                rim_attempts += 1
            if index % stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness_samples.append(float(np.mean(gray)))
                blur_samples.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                latest_detections = detector.detect(rgb, threshold=0.18)
                sampled_frames += 1
                people = [d for d in latest_detections if d.class_name == "person"]
                if player_region:
                    rx1, ry1, rx2, ry2 = player_region
                    people = [
                        d for d in people if rx1 <= d.center[0] <= rx2 and ry1 <= d.center[1] <= ry2
                    ]
                eligible = len(people) == 1
                ambiguous_frames += int(len(people) > 1)
                primary = (
                    select_primary_player(people, frame.shape, tracked_box) if eligible else None
                )
                if primary:
                    player_detected_frames += 1
                    tracked_box = smooth_box(tracked_box, primary.xyxy)
                else:
                    tracked_box = None
                balls = [d for d in latest_detections if d.class_name == "sports_ball"]
                if eligible and len(balls) == 1:
                    ball_detected_frames += 1
                    best_ball = max(
                        balls,
                        key=lambda item: item.confidence,
                    )
                    ball_observations.append(
                        {
                            "frame": index,
                            "center": np.array(best_ball.center, dtype=np.float64),
                            "confidence": best_ball.confidence,
                        }
                    )

            crop_box = (0, 0, width, height)
            if tracked_box is not None:
                crop_box = padded_crop(tracked_box, width, height)
            if player_region is not None:
                crop_box = player_region
            x1, y1, x2, y2 = crop_box
            crop = rgb[y1:y2, x1:x2]
            prediction = pose.process(crop) if crop.size and eligible else None
            landmarks = prediction.pose_landmarks if prediction is not None else None
            record = _extract_record(landmarks, crop_box, index)
            record["timestamp_ms"] = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            frames.append(record)

            if landmarks:
                _draw_pose(frame, record, pose_module.POSE_CONNECTIONS)
            _draw_detections(frame, latest_detections, tracked_box)
            _draw_rim(frame, rim_detection)
            _draw_status(frame, record, index / fps)
            writer.write(frame)
            index += 1
            if on_progress and index % 5 == 0:
                on_progress(min(index / frame_count, 1.0))

    capture.release()
    writer.release()
    if not frames:
        raise ValueError("לא נמצאו פריימים קריאים")
    if len(frames) < frame_count:
        raise ValueError("הסרטון הסתיים לפני מספר הפריימים הצפוי; יש לבדוק את קובץ המקור")
    if trace_path:
        with trace_path.open("w", encoding="utf-8") as handle:
            for record in frames:
                handle.write(json.dumps(record, default=lambda v: v.tolist()) + "\n")

    valid = [frame for frame in frames if bool(frame["valid"])]
    pose_confidence = len(valid) / len(frames)
    event_window = _event_window(valid, fps)
    release = _release_event(valid, ball_observations, fps)
    metrics = _metrics(valid, event_window, release, fps)
    player_detection_rate = player_detected_frames / max(sampled_frames, 1)
    ball_detection_rate = ball_detected_frames / max(sampled_frames, 1)
    mean_brightness = float(np.mean(brightness_samples)) if brightness_samples else 0.0
    mean_blur_score = float(np.mean(blur_samples)) if blur_samples else 0.0
    quality_warnings = _quality_warnings(metadata, mean_brightness, mean_blur_score)
    if ambiguous_frames:
        quality_warnings.append("נראו כמה שחקנים באזור הניתוח; הפריימים האלה לא נותחו")
    status = (
        "ok"
        if pose_confidence >= MIN_POSE_CONFIDENCE and player_detection_rate >= 0.8
        else "abstained"
    )
    message = (
        "הניתוח הושלם. המדדים מבוססים על פריימים שבהם הגוף זוהה בביטחון מספק."
        if status == "ok"
        else "המודל נמנע מלתת מסקנה: הגוף לא נראה בביטחון מספק ברוב הסרטון."
    )
    outcome = _shot_outcome(ball_observations, rim_detection)
    candidates = propose_shots(frames, ball_observations, fps, len(frames), _release_event)
    coaching = _coaching_feedback(
        status if len(candidates) == 1 else "abstained", metrics, release, quality_warnings
    )
    return AnalysisResult(
        status=status,
        message=message,
        video=metadata,
        pose_confidence=pose_confidence,
        valid_pose_frames=len(valid),
        event_window=event_window,
        release=release,
        outcome=outcome,
        metrics=metrics,
        detections=DetectionSummary(
            detector=detector.name,
            sampled_frames=sampled_frames,
            player_detected_frames=player_detected_frames,
            ball_detected_frames=ball_detected_frames,
            player_detection_rate=player_detection_rate,
            ball_detection_rate=ball_detection_rate,
            rim_detected=rim_detection is not None,
            rim_confidence=rim_detection.confidence if rim_detection else 0.0,
            rim_box=rim_detection.xyxy if rim_detection else None,
        ),
        quality=QualitySummary(
            mean_brightness=mean_brightness,
            mean_blur_score=mean_blur_score,
            warnings=quality_warnings,
        ),
        coaching=coaching,
        limitations=[
            "זיהוי הכדור הוא COCO baseline ועדיין לא fine-tuned לכדורסל מהיר ומטושטש",
            "זיהוי הסל הוא zero-shot ודורש fine-tuning על Golden Dataset",
            "חצייה בתמונה אינה הוכחת קליעה; התוצאה נשארת לא ידועה עד אישור מאמן",
            "release מבוסס על הפרדת כדור-יד ודורש אימות מול release frames מתויגים",
            "זוויות מחושבות בהיטל דו־ממדי ותלויות בזווית המצלמה",
            "המערכת אינה מסיקה עדיין מה גרם לקליעה או להחטאה",
        ],
        candidates=candidates,
        provenance={
            "pipeline": "swisher_0.3",
            "pose_model": "mediapipe_blazepose_heavy",
            "pose_api": "Tasks PoseLandmarker IMAGE",
            "pose_asset_sha256": _file_hash(pose.path),
            "mediapipe_version": version("mediapipe"),
            "rfdetr_version": version("rfdetr"),
            "detector": detector.name,
            "detection_stride": stride,
            "player_region": player_region,
            "processed_frames": len(frames),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "source_sha256": _file_hash(input_path),
            "pose_confidence_semantics": "valid frame coverage, not calibrated probability",
            "release_confidence_semantics": "heuristic score, not calibrated probability",
            "timing": "frame/fps estimates; decoder timestamps saved in trace",
        },
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _release_resources(capture, writer):
    try:
        yield
    finally:
        capture.release()
        writer.release()


def _video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("לא ניתן ליצור קובץ וידאו מסומן")


def _extract_record(
    landmarks: object | None, crop_box: tuple[int, int, int, int], index: int
) -> dict[str, object]:
    record: dict[str, object] = {"index": index, "valid": False}
    if landmarks is None:
        return record
    points = landmarks.landmark  # type: ignore[attr-defined]
    x1, y1, x2, y2 = crop_box
    width, height = x2 - x1, y2 - y1
    required = {
        "ls": 11,
        "rs": 12,
        "le": 13,
        "re": 14,
        "lw": 15,
        "rw": 16,
        "li": 19,
        "ri": 20,
        "lh": 23,
        "rh": 24,
        "lk": 25,
        "rk": 26,
        "la": 27,
        "ra": 28,
    }
    visibility_by_name = {name: float(points[i].visibility) for name, i in required.items()}
    core_visible = min(visibility_by_name[name] for name in ("ls", "rs", "lh", "rh")) >= 0.4
    arm_visible = any(
        min(visibility_by_name[name] for name in arm) >= 0.4
        for arm in (("ls", "le", "lw"), ("rs", "re", "rw"))
    )
    leg_visible = any(
        min(visibility_by_name[name] for name in leg) >= 0.35
        for leg in (("lh", "lk", "la"), ("rh", "rk", "ra"))
    )
    if not (core_visible and arm_visible and leg_visible):
        return record
    xy = {
        name: np.array([x1 + points[i].x * width, y1 + points[i].y * height], dtype=np.float64)
        for name, i in required.items()
    }
    record.update(
        {
            "valid": True,
            "visibility": float(np.mean(list(visibility_by_name.values()))),
            "visibility_by_name": visibility_by_name,
            "xy": xy,
        }
    )
    return record


def _draw_pose(frame: np.ndarray, record: dict[str, object], connections: object) -> None:
    if not record["valid"]:
        return
    xy = record["xy"]  # type: ignore[assignment]
    visibility = record["visibility_by_name"]
    index_to_name = {
        11: "ls",
        12: "rs",
        13: "le",
        14: "re",
        15: "lw",
        16: "rw",
        23: "lh",
        24: "rh",
        25: "lk",
        26: "rk",
        27: "la",
        28: "ra",
    }
    for start, end in connections:  # type: ignore[union-attr]
        if start not in index_to_name or end not in index_to_name:
            continue
        if min(visibility[index_to_name[start]], visibility[index_to_name[end]]) < 0.5:
            continue
        a = tuple(np.round(xy[index_to_name[start]]).astype(int))
        b = tuple(np.round(xy[index_to_name[end]]).astype(int))
        cv2.line(frame, a, b, (255, 180, 30), 2, cv2.LINE_AA)
    for name, point in xy.items():
        if visibility[name] < 0.5:
            continue
        cv2.circle(frame, tuple(np.round(point).astype(int)), 4, (0, 220, 255), -1)


def _draw_detections(
    frame: np.ndarray,
    detections: list[object],
    tracked_box: tuple[float, float, float, float] | None,
) -> None:
    for detection in detections:
        if detection.class_name != "sports_ball":  # type: ignore[attr-defined]
            continue
        x1, y1, x2, y2 = (int(value) for value in detection.xyxy)  # type: ignore[attr-defined]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 210, 50), 2)
        cv2.putText(
            frame, "ball", (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 210, 50), 2
        )
    if tracked_box:
        x1, y1, x2, y2 = (int(value) for value in tracked_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 30), 2)
        cv2.putText(
            frame, "player", (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 30), 2
        )


def _draw_rim(frame: np.ndarray, rim: RimDetection | None) -> None:
    if rim is None:
        return
    x1, y1, x2, y2 = (int(value) for value in rim.xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 70, 220), 2)
    cv2.putText(
        frame,
        f"rim {rim.confidence:.2f}",
        (x1, max(16, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 70, 220),
        2,
    )


def _shot_outcome(
    ball_observations: list[dict[str, object]], rim: RimDetection | None
) -> ShotOutcome:
    if rim is None:
        return ShotOutcome(
            status="unknown",
            confidence=0.0,
            reason="הסל לא זוהה בביטחון מספיק",
        )
    if len(ball_observations) < 4:
        return ShotOutcome(
            status="unknown",
            confidence=0.0,
            reason="אין מספיק נקודות במסלול הכדור ליד הסל",
        )
    observations = sorted(ball_observations, key=lambda item: int(item["frame"]))
    x1, y1, x2, y2 = rim.xyxy
    rim_y = (y1 + y2) / 2
    margin = max(rim.width * 0.2, 3.0)
    for previous, current in pairwise(observations):
        previous_center = previous["center"]  # type: ignore[assignment]
        current_center = current["center"]  # type: ignore[assignment]
        crosses_downward = previous_center[1] < rim_y <= current_center[1]
        horizontal_center = (previous_center[0] + current_center[0]) / 2
        inside_rim = x1 - margin <= horizontal_center <= x2 + margin
        if crosses_downward and inside_rim:
            return ShotOutcome(
                status="unknown",
                confidence=0.0,
                reason="חצייה בתוך רוחב הסל בתמונה — דרוש אישור מאמן; עומק אינו נמדד",
            )
        near_rim = x1 - rim.width * 1.5 <= horizontal_center <= x2 + rim.width * 1.5
        clearly_outside = horizontal_center < x1 - margin or horizontal_center > x2 + margin
        if crosses_downward and near_rim and clearly_outside:
            return ShotOutcome(
                status="unknown",
                confidence=0.0,
                reason="חצייה מחוץ לרוחב הסל בתמונה — דרוש אישור מאמן",
            )
    return ShotOutcome(
        status="unknown",
        confidence=0.25,
        reason="הסל זוהה, אך מסלול הכדור אינו שלם מספיק להכרעת make/miss",
    )


def _event_window(valid: list[dict[str, object]], fps: float) -> EventWindow | None:
    if len(valid) < max(8, int(fps * 0.25)):
        return None
    indices = np.array([int(frame["index"]) for frame in valid])
    wrist_y = np.array(
        [
            min(float(frame["xy"]["lw"][1]), float(frame["xy"]["rw"][1]))  # type: ignore[index]
            for frame in valid
        ]
    )
    if len(wrist_y) >= 5:
        wrist_y = np.convolve(np.pad(wrist_y, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")
    velocity = np.abs(np.gradient(wrist_y, indices))
    peak_pos = int(np.argmax(velocity))
    radius = max(int(fps * 0.75), 4)
    start_pos = int(np.searchsorted(indices, indices[peak_pos] - radius))
    end_pos = min(len(indices) - 1, int(np.searchsorted(indices, indices[peak_pos] + radius)))
    start_frame, end_frame = int(indices[start_pos]), int(indices[end_pos])
    baseline = float(np.median(velocity)) + 1e-6
    confidence = min(float(velocity[peak_pos] / (baseline * 5)), 1.0)
    return EventWindow(
        start_frame=start_frame,
        end_frame=end_frame,
        start_seconds=start_frame / fps,
        end_seconds=end_frame / fps,
        confidence=confidence,
    )


def _metrics(
    valid: list[dict[str, object]],
    event_window: EventWindow | None,
    release: ReleaseEvent | None,
    fps: float,
) -> ShotMetrics:
    selected = valid
    if event_window:
        selected = [
            frame
            for frame in valid
            if event_window.start_frame <= int(frame["index"]) <= event_window.end_frame
        ]
    elbows: list[float] = []
    knees: list[float] = []
    leans: list[float] = []
    for frame in selected:
        xy = frame["xy"]  # type: ignore[assignment]
        visibility = frame["visibility_by_name"]  # type: ignore[assignment]
        for a, b, c in (("ls", "le", "lw"), ("rs", "re", "rw")):
            value = (
                angle_deg(xy[a], xy[b], xy[c])
                if min(visibility[a], visibility[b], visibility[c]) >= 0.5
                else None
            )
            if value is not None and 10 <= value <= 180:
                elbows.append(value)
        for a, b, c in (("lh", "lk", "la"), ("rh", "rk", "ra")):
            value = (
                angle_deg(xy[a], xy[b], xy[c])
                if min(visibility[a], visibility[b], visibility[c]) >= 0.45
                else None
            )
            if value is not None and 20 <= value <= 180:
                knees.append(value)
        shoulder_mid = (xy["ls"] + xy["rs"]) / 2
        hip_mid = (xy["lh"] + xy["rh"]) / 2
        leans.append(trunk_lean_deg(shoulder_mid, hip_mid))
    consistency = None
    # No unvalidated 0-100 score derived from arbitrary angular thresholds.
    (
        release_elbow,
        release_knee,
        release_wrist,
        release_height,
        jump_height,
        landing_drift,
        follow_through,
    ) = _release_metrics(valid, release, event_window, fps)
    return ShotMetrics(
        min_elbow_angle_deg=_percentile(elbows, 5),
        max_elbow_angle_deg=_percentile(elbows, 95),
        min_knee_angle_deg=_percentile(knees, 5),
        max_knee_angle_deg=_percentile(knees, 95),
        trunk_lean_range_deg=_robust_range(leans),
        pose_consistency=consistency,
        release_elbow_angle_deg=release_elbow,
        release_knee_angle_deg=release_knee,
        release_wrist_angle_deg=release_wrist,
        release_height_body_ratio=release_height,
        jump_height_body_ratio=jump_height,
        landing_drift_shoulder_widths=landing_drift,
        follow_through_hold_seconds=follow_through,
    )


def _release_event(
    valid: list[dict[str, object]], ball_observations: list[dict[str, object]], fps: float
) -> ReleaseEvent | None:
    if len(ball_observations) < 3 or len(valid) < 3:
        return None
    pose_by_frame = {int(item["index"]): item for item in valid}
    candidates: list[dict[str, object]] = []
    for ball in ball_observations:
        ball_frame = int(ball["frame"])
        nearby_frames = [
            frame for frame in pose_by_frame if abs(frame - ball_frame) <= DETECTION_STRIDE
        ]
        if not nearby_frames:
            continue
        pose_frame = min(nearby_frames, key=lambda frame: abs(frame - ball_frame))
        pose = pose_by_frame[pose_frame]
        xy = pose["xy"]  # type: ignore[assignment]
        visibility = pose.get("visibility_by_name", {})
        if min(visibility.get(k, 0) for k in ("ls", "rs", "lw", "rw")) < 0.5:
            continue
        shoulder_width = float(np.linalg.norm(xy["ls"] - xy["rs"]))
        if shoulder_width < 12:
            continue
        ball_center = ball["center"]  # type: ignore[assignment]
        left_distance = float(np.linalg.norm(ball_center - xy["lw"])) / shoulder_width
        right_distance = float(np.linalg.norm(ball_center - xy["rw"])) / shoulder_width
        side = "left" if left_distance < right_distance else "right"
        wrist, shoulder = ("lw", "ls") if side == "left" else ("rw", "rs")
        if xy[wrist][1] > xy[shoulder][1]:
            continue  # do not label a dribble or low pass as a release
        candidates.append(
            {
                "frame": ball_frame,
                "distance": min(left_distance, right_distance),
                "side": "left" if left_distance < right_distance else "right",
                "ball_y": float(ball_center[1]),
                "confidence": float(ball["confidence"]),
            }
        )
    if len(candidates) < 3:
        return None
    candidates.sort(key=lambda item: int(item["frame"]))
    distances = np.array([float(item["distance"]) for item in candidates])
    separation = np.diff(distances)
    for index in np.argsort(separation)[::-1]:
        next_index = int(index) + 1
        before = candidates[int(index)]
        after = candidates[next_index]
        if not (0 < int(after["frame"]) - int(before["frame"]) <= max(3, int(fps * 0.12))):
            continue
        if before["side"] != after["side"]:
            continue
        upward = float(after["ball_y"]) < float(before["ball_y"])
        if float(before["distance"]) <= 1.4 and separation[int(index)] >= 0.18 and upward:
            confidence = float(
                np.clip(
                    0.35 + separation[int(index)] * 0.35 + float(after["confidence"]) * 0.35,
                    0.0,
                    0.95,
                )
            )
            return ReleaseEvent(
                frame=int(after["frame"]),
                seconds=int(after["frame"]) / fps,
                confidence=confidence,
                shooting_side=str(after["side"]),
                wrist_ball_distance_norm=float(after["distance"]),
                interval_start_frame=int(before["frame"]),
                interval_end_frame=int(after["frame"]),
            )
    return None


def _release_metrics(
    valid: list[dict[str, object]],
    release: ReleaseEvent | None,
    event_window: EventWindow | None,
    fps: float,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    if release is None or not valid:
        return (None,) * 7
    release_pose = min(valid, key=lambda item: abs(int(item["index"]) - release.frame))
    if abs(int(release_pose["index"]) - release.frame) > DETECTION_STRIDE:
        return (None,) * 7
    xy = release_pose["xy"]  # type: ignore[assignment]
    visibility = release_pose["visibility_by_name"]
    side = release.shooting_side
    shoulder, elbow, wrist, _finger = (
        ("ls", "le", "lw", "li") if side == "left" else ("rs", "re", "rw", "ri")
    )
    hip, knee, ankle = ("lh", "lk", "la") if side == "left" else ("rh", "rk", "ra")
    elbow_angle = (
        angle_deg(xy[shoulder], xy[elbow], xy[wrist])
        if min(visibility[k] for k in (shoulder, elbow, wrist)) >= 0.6
        else None
    )
    knee_angle = (
        angle_deg(xy[hip], xy[knee], xy[ankle])
        if min(visibility[k] for k in (hip, knee, ankle)) >= 0.6
        else None
    )
    wrist_angle = None  # hand landmarks are not validated for wrist-snap measurement
    hold_frames = 0
    if max(int(p["index"]) for p in valid) < release.frame + fps * 0.3:
        return elbow_angle, knee_angle, None, None, None, None, None
    last_frame = release.frame - 1
    for pose in sorted(valid, key=lambda item: int(item["index"])):
        frame_index = int(pose["index"])
        if frame_index < release.frame or frame_index > release.frame + int(fps * 1.2):
            continue
        points = pose["xy"]  # type: ignore[assignment]
        if frame_index - last_frame > 1:
            return elbow_angle, knee_angle, None, None, None, None, None
        last_frame = frame_index
        if min(pose["visibility_by_name"][k] for k in (shoulder, elbow, wrist)) < 0.6:
            return elbow_angle, knee_angle, None, None, None, None, None
        arm_angle = angle_deg(points[shoulder], points[elbow], points[wrist])
        if arm_angle is not None and arm_angle >= 145 and points[wrist][1] < points[elbow][1]:
            hold_frames += 1
        else:
            break
    # Hip displacement is not jump height; pose alone is not landing ground truth.
    release_height = jump_height = landing_drift = None
    return (
        elbow_angle,
        knee_angle,
        wrist_angle,
        release_height,
        jump_height,
        landing_drift,
        hold_frames / fps,
    )


def _coaching_feedback(
    status: str,
    metrics: ShotMetrics,
    release: ReleaseEvent | None,
    quality_warnings: list[str],
) -> CoachingFeedback:
    if status != "ok" or quality_warnings or release is None or release.confidence < 0.65:
        return CoachingFeedback(
            available=False,
            focus="איכות הצילום",
            observation="אין מספיק פריימים אמינים כדי לתת עצת קליעה.",
            cue="מקם את הטלפון כך שכל הגוף, הכדור והסל יישארו בפריים.",
            drill="צלם סט חדש של 10 זריקות ממצלמה יציבה וב־60 FPS.",
            confidence=0.0,
            evidence=quality_warnings or ["כיסוי Pose נמוך מסף האמינות"],
        )
    if metrics.trunk_lean_range_deg is not None and metrics.trunk_lean_range_deg > 18:
        return CoachingFeedback(
            available=True,
            focus="יציבות הגוף",
            observation="נמדד שינוי בנטיית הגו בהיטל התמונה. מאמן צריך לבדוק אם הוא רצוי.",
            cue="לבדיקת המאמן: האם השחקן מסיים את התנועה בשליטה?",
            drill="אם המאמן מאשר: 5 זריקות קרובות באותו קצב, ואז השוואת הסרטונים.",
            confidence=0.0,
            evidence=[f"טווח נטיית גו דו־ממדית: {metrics.trunk_lean_range_deg:.0f}°"],
        )
    if (
        metrics.follow_through_hold_seconds is not None
        and metrics.follow_through_hold_seconds < 0.25
    ):
        return CoachingFeedback(
            available=True,
            focus="Follow-through",
            observation="חלון היד הגבוהה אחרי השחרור המשוער היה קצר. יש לבדוק בקליפ.",
            cue="לבדיקת המאמן: נסו לסיים עם יד גבוהה ונינוחה, ללא נעילה בכוח.",
            drill="רק באישור המאמן: 5 זריקות קרובות עם אותו דגש ובדיקה חוזרת.",
            confidence=0.0,
            evidence=[f"החזקה לאחר שחרור: {metrics.follow_through_hold_seconds:.2f} שניות"],
        )
    return CoachingFeedback(
        available=False,
        focus="סקירת המאמן",
        observation="אין כרגע מספיק ראיות לדגש אישי אוטומטי. זה לא אישור שהטכניקה מושלמת.",
        cue="בחרו יחד עם המאמן דגש אחד לסט הבא.",
        drill="צלמו 5 זריקות נוספות מאותה נקודה, ללא שינוי בזווית הצילום.",
        confidence=0.0,
        evidence=[f"שחרור משוער בפריים {release.frame}; נדרש אימות"],
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _robust_range(values: list[float]) -> float | None:
    if not values:
        return None
    unwrapped = np.degrees(np.unwrap(np.radians(values)))
    return float(np.percentile(unwrapped, 95) - np.percentile(unwrapped, 5))


def _quality_warnings(
    metadata: VideoMetadata, mean_brightness: float, mean_blur_score: float
) -> list[str]:
    warnings: list[str] = []
    if metadata.width < 720 or metadata.height < 720:
        warnings.append("הרזולוציה נמוכה; מומלץ לצלם לפחות 1080p")
    if metadata.fps < 30:
        warnings.append("קצב הפריימים נמוך; מומלץ 60 FPS למדידת שחרור")
    if mean_brightness < 45:
        warnings.append("הסרטון חשוך מדי")
    elif mean_brightness > 215:
        warnings.append("הסרטון בהיר מדי")
    if mean_blur_score < 55:
        warnings.append("זוהה טשטוש משמעותי")
    return warnings


def _draw_status(frame: np.ndarray, record: dict[str, object], seconds: float) -> None:
    label = f"Swisher v0.1 | {seconds:.2f}s | pose: {'OK' if record['valid'] else 'LOW'}"
    cv2.rectangle(frame, (12, 12), (500, 48), (15, 15, 15), -1)
    cv2.putText(frame, label, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 255), 2)
