from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

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
from swisher.rim import GroundingDinoRimDetector, RimDetection

ProgressCallback = Callable[[float], None]
MIN_POSE_CONFIDENCE = 0.55
DETECTION_STRIDE = 3
RIM_SEARCH_STRIDE = 30


def analyze_video(
    input_path: Path,
    output_path: Path,
    on_progress: ProgressCallback | None = None,
) -> AnalysisResult:
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

    writer = _video_writer(output_path, fps, width, height)
    pose_module = mp.solutions.pose
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

    with pose_module.Pose(
        static_image_mode=True,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rim_detection is None and index % RIM_SEARCH_STRIDE == 0 and rim_attempts < 3:
                rim_detection = rim_detector.detect(rgb)
                rim_attempts += 1
            if index % DETECTION_STRIDE == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness_samples.append(float(np.mean(gray)))
                blur_samples.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                latest_detections = detector.detect(rgb, threshold=0.18)
                sampled_frames += 1
                primary = select_primary_player(latest_detections, frame.shape, tracked_box)
                if primary:
                    player_detected_frames += 1
                    tracked_box = smooth_box(tracked_box, primary.xyxy)
                if any(item.class_name == "sports_ball" for item in latest_detections):
                    ball_detected_frames += 1
                    best_ball = max(
                        (item for item in latest_detections if item.class_name == "sports_ball"),
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
            x1, y1, x2, y2 = crop_box
            crop = rgb[y1:y2, x1:x2]
            prediction = pose.process(crop) if crop.size else None
            landmarks = prediction.pose_landmarks if prediction is not None else None
            record = _extract_record(landmarks, crop_box, index)
            frames.append(record)

            if landmarks:
                _draw_pose(frame, record, pose_module.POSE_CONNECTIONS)
            _draw_detections(frame, latest_detections, tracked_box)
            _draw_rim(frame, rim_detection)
            _draw_status(frame, record, index / fps)
            writer.write(frame)
            index += 1
            if on_progress and index % 5 == 0:
                on_progress(index / frame_count)

    capture.release()
    writer.release()
    if not frames:
        raise ValueError("לא נמצאו פריימים קריאים")

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
    coaching = _coaching_feedback(status, metrics, release, quality_warnings)
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
            "make/miss נקבע רק כאשר מסלול הכדור חוצה בבירור את מישור הטבעת",
            "release מבוסס על הפרדת כדור-יד ודורש אימות מול release frames מתויגים",
            "זוויות מחושבות בהיטל דו־ממדי ותלויות בזווית המצלמה",
            "המערכת אינה מסיקה עדיין מה גרם לקליעה או להחטאה",
        ],
    )


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
        a = tuple(np.round(xy[index_to_name[start]]).astype(int))
        b = tuple(np.round(xy[index_to_name[end]]).astype(int))
        cv2.line(frame, a, b, (255, 180, 30), 2, cv2.LINE_AA)
    for point in xy.values():
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
            confidence = min(
                0.92,
                0.45
                + 0.25 * rim.confidence
                + 0.15 * float(previous["confidence"])
                + 0.15 * float(current["confidence"]),
            )
            return ShotOutcome(
                status="made",
                confidence=confidence,
                reason="מסלול הכדור חצה מטה את מישור הטבעת בתוך רוחב הסל",
            )
        near_rim = x1 - rim.width * 1.5 <= horizontal_center <= x2 + rim.width * 1.5
        clearly_outside = horizontal_center < x1 - margin or horizontal_center > x2 + margin
        if crosses_downward and near_rim and clearly_outside:
            confidence = min(
                0.88,
                0.42
                + 0.2 * rim.confidence
                + 0.13 * float(previous["confidence"])
                + 0.13 * float(current["confidence"]),
            )
            return ShotOutcome(
                status="missed",
                confidence=confidence,
                reason="הכדור חצה מטה את גובה הטבעת מחוץ לרוחב הסל",
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
        wrist_y = np.convolve(wrist_y, np.ones(5) / 5, mode="same")
    velocity = np.abs(np.gradient(wrist_y, indices))
    peak_pos = int(np.argmax(velocity))
    radius = max(int(fps * 0.75), 4)
    start_pos = max(0, peak_pos - radius)
    end_pos = min(len(indices) - 1, peak_pos + radius)
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
    if leans:
        robust_range = _robust_range(leans) or 0.0
        consistency = float(np.clip(1.0 - robust_range / 35.0, 0.0, 1.0))
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
        shoulder_width = max(float(np.linalg.norm(xy["ls"] - xy["rs"])), 1.0)
        ball_center = ball["center"]  # type: ignore[assignment]
        left_distance = float(np.linalg.norm(ball_center - xy["lw"])) / shoulder_width
        right_distance = float(np.linalg.norm(ball_center - xy["rw"])) / shoulder_width
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
        upward = float(after["ball_y"]) <= float(before["ball_y"]) + 12
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
    side = release.shooting_side
    shoulder, elbow, wrist, finger = (
        ("ls", "le", "lw", "li") if side == "left" else ("rs", "re", "rw", "ri")
    )
    hip, knee, ankle = ("lh", "lk", "la") if side == "left" else ("rh", "rk", "ra")
    elbow_angle = angle_deg(xy[shoulder], xy[elbow], xy[wrist])
    knee_angle = angle_deg(xy[hip], xy[knee], xy[ankle])
    wrist_angle = angle_deg(xy[elbow], xy[wrist], xy[finger])
    ankle_y = max(float(xy["la"][1]), float(xy["ra"][1]))
    shoulder_mid = (xy["ls"] + xy["rs"]) / 2
    hip_mid = (xy["lh"] + xy["rh"]) / 2
    body_height = max(
        ankle_y - float(shoulder_mid[1]) + float(np.linalg.norm(shoulder_mid - hip_mid)), 1.0
    )
    release_height = (ankle_y - float(xy[wrist][1])) / body_height

    event_poses = valid
    if event_window:
        event_poses = [
            pose
            for pose in valid
            if event_window.start_frame <= int(pose["index"]) <= event_window.end_frame
        ]
    hip_y = [
        float((pose["xy"]["lh"][1] + pose["xy"]["rh"][1]) / 2)  # type: ignore[index]
        for pose in event_poses
    ]
    jump_height = (float(np.percentile(hip_y, 80)) - min(hip_y)) / body_height if hip_y else None

    start_pose = min(
        valid, key=lambda item: abs(int(item["index"]) - max(0, release.frame - int(fps * 0.5)))
    )
    landing_pose = min(
        valid, key=lambda item: abs(int(item["index"]) - (release.frame + int(fps * 0.7)))
    )
    start_hip = (start_pose["xy"]["lh"] + start_pose["xy"]["rh"]) / 2  # type: ignore[index]
    landing_hip = (landing_pose["xy"]["lh"] + landing_pose["xy"]["rh"]) / 2  # type: ignore[index]
    shoulder_width = max(float(np.linalg.norm(xy["ls"] - xy["rs"])), 1.0)
    landing_drift = float(np.linalg.norm(landing_hip - start_hip)) / shoulder_width
    hold_frames = 0
    for pose in sorted(valid, key=lambda item: int(item["index"])):
        frame_index = int(pose["index"])
        if frame_index < release.frame or frame_index > release.frame + int(fps * 1.2):
            continue
        points = pose["xy"]  # type: ignore[assignment]
        arm_angle = angle_deg(points[shoulder], points[elbow], points[wrist])
        if arm_angle is not None and arm_angle >= 145 and points[wrist][1] < points[elbow][1]:
            hold_frames += 1
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
    if status != "ok":
        return CoachingFeedback(
            available=False,
            focus="איכות הצילום",
            observation="אין מספיק פריימים אמינים כדי לתת עצת קליעה.",
            cue="מקם את הטלפון כך שכל הגוף, הכדור והסל יישארו בפריים.",
            drill="צלם סט חדש של 10 זריקות ממצלמה יציבה וב־60 FPS.",
            confidence=0.98,
            evidence=quality_warnings or ["כיסוי Pose נמוך מסף האמינות"],
        )
    if metrics.pose_consistency is not None and metrics.pose_consistency < 0.55:
        return CoachingFeedback(
            available=True,
            focus="יציבות הגוף",
            observation="נטיית הגוף השתנתה בצורה משמעותית לאורך התנועה.",
            cue="סיים גבוה ונחת באותו קו שבו התחלת.",
            drill="10 זריקות form shooting עם עצירה של שתי שניות בנחיתה.",
            confidence=0.72,
            evidence=[f"מדד עקביות גוף: {metrics.pose_consistency:.0%}"],
        )
    if metrics.min_knee_angle_deg is not None and metrics.min_knee_angle_deg > 150:
        return CoachingFeedback(
            available=True,
            focus="שימוש ברגליים",
            observation="כיפוף הברך היה קטן יחסית לאורך חלון הזריקה.",
            cue="טען את הרגליים לפני שהכדור מתחיל לעלות.",
            drill="3 סטים של 8 זריקות קרובות עם dip רגוע ומתוזמן.",
            confidence=0.68,
            evidence=[f"זווית ברך מינימלית: {metrics.min_knee_angle_deg:.1f}°"],
        )
    if release is None:
        return CoachingFeedback(
            available=True,
            focus="עקביות השחרור",
            observation="הגוף זוהה, אך הפרדת הכדור מהיד לא הייתה ברורה מספיק.",
            cue="שמור את הכדור גלוי למצלמה וסיים עם יד גבוהה.",
            drill="10 זריקות קרובות מזווית צד־אלכסון.",
            confidence=0.55,
            evidence=["לא נמצא release שעבר את סף הביטחון"],
        )
    if (
        metrics.follow_through_hold_seconds is not None
        and metrics.follow_through_hold_seconds < 0.25
    ):
        return CoachingFeedback(
            available=True,
            focus="Follow-through",
            observation="היד ירדה מהר לאחר רגע השחרור המשוער.",
            cue="השאר את האצבעות בתוך הסל עד שהכדור מגיע לטבעת.",
            drill="10 זריקות עם החזקת follow-through לשתי שניות.",
            confidence=min(0.8, release.confidence),
            evidence=[f"החזקה לאחר שחרור: {metrics.follow_through_hold_seconds:.2f} שניות"],
        )
    return CoachingFeedback(
        available=True,
        focus="שימור המכניקה",
        observation="לא נמצא כרגע דפוס אחד שחורג באופן ברור מספי המנוע.",
        cue="שמור על אותו קצב ושחרור לאורך הסט הבא.",
        drill="סט של 10 זריקות מאותה נקודה ומדידת עקביות.",
        confidence=0.6,
        evidence=[f"release confidence: {release.confidence:.0%}"],
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
