from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2


def clip(source: Path, target: Path, start: int, end: int, fps: float, visuals=None):
    """Browser-playable H.264 preview; inclusive source frame bounds retained."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial.mp4")
    count = 0
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("לא ניתן לפתוח את סרטון המקור")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    try:
        with av.open(str(temporary), "w") as outgoing:
            source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = min(960, source_width)
            width -= width % 2
            height = round(source_height * width / source_width / 2) * 2
            output = outgoing.add_stream("libx264", rate=Fraction(fps).limit_denominator(100000))
            output.width, output.height = width, height
            output.pix_fmt = "yuv420p"
            output.options = {"preset": "ultrafast", "crf": "24"}
            for _ in range(start, end + 1):
                ok, image = capture.read()
                if not ok:
                    break
                if visuals:
                    poses, balls = visuals
                    draw_tracking(image, poses.get(start + count), balls.get(start + count))
                image = cv2.resize(image, (width, height))
                frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                frame.pts = count
                frame.time_base = Fraction(1, 1) / Fraction(fps).limit_denominator(100000)
                for packet in output.encode(frame):
                    outgoing.mux(packet)
                count += 1
            for packet in output.encode():
                outgoing.mux(packet)
    finally:
        capture.release()
    if count != end - start + 1:
        temporary.unlink(missing_ok=True)
        raise ValueError("לא ניתן לקרוא את כל פריימי הזריקה")
    temporary.replace(target)


def frame(source: Path, target: Path, index: int, region=None, visuals=None):
    capture = cv2.VideoCapture(str(source))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if not ok:
            raise ValueError("לא ניתן לקרוא את הפריים")
        if visuals:
            poses, balls = visuals
            draw_tracking(image, poses.get(index), balls.get(index))
        if region:
            x1, y1, x2, y2 = map(int, region)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            if x2 > x1 and y2 > y1:
                image = image[y1:y2, x1:x2]
        width = min(960, image.shape[1])
        image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))
        if not cv2.imwrite(str(target), image):
            raise ValueError("שמירת הפריים נכשלה")
    finally:
        capture.release()


EDGES = [
    ("ls", "rs"),
    ("ls", "le"),
    ("le", "lw"),
    ("rs", "re"),
    ("re", "rw"),
    ("ls", "lh"),
    ("rs", "rh"),
    ("lh", "rh"),
    ("lh", "lk"),
    ("lk", "la"),
    ("rh", "rk"),
    ("rk", "ra"),
]


def draw_tracking(image, pose=None, balls=None):
    """Draw only visible joints and actual observations from this exact frame."""
    if pose and pose.get("valid"):
        xy, visibility = pose.get("xy", {}), pose.get("visibility_by_name", {})
        for a, b in EDGES:
            if a in xy and b in xy and min(visibility.get(a, 0), visibility.get(b, 0)) >= 0.5:
                pa, pb = tuple(round(v) for v in xy[a]), tuple(round(v) for v in xy[b])
                cv2.line(image, pa, pb, (255, 190, 40), 5, cv2.LINE_AA)
        for name, point in xy.items():
            if visibility.get(name, 0) >= 0.5:
                cv2.circle(
                    image, tuple(round(v) for v in point), 6, (20, 225, 255), -1, cv2.LINE_AA
                )
    for ball in balls or []:
        point = tuple(round(v) for v in ball["center"])
        cv2.circle(image, point, 14, (70, 240, 90), 3, cv2.LINE_AA)
        cv2.putText(
            image,
            "BALL",
            (point[0] + 17, point[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (70, 240, 90),
            2,
            cv2.LINE_AA,
        )
    return image
