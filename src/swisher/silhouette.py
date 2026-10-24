"""Segmentation vs skeleton extent: diagnostic comparison, never accuracy ground truth."""

import numpy as np


def compare(mask, landmarks):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isfinite(mask).all():
        return {"available": False, "reason": "invalid mask"}
    ys, xs = np.where(mask >= 0.5)
    if len(xs) < 20:
        return {"available": False, "reason": "insufficient foreground"}
    h, w = mask.shape
    points = [
        (p.x, p.y)
        for p in landmarks
        if p.visibility >= 0.6 and np.isfinite([p.x, p.y]).all() and 0 <= p.x <= 1 and 0 <= p.y <= 1
    ]
    if len(points) < 8:
        return {"available": False, "reason": "insufficient visible landmarks"}
    px, py = np.array(points).T
    box = [float(px.min()), float(py.min()), float(px.max()), float(py.max())]
    envelope = [
        float(xs.min() / w),
        float(ys.min() / h),
        float((xs.max() + 1) / w),
        float((ys.max() + 1) / h),
    ]
    inside = sum(
        bool(mask[min(h - 1, int(y * h)), min(w - 1, int(x * w))] >= 0.5) for x, y in points
    )
    return {
        "available": True,
        "method": "segmentation_vs_landmark_extent_v1",
        "mask_box": envelope,
        "skeleton_box": box,
        "foreground_fraction": float(len(xs) / mask.size),
        "visible_landmarks_inside_mask": inside / len(points),
        "mask_boundary_certainty": float(np.mean(np.abs(mask - 0.5) * 2)),
        "calibrated_probability": None,
        "limitation": "Same pretrained backbone; agreement is not accuracy. Needs manual mask IoU and coach review.",
    }
