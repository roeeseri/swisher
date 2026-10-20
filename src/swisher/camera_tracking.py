"""Map a user-selected shooter region through camera motion using background features.

Anchor-to-current matching avoids cumulative drift. Failed estimates abstain rather
than falling back to a fixed region which may now contain another person.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraTransform:
    matrix: np.ndarray | None
    inliers: int
    matches: int


class CameraRegionTracker:
    def __init__(self, frame: np.ndarray, excluded_region=None):
        height, width = frame.shape[:2]
        self.width, self.height = width, height
        self.scale = min(640 / width, 1.0)
        self.orb = cv2.ORB_create(nfeatures=1200)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        gray = self._gray(frame)
        mask = np.full(gray.shape, 255, np.uint8)
        if excluded_region:
            x1, y1, x2, y2 = (int(v * self.scale) for v in excluded_region)
            mask[y1:y2, x1:x2] = 0
        self.anchor_points, self.anchor_descriptors = self.orb.detectAndCompute(gray, mask)
        self.last = CameraTransform(np.eye(2, 3), 0, 0)

    def _gray(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, None, fx=self.scale, fy=self.scale)

    def update(self, frame: np.ndarray) -> CameraTransform:
        points, descriptors = self.orb.detectAndCompute(self._gray(frame), None)
        if descriptors is None or self.anchor_descriptors is None:
            self.last = CameraTransform(None, 0, 0)
            return self.last
        pairs = self.matcher.knnMatch(self.anchor_descriptors, descriptors, k=2)
        matches = [
            a
            for pair in pairs
            if len(pair) == 2
            for a, b in [pair]
            if a.distance < 0.72 * b.distance
        ]
        if len(matches) < 15:
            self.last = CameraTransform(None, 0, len(matches))
            return self.last
        src = np.float32([self.anchor_points[m.queryIdx].pt for m in matches])
        dst = np.float32([points[m.trainIdx].pt for m in matches])
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
        )
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if matrix is not None:
            scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
            if inliers < 15 or inliers / len(matches) < 0.5 or not 0.7 <= scale <= 1.4:
                matrix = None
            else:
                matrix[:, 2] /= self.scale
        self.last = CameraTransform(matrix, inliers, len(matches))
        return self.last

    def region(self, original_region) -> tuple[int, int, int, int] | None:
        if self.last.matrix is None:
            return None
        x1, y1, x2, y2 = original_region
        corners = np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])
        moved = cv2.transform(corners, self.last.matrix)[0]
        lo = np.floor(moved.min(axis=0)).astype(int)
        hi = np.ceil(moved.max(axis=0)).astype(int)
        rx1, ry1 = np.maximum(lo, [0, 0])
        rx2, ry2 = np.minimum(hi, [self.width, self.height])
        if rx2 <= rx1 or ry2 <= ry1:
            return None
        return int(rx1), int(ry1), int(rx2), int(ry2)
