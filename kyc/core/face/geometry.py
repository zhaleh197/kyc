"""Face geometry from landmarks: head pose and eye openness.

Pure numpy, no model, so every function here is unit-testable with
hand-made points.
"""

from __future__ import annotations

import numpy as np

# 2d106det indices, read off a rendered face (image left = the person's right).
# Each eye: two corners, then three upper-lid / lower-lid pairs left to right.
EYE_LEFT_IMG = {"corners": (35, 39), "pairs": ((41, 36), (40, 33), (42, 37))}
EYE_RIGHT_IMG = {"corners": (89, 93), "pairs": ((95, 90), (94, 87), (96, 91))}

# Generic face proportions behind the closed-form pose below (arbitrary units):
# eye centres at x = +-EYE_HALF_WIDTH, EYE_DEPTH behind the nose tip; eyes
# EYE_HEIGHT above and the mouth MOUTH_DROP below the nose, the mouth
# MOUTH_DEPTH behind it. Only the ratios matter. The resulting angles are
# a stable "head turned past N degrees" signal, not a metrology-grade pose;
# the limits are calibrated on real frames.
EYE_HALF_WIDTH = 30.0
EYE_DEPTH = 24.0
EYE_HEIGHT = 32.0
MOUTH_DROP = 30.0
MOUTH_DEPTH = 22.0


def eye_aspect_ratio(points: np.ndarray, eye: dict) -> float:
    """Mean lid opening over eye width. ~0.25-0.35 open, falls towards 0 closed."""
    c1, c2 = eye["corners"]
    width = float(np.linalg.norm(points[c1] - points[c2]))
    if width <= 1e-6:
        return 0.0
    opening = np.mean([np.linalg.norm(points[a] - points[b]) for a, b in eye["pairs"]])
    return float(opening / width)


def eye_openness(points: np.ndarray) -> tuple[float, float]:
    """(image-left eye, image-right eye) aspect ratios from 106 landmarks."""
    return eye_aspect_ratio(points, EYE_LEFT_IMG), eye_aspect_ratio(points, EYE_RIGHT_IMG)


def roll_degrees(kps: np.ndarray) -> float:
    """In-plane tilt from the eye line; positive = clockwise in the image."""
    dx, dy = kps[1] - kps[0]
    return float(np.degrees(np.arctan2(dy, dx)))


def _derolled(kps: np.ndarray) -> np.ndarray:
    """Rotate the keypoints about the eye midpoint so the eye line is level."""
    angle = np.radians(roll_degrees(kps))
    c, s = np.cos(-angle), np.sin(-angle)
    centre = (kps[0] + kps[1]) / 2
    return (kps - centre) @ np.array([[c, s], [-s, c]]) + centre


def head_pose(kps: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from SCRFD's 5 keypoints.

    Closed form against the generic proportions above, instead of solvePnP:
    5-point EPnP was measured to be unstable here (mirroring an image moved
    its yaw from -6 to +20 degrees instead of to +6). These formulas are
    mirror-symmetric by construction.

    Yaw: rotating the model by t about the vertical axis makes the
    nose-to-eye distances 30cos t +- 24sin t, so their normalised difference
    r = (24/30) tan t.  Pitch: likewise from where the nose sits between the eye
    line and the mouth line.

    Signs: yaw > 0 when the nose moves towards image right; pitch > 0 when
    the chin goes up; roll > 0 clockwise in the image.
    """
    roll = roll_degrees(kps)
    k = _derolled(kps.astype(np.float64))
    eye_l, eye_r, nose, mouth_l, mouth_r = k

    d_left = nose[0] - eye_l[0]
    d_right = eye_r[0] - nose[0]
    span = d_left + d_right
    r = (d_left - d_right) / span if abs(span) > 1e-6 else 0.0
    yaw = float(np.degrees(np.arctan(r * EYE_HALF_WIDTH / EYE_DEPTH)))

    eyes_y = (eye_l[1] + eye_r[1]) / 2
    mouth_y = (mouth_l[1] + mouth_r[1]) / 2
    up = nose[1] - eyes_y
    down = mouth_y - nose[1]
    total = up + down
    q = up / total if abs(total) > 1e-6 else EYE_HEIGHT / (EYE_HEIGHT + MOUTH_DROP)
    # q(62cos p + 2sin p) = 32cos p + 24sin p  ->  tan p = (62q - 32) / (24 - 2q)
    a = EYE_HEIGHT + MOUTH_DROP
    b = EYE_DEPTH - MOUTH_DEPTH
    tan_p = (a * q - EYE_HEIGHT) / (EYE_DEPTH - b * q)
    # Nose nearer the mouth (q up) means the head is tilted down.
    pitch = -float(np.degrees(np.arctan(tan_p)))
    return yaw, pitch, roll
