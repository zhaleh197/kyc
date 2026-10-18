"""Bring a photographed card upright before its fields are cropped and read.

    photo -> deskew (small tilt, from edge directions) -> pick the quarter turn
          (0/90/180/270) whose detections best fit the profile's layout

The field detector still finds boxes on a card turned 90 degrees (Roboflow
trained it with rotations), but a field cropped from a sideways card holds
vertical text, which the recognisers cannot read - so orientation must be
fixed first, not inferred from detection confidence: measured on a real card,
a 90-degree turn even scored *higher* mean confidence (0.86) than upright
(0.80). Upside down, the detector lost 5 of 8 fields and face match read an
upside-down face (similarity 0.09 instead of 0.44).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import Detection, best_per_class
from .profiles import DocumentProfile

DetectFn = Callable[[np.ndarray], list[Detection]]


@dataclass
class Orientation:
    image: np.ndarray
    rotation: int  # quarter turn applied, degrees clockwise
    deskew_deg: float  # small rotation applied before it, degrees counter-clockwise
    detections: list[Detection]
    layout_score: float
    # 3x3 homography from the input image to `image`, so boxes found on the
    # upright card can be mapped back onto the original photo.
    matrix: np.ndarray = field(default_factory=lambda: np.eye(3))


def estimate_skew(gray: np.ndarray) -> float:
    """Dominant edge direction folded into [-45, 45) degrees. Card borders and
    text lines all run along the card's axes, so the strongest gradients point
    along them. Returns the angle the image must be rotated by (counter-
    clockwise, OpenCV convention) to level it."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    strong = mag > np.percentile(mag, 90)
    if strong.sum() < 50:
        return 0.0
    angles = np.degrees(np.arctan2(gy[strong], gx[strong]))
    folded = (angles + 45.0) % 90.0 - 45.0
    hist, edges = np.histogram(folded, bins=90, range=(-45, 45), weights=mag[strong])
    # Circular smoothing: -45 and +45 are the same direction.
    smooth = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), np.ones(5) / 5, mode="valid")
    peak = int(np.argmax(smooth))
    return float((edges[peak] + edges[peak + 1]) / 2)


def rotate_bound_matrix(w: int, h: int, angle: float) -> tuple[np.ndarray, tuple[int, int]]:
    """2x3 affine for rotate_bound on a (w, h) image, and the new canvas size."""
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return m, (nw, nh)


def rotate_bound(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate counter-clockwise by `angle` degrees, growing the canvas so
    nothing is cut off."""
    h, w = img.shape[:2]
    m, size = rotate_bound_matrix(w, h, angle)
    return cv2.warpAffine(img, m, size, borderMode=cv2.BORDER_REPLICATE)


def quarter_turn_matrix(w: int, h: int, rotation: int) -> np.ndarray:
    """3x3 homography for cv2.rotate by `rotation` degrees clockwise on (w, h)."""
    return {
        0: np.eye(3),
        90: np.array([[0.0, -1.0, h - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        180: np.array([[-1.0, 0.0, w - 1.0], [0.0, -1.0, h - 1.0], [0.0, 0.0, 1.0]]),
        270: np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, w - 1.0], [0.0, 0.0, 1.0]]),
    }[rotation]


_QUARTER = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def layout_score(located: dict[str, Detection], profile: DocumentProfile) -> float:
    """Fraction of the profile's layout rules the detections satisfy (0 if
    none can be checked)."""
    checked = satisfied = 0
    for a, relation, b in profile.layout:
        if a not in located or b not in located:
            continue
        ax1, ay1, ax2, ay2 = located[a].box
        bx1, by1, bx2, by2 = located[b].box
        acx, acy, bcx, bcy = (ax1 + ax2) / 2, (ay1 + ay2) / 2, (bx1 + bx2) / 2, (by1 + by2) / 2
        checked += 1
        satisfied += {"left_of": acx < bcx, "above": acy < bcy}[relation]
    return satisfied / checked if checked else 0.0


def wide_fraction(located: dict[str, Detection], profile: DocumentProfile) -> float:
    """Fraction of detected text/digit/date fields wider than tall. Lines of
    text are wide on an upright (or upside-down) card and tall on a sideways
    one - the layout rules alone can tie between those."""
    text = [s.detector_class for s in profile.value_fields if s.detector_class in located]
    if not text:
        return 0.0
    wide = sum((d.box[2] - d.box[0]) > (d.box[3] - d.box[1]) for d in (located[c] for c in text))
    return wide / len(text)


def orient(
    img: np.ndarray,
    detect: DetectFn,
    profile: DocumentProfile,
    *,
    min_skew_deg: float = 2.0,
    max_skew_deg: float = 35.0,
) -> Orientation:
    # 1. Small tilt: measured on the card itself where the detector finds it.
    first = detect(img)
    located = best_per_class(first)
    card = located.get(profile.container_class) if profile.container_class else None
    region = img
    if card is not None:
        x1, y1, x2, y2 = card.box
        if x2 - x1 > 32 and y2 - y1 > 32:
            region = img[y1:y2, x1:x2]
    skew = estimate_skew(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY))
    # Near +-45 the folded histogram cannot tell a tilt from diagonal
    # content (measured: -42.5 on a card tilted -20); quarter turns cover
    # large rotations, so only trust moderate estimates.
    deskew = np.eye(3)
    if min_skew_deg <= abs(skew) <= max_skew_deg:
        h, w = img.shape[:2]
        deskew[:2] = rotate_bound_matrix(w, h, skew)[0]
        img = rotate_bound(img, skew)
        first = detect(img)
    else:
        skew = 0.0

    # 2. Quarter turn: the profile's layout decides, detections break ties.
    best: Orientation | None = None
    best_rank: tuple = ()
    for rotation in (0, 90, 180, 270):
        candidate = img if rotation == 0 else cv2.rotate(img, _QUARTER[rotation])
        detections = first if rotation == 0 else detect(candidate)
        found = best_per_class(detections)
        score = layout_score(found, profile)
        # Layout first, then wide text lines, then a small preference for
        # leaving the image as it is; field count and confidence last.
        fit = score + wide_fraction(found, profile) + (0.05 if rotation == 0 else 0.0)
        rank = (round(fit, 3), len(found), float(np.mean([d.confidence for d in found.values()])) if found else 0.0)
        if best is None or rank > best_rank:
            turn = quarter_turn_matrix(img.shape[1], img.shape[0], rotation)
            best, best_rank = Orientation(candidate, rotation, round(skew, 1), detections, score, turn @ deskew), rank
        if rotation == 0 and score == 1.0 and len(found) >= len(profile.class_names) - 2:
            break  # upright and complete: no need to try other turns
    return best
