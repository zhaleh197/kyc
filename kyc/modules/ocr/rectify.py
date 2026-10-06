"""Locate the card in the photo and flatten it to a canonical rectangle.

Why this exists: a phone photo of an ID card is rotated, tilted and shot at an
angle. Feeding that straight into a field detector wastes most of the model's
capacity on learning perspective. Rectifying first means the detector always
sees the same layout at the same scale, which is what makes a small CPU model
good enough.

The classical contour approach here runs with zero trained weights, so the
pipeline is usable before the field detector exists. When the detector is
available the pipeline can refine the crop using its `idcard` box instead.
"""

from __future__ import annotations

import cv2
import numpy as np

from kyc.core.imaging import brightness, glare_ratio, limit_size, sharpness, to_gray
from kyc.core.schemas import Reason

# A quad is only believable as a card if it covers a decent slice of the frame
# and is roughly card-shaped.
MIN_AREA_RATIO = 0.15
ASPECT_TOLERANCE = 0.45


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return the 4 points as top-left, top-right, bottom-right, bottom-left.

    Uses coordinate sums/differences rather than angles: the corner with the
    smallest x+y is always top-left regardless of rotation up to ~45 degrees.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _quad_aspect(quad: np.ndarray) -> float:
    tl, tr, br, bl = quad
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return float(width / height) if height > 1 else 0.0


def find_card_quad(img: np.ndarray, target_aspect: float) -> np.ndarray | None:
    """Best 4-point card outline in image coordinates, or None."""
    work_width = 800
    h, w = img.shape[:2]
    scale = work_width / max(w, 1)
    small = cv2.resize(img, (work_width, max(1, int(h * scale)))) if w > work_width else img.copy()
    inv_scale = w / small.shape[1]

    gray = to_gray(small)
    # Bilateral keeps the card border crisp while flattening the print inside,
    # which stops Canny from firing on the text and photo.
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    median = float(np.median(gray))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = small.shape[0] * small.shape[1]
    best: np.ndarray | None = None
    best_area = 0.0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(contour)
        if area < frame_area * MIN_AREA_RATIO:
            break  # sorted descending: everything after is smaller too
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = order_corners(approx)
        aspect = _quad_aspect(quad)
        if abs(aspect - target_aspect) > target_aspect * ASPECT_TOLERANCE:
            continue
        if area > best_area:
            best_area, best = area, quad

    if best is None:
        return None
    return best * inv_scale


def warp_card(img: np.ndarray, quad: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_CUBIC)


def scale_matrix(src_size: tuple[int, int], dst_size: tuple[int, int]) -> np.ndarray:
    """Homography for a plain resize, so the fallback path is composable too."""
    (src_w, src_h), (dst_w, dst_h) = src_size, dst_size
    return np.array(
        [[dst_w / src_w, 0.0, 0.0], [0.0, dst_h / src_h, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotate_cw_matrix(size: tuple[int, int]) -> np.ndarray:
    """Homography for cv2.ROTATE_90_CLOCKWISE on an image of `size` (w, h).

    Under that rotation a point (x, y) lands at (h - 1 - y, x).
    """
    _, height = size
    return np.array([[0.0, -1.0, height - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rectify_with_transform(
    img: np.ndarray,
    target_size: tuple[int, int],
    target_aspect: float,
) -> tuple[np.ndarray, list[Reason], np.ndarray]:
    """Rectify and also return the 3x3 transform from source to card pixels.

    The transform is what lets a *labelled* dataset be rectified alongside its
    images: the detector is trained on rectified cards because that is what it
    sees at inference time, so the annotations have to move through the exact
    same geometry. Returning it here keeps that single source of truth instead
    of a second, drifting implementation in the dataset tooling.
    """
    reasons: list[Reason] = []
    src_h, src_w = img.shape[:2]
    quad = find_card_quad(img, target_aspect)

    if quad is None:
        reasons.append(
            Reason.warn(
                "card_border_not_found",
                "Could not find the card outline; using the whole frame",
                "لبهٔ کارت پیدا نشد؛ کل تصویر پردازش می‌شود",
            )
        )
        card = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        matrix = scale_matrix((src_w, src_h), target_size)
    else:
        width, height = target_size
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst).astype(np.float64)
        card = cv2.warpPerspective(img, matrix, target_size, flags=cv2.INTER_CUBIC)

    # A card photographed rotated 90 degrees warps into a portrait rectangle.
    if card.shape[0] > card.shape[1]:
        rotated_size = (card.shape[0], card.shape[1])  # w, h swap
        rotation = rotate_cw_matrix((card.shape[1], card.shape[0]))
        card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
        resize = scale_matrix(rotated_size, target_size)
        card = cv2.resize(card, target_size, interpolation=cv2.INTER_CUBIC)
        matrix = resize @ rotation @ matrix
        reasons.append(
            Reason.info("card_rotated", "Card was rotated to landscape", "کارت به حالت افقی چرخانده شد")
        )
    return card, reasons, matrix


def rectify(img: np.ndarray, target_size: tuple[int, int], target_aspect: float) -> tuple[np.ndarray, list[Reason]]:
    """Return (rectified card, findings).

    Falls back to a plain resize when no card outline is found, so a tightly
    cropped scan still works - it just gets a warning that no border was seen.
    """
    card, reasons, _ = rectify_with_transform(img, target_size, target_aspect)
    return card, reasons


def transform_points(
    matrix: np.ndarray,
    points: list[tuple[float, float]],
    size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Project an outline through `matrix` and return its bounding box.

    Takes points rather than a box so a polygon annotation keeps all of its
    vertices through the warp. Collapsing to a box first and warping that would
    lose the shape and inflate the result on a tilted card.
    """
    if not points:
        return None
    array = np.array([[p] for p in points], dtype=np.float32)
    projected = cv2.perspectiveTransform(array, matrix.astype(np.float64)).reshape(-1, 2)
    width, height = size
    x1 = max(0.0, float(projected[:, 0].min()))
    y1 = max(0.0, float(projected[:, 1].min()))
    x2 = min(float(width), float(projected[:, 0].max()))
    y2 = min(float(height), float(projected[:, 1].max()))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return x1, y1, x2, y2


def transform_box(
    matrix: np.ndarray,
    box: tuple[float, float, float, float],
    size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Map an axis-aligned box through `matrix`, returning its new AABB.

    All four corners are projected, not just two: under a perspective warp the
    corners move independently, so mapping only the diagonal would shrink or
    skew the box. Returns None if the result lands entirely outside the frame.
    """
    x1, y1, x2, y2 = box
    corners = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(corners, matrix.astype(np.float64)).reshape(4, 2)
    width, height = size
    nx1, ny1 = projected.min(axis=0)
    nx2, ny2 = projected.max(axis=0)
    nx1, ny1 = max(0.0, float(nx1)), max(0.0, float(ny1))
    nx2, ny2 = min(float(width), float(nx2)), min(float(height), float(ny2))
    if nx2 - nx1 < 1.0 or ny2 - ny1 < 1.0:
        return None
    return nx1, ny1, nx2, ny2


def assess_quality(
    card: np.ndarray,
    *,
    min_sharpness: float,
    min_brightness: float,
    max_brightness: float,
    max_glare_ratio: float,
) -> tuple[list[Reason], dict]:
    """Gate the rectified card before spending CPU on detection and OCR.

    Every metric is returned alongside the findings so a tuning session can
    look at real numbers instead of guessing at thresholds.
    """
    metrics = {
        "sharpness": round(sharpness(card), 2),
        "brightness": round(brightness(card), 2),
        "glare_ratio": round(glare_ratio(card), 4),
    }
    reasons: list[Reason] = []

    if metrics["sharpness"] < min_sharpness:
        reasons.append(
            Reason.error(
                "card_blurry",
                f"Card is out of focus (sharpness {metrics['sharpness']} < {min_sharpness})",
                "تصویر کارت تار است؛ لطفاً دوباره و واضح‌تر عکس بگیرید",
            )
        )
    if metrics["brightness"] < min_brightness:
        reasons.append(
            Reason.error(
                "card_too_dark",
                f"Card is underexposed (brightness {metrics['brightness']})",
                "تصویر کارت خیلی تاریک است",
            )
        )
    elif metrics["brightness"] > max_brightness:
        reasons.append(
            Reason.error(
                "card_too_bright",
                f"Card is overexposed (brightness {metrics['brightness']})",
                "تصویر کارت خیلی روشن است",
            )
        )
    if metrics["glare_ratio"] > max_glare_ratio:
        reasons.append(
            Reason.warn(
                "card_glare",
                f"Reflection covers {metrics['glare_ratio'] * 100:.1f}% of the card",
                "بازتاب نور روی کارت افتاده است",
            )
        )
    return reasons, metrics


def prepare(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale a huge phone photo once, up front."""
    return limit_size(img, max_side)
