"""Pixel-level tamper features for one region of a card photo. Pure functions.

Each feature compares a region (the portrait, a digit field) with a ring of
card around it in the *same* photo, so the phone's own image pipeline -
sharpening, noise reduction, JPEG quality - cancels out. What is left is a
difference between the region and its surroundings:

    noise   log ratio of the high-frequency noise level (region / ring).
            A pasted photo carries its own camera's noise, usually smoother
            or grainier than the card around it.
    ela     log ratio of the JPEG recompression error (error level analysis).
            Pixels already compressed twice at the photo's quality barely
            change when saved again; a pasted, once-compressed patch changes
            more.
    clone   (character cells only) how well a cell's noise pattern matches
            somewhere else in the same field. Two genuinely printed digits
            share a shape but never their sensor noise; a digit copied from
            elsewhere on the line carries an identical copy of it.
    grid    difference in 8x8 JPEG block-grid strength at the photo's own
            grid phase. Content pasted after resizing or from another JPEG
            sits off the grid the rest of the photo shares.

None of these is a probability, and a genuine card's portrait legitimately
differs from its background (a printed photo next to guilloche). That is why
the pipeline never thresholds a raw value: it scores how far a feature is
from what genuine cards measured (models/forensics_calibration.json).
"""

from __future__ import annotations

import cv2
import numpy as np

FEATURES = ("noise", "ela", "grid", "clone")
_EPS = 1e-3


def to_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return gray.astype(np.float32)


def ring_mask(shape: tuple[int, int], box: tuple[int, int, int, int], margin: float) -> np.ndarray:
    """True on a band of width margin * box size around `box`, False inside it."""
    h, w = shape
    x1, y1, x2, y2 = box
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    mask = np.zeros((h, w), bool)
    mask[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)] = True
    mask[y1:y2, x1:x2] = False
    return mask


def box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def residual_map(gray: np.ndarray) -> np.ndarray:
    """Signed high-pass residual: what a 3x3 median filter removes."""
    smooth = cv2.medianBlur(np.clip(gray, 0, 255).astype(np.uint8), 3).astype(np.float32)
    return gray - smooth


def noise_map(gray: np.ndarray) -> np.ndarray:
    return np.abs(residual_map(gray))


def clone_score(residual: np.ndarray, cell: tuple[int, int, int, int]) -> float:
    """Best normalised correlation of the cell's residual with any other
    place in `residual` at least half a cell away from itself."""
    x1, y1, x2, y2 = cell
    tmpl = residual[y1:y2, x1:x2]
    if tmpl.shape[0] < 4 or tmpl.shape[1] < 4 or float(tmpl.std()) < 1e-3:
        return 0.0
    res = cv2.matchTemplate(residual, tmpl, cv2.TM_CCOEFF_NORMED)
    reach = max(1, (x2 - x1) // 2)
    res[max(0, y1 - reach):y1 + reach + 1, max(0, x1 - reach):x1 + reach + 1] = -1.0  # itself
    return float(np.nan_to_num(res).max())


def ela_map(img: np.ndarray, quality: int = 90) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not re-encode for ELA")
    again = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return np.abs(to_gray(img) - to_gray(again))


def grid_maps(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|second difference| across columns and across rows. JPEG block edges
    leave a step every 8 pixels, which shows up as extra energy at one phase."""
    dx = np.zeros_like(gray)
    dy = np.zeros_like(gray)
    dx[:, 1:-1] = np.abs(gray[:, 2:] - 2 * gray[:, 1:-1] + gray[:, :-2])
    dy[1:-1, :] = np.abs(gray[2:, :] - 2 * gray[1:-1, :] + gray[:-2, :])
    return dx, dy


def grid_phase(dx: np.ndarray, dy: np.ndarray, mask: np.ndarray | None = None) -> tuple[int, int]:
    """The photo's own 8x8 grid offset: the column/row phase with most energy."""
    return int(np.argmax(_phase_energy(dx, mask, axis=1))), int(np.argmax(_phase_energy(dy, mask, axis=0)))


def _phase_energy(d: np.ndarray, mask: np.ndarray | None, axis: int, offset: int = 0) -> np.ndarray:
    """Mean energy per grid phase; `offset` is the window's position in the
    photo, so phases stay those of the photo's own grid."""
    idx = (np.arange(d.shape[axis]) + offset) % 8
    out = np.zeros(8)
    for p in range(8):
        sel = d[:, idx == p] if axis == 1 else d[idx == p, :]
        if mask is not None:
            m = mask[:, idx == p] if axis == 1 else mask[idx == p, :]
            out[p] = sel[m].mean() if m.any() else 0.0
        else:
            out[p] = sel.mean()
    return out


def grid_strength(
    dx: np.ndarray, dy: np.ndarray, mask: np.ndarray, phase: tuple[int, int], origin: tuple[int, int] = (0, 0)
) -> float:
    """Energy at the photo's grid phase relative to the other seven phases.
    `origin` = (x, y) of the window these maps were cut from."""
    total = 0.0
    for d, axis, p, off in ((dx, 1, phase[0], origin[0]), (dy, 0, phase[1], origin[1])):
        e = _phase_energy(d, mask, axis, off)
        others = np.delete(e, p).mean()
        total += e[p] / (others + _EPS) - 1.0
    return total / 2


def _robust_level(values: np.ndarray) -> float:
    # Mean of the lower 80% of a heavy-tailed |residual|: text strokes and
    # edges are the tail, the noise floor is the bulk. (A plain median is
    # useless here - residuals are near-integers, so it is mostly 0 or 1.)
    if not values.size:
        return 0.0
    return float(values[values <= np.quantile(values, 0.8)].mean())


class PhotoMaps:
    """Per-photo maps computed once and shared by every region."""

    def __init__(self, img: np.ndarray, ela_quality: int = 90):
        self.shape = img.shape[:2]
        gray = to_gray(img)
        self.residual = residual_map(gray)
        self.noise = np.abs(self.residual)
        self.ela = ela_map(img, ela_quality)
        self.dx, self.dy = grid_maps(gray)
        self.phase = grid_phase(self.dx, self.dy)

    def features(
        self, box: tuple[int, int, int, int], cells: int = 1, margin: float = 0.35
    ) -> list[dict[str, float]] | None:
        """Features for `box`, or for each of `cells` equal vertical slices of
        it (one per character of a digit field).

        One cell: the region against a ring of card around it. Several cells:
        each slice against the rest of the field - digits against digits, so
        an edited character stands out from its neighbours instead of being
        averaged away over the whole line.
        """
        h, w = self.shape
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 16 or y2 - y1 < 8:
            return None
        # Work in a window around the box: whole-photo masks per character
        # cell cost seconds per card.
        mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
        wx1, wy1, wx2, wy2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
        win = (slice(wy1, wy2), slice(wx1, wx2))
        maps = (self.noise[win], self.ela[win], self.dx[win], self.dy[win], (wx1, wy1))
        shape = (wy2 - wy1, wx2 - wx1)
        bx1, by1, bx2, by2 = x1 - wx1, y1 - wy1, x2 - wx1, y2 - wy1
        whole = box_mask(shape, (bx1, by1, bx2, by2))
        if cells <= 1:
            ring = ~whole
            if ring.sum() < whole.sum() * 0.25:
                return None  # box at the photo's edge: nothing to compare with
            return [{**self._compare(maps, whole, ring), "clone": 0.0}]
        step = (bx2 - bx1) / cells
        if step < 4:
            return None
        field = self.residual[y1:y2, x1:x2]
        out = []
        for i in range(cells):
            cx1, cx2 = int(bx1 + i * step), int(bx1 + (i + 1) * step)
            cell = box_mask(shape, (cx1, by1, cx2, by2))
            clone = clone_score(field, (cx1 - bx1, 0, cx2 - bx1, by2 - by1))
            out.append({**self._compare(maps, cell, whole & ~cell), "clone": clone})
        return out

    def _compare(self, maps, inside: np.ndarray, ref: np.ndarray) -> dict[str, float]:
        noise, ela, dx, dy, origin = maps
        return {
            "noise": float(np.log((_robust_level(noise[inside]) + _EPS)
                                  / (_robust_level(noise[ref]) + _EPS))),
            "ela": float(np.log((ela[inside].mean() + _EPS) / (ela[ref].mean() + _EPS))),
            "grid": grid_strength(dx, dy, inside, self.phase, origin)
            - grid_strength(dx, dy, ref, self.phase, origin),
        }
