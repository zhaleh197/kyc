"""Is the card photo a photo of a screen or of a print, rather than of the card?

A card shown on a monitor or phone, or printed and photographed, keeps no
trace of how it was edited - the re-photograph erases the pixel signals in
signals.py - but leaves the screen's or printer's own fingerprint:

    spectral_peak   moire and pixel grids (screens) and halftone dot grids
                    (prints) are periodic, so the luma power spectrum has
                    sharp peaks above its smooth 1/f falloff
    chroma_peak     the same on the colour channels: screen moire is
                    coloured (the RGB subpixels beat against the camera's
                    Bayer grid)
    chroma_hf       colour fine detail relative to luma fine detail - a real
                    card's print has little colour texture at pixel scale
    dynamic_range   luma 1st-99th percentile span - prints lose contrast,
                    screens clip

The card itself is printed with fine guilloche lines, which are periodic too:
that is why nothing here is thresholded raw. Like the edit signals, each
feature is scored against genuine card photos (calibration.py region "card").
A learned patch classifier can replace this once there are enough labelled
screen / print shots (spec 06 section 13).
"""

from __future__ import annotations

import cv2
import numpy as np

FEATURES = ("spectral_peak", "chroma_peak", "chroma_hf", "dynamic_range")

TILE = 256
MAX_TILES = 16
BAND = (0.06, 0.48)  # cycles / pixel: above the card's layout, below aliasing


def _tiles(channel: np.ndarray) -> list[np.ndarray]:
    h, w = channel.shape
    ys = range(0, max(1, h - TILE + 1), TILE)
    xs = range(0, max(1, w - TILE + 1), TILE)
    tiles = [channel[y:y + TILE, x:x + TILE] for y in ys for x in xs]
    tiles = [t for t in tiles if t.shape == (TILE, TILE)]
    if len(tiles) > MAX_TILES:  # evenly spread, deterministic
        tiles = [tiles[i] for i in np.linspace(0, len(tiles) - 1, MAX_TILES).astype(int)]
    return tiles


_WINDOW = np.outer(np.hanning(TILE), np.hanning(TILE)).astype(np.float32)
_fy, _fx = np.meshgrid(np.fft.fftfreq(TILE), np.fft.fftfreq(TILE), indexing="ij")
_radius = np.sqrt(_fx**2 + _fy**2)
_ring_of = np.minimum((_radius * TILE).astype(int), TILE // 2)
_in_band = (_radius >= BAND[0]) & (_radius <= BAND[1])


def spectral_peak(channel: np.ndarray) -> float:
    """Strongest spectral peak above the radial (1/f) trend, in log10 power,
    from a Welch average over tiles. 0 = no periodic structure."""
    tiles = _tiles(channel.astype(np.float32))
    if not tiles:
        return 0.0
    power = np.zeros((TILE, TILE), np.float64)
    for t in tiles:
        power += np.abs(np.fft.fft2((t - t.mean()) * _WINDOW)) ** 2
    logp = np.log10(power / len(tiles) + 1e-6)
    trend = np.zeros(TILE // 2 + 1)
    for r in range(TILE // 2 + 1):
        ring = logp[_ring_of == r]
        trend[r] = np.median(ring) if ring.size else 0.0
    excess = logp - trend[_ring_of]
    return float(excess[_in_band].max())


def _highpass_energy(channel: np.ndarray) -> float:
    c = channel.astype(np.float32)
    return float(np.mean(np.abs(c - cv2.GaussianBlur(c, (0, 0), 1.5))))


def card_features(img: np.ndarray, box: tuple[int, int, int, int] | None = None) -> dict[str, float] | None:
    """Features on the card region of the photo as submitted (no resizing:
    moire lives at pixel scale)."""
    if box is not None:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        img = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if img.ndim != 3 or min(img.shape[:2]) < TILE:
        return None
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]
    lo, hi = np.percentile(y, (1, 99))
    return {
        "spectral_peak": spectral_peak(y),
        "chroma_peak": max(spectral_peak(cr), spectral_peak(cb)),
        "chroma_hf": float(np.log((_highpass_energy(cr) + _highpass_energy(cb) + 1e-3)
                                  / (_highpass_energy(y) + 1e-3))),
        "dynamic_range": float(hi - lo),
    }
