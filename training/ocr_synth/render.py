"""Draw one line of Arabic-script text, properly shaped.

PIL without libraqm draws Persian and Kurdish letter by letter, unjoined - a
training image that looks like that teaches the wrong thing. Shaping is done
by HarfBuzz (contextual forms, ligatures, right-to-left runs) and the glyphs
are rasterised by FreeType at the positions HarfBuzz gives.

    mask = render_line("ڕێباز حەمە", font, size_px=40)   # uint8, 255 = ink
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import freetype
import numpy as np
import uharfbuzz as hb


@lru_cache(maxsize=64)
def _hb_font(path: str) -> tuple[hb.Font, int]:
    blob = hb.Blob.from_file_path(path)
    face = hb.Face(blob)
    font = hb.Font(face)
    return font, face.upem


@lru_cache(maxsize=64)
def _ft_face(path: str) -> freetype.Face:
    return freetype.Face(path)


def covers(path: Path, text: str) -> bool:
    """True when the font has a glyph for every non-space character."""
    face = _ft_face(str(path))
    return all(face.get_char_index(ord(c)) for c in set(text) if not c.isspace() and c != "‌")


def render_line(text: str, font_path: Path, size_px: int) -> np.ndarray:
    """Shaped text as a tight uint8 mask (255 = ink), drawn left to right as
    it appears on paper."""
    path = str(font_path)
    font, upem = _hb_font(path)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()  # right-to-left, Arabic script
    hb.shape(font, buf, {"kern": True, "liga": True})

    face = _ft_face(path)
    face.set_pixel_sizes(0, size_px)
    scale = size_px / upem
    ascender = face.size.ascender / 64
    descender = -face.size.descender / 64
    height = int(np.ceil(ascender + descender)) + 4
    width = int(np.ceil(sum(p.x_advance for p in buf.glyph_positions) * scale)) + size_px
    canvas = np.zeros((height, max(width, 1)), np.uint8)

    pen_x = size_px / 2
    baseline = ascender + 2
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions, strict=True):
        face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
        glyph = face.glyph
        bitmap = glyph.bitmap
        if bitmap.width and bitmap.rows:
            glyph_img = np.array(bitmap.buffer, np.uint8).reshape(bitmap.rows, bitmap.pitch)[:, : bitmap.width]
            x = int(round(pen_x + pos.x_offset * scale + glyph.bitmap_left))
            y = int(round(baseline - pos.y_offset * scale - glyph.bitmap_top))
            _blit_max(canvas, glyph_img, x, y)
        pen_x += pos.x_advance * scale

    ys, xs = np.nonzero(canvas)
    if not len(xs):
        return canvas[:1, :1]
    return canvas[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _blit_max(canvas: np.ndarray, glyph: np.ndarray, x: int, y: int) -> None:
    """Draw `glyph` at (x, y), keeping the darker ink where glyphs overlap
    (joined letters overlap a pixel or two)."""
    h, w = glyph.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    region = canvas[y0:y1, x0:x1]
    np.maximum(region, glyph[y0 - y : y1 - y, x0 - x : x1 - x], out=region)
