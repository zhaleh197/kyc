"""Field boxes put back on their own rows.

On a card whose value rows are printed one under the other at an even pitch
(the Iraqi card: name, father, grandfather, surname, mother, ...), a row's
position says which field it is better than the detector's label does. The
detector learnt the rows from cards that had all of them, so on a card with
an empty row (no surname) it labels the rows below one row up: the mother's
name comes back as "surname" and "mother" is not found.

`place_rows` fits the pitch to the detected rows (the line through the most
of them), moves every box to the row it sits on and lists the rows in the
middle that have no box.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np

from .detector import Detection

# A box belongs to a row when its centre is within this share of the pitch.
ON_ROW = 0.3


@dataclass(frozen=True)
class RowPlacement:
    located: dict[str, Detection]
    moved: list[tuple[str, str]]  # (detector's label, row it sits on)
    empty: list[str]  # rows between found rows that have no box
    boxes: dict[str, tuple[int, int, int, int]]  # where each empty row is


def _centre(d: Detection) -> float:
    return (d.box[1] + d.box[3]) / 2


def place_rows(located: dict[str, Detection], order: tuple[str, ...]) -> RowPlacement:
    rows = [(order.index(k), _centre(d), k) for k, d in located.items() if k in order]
    unchanged = RowPlacement(dict(located), [], [], {})
    if len(rows) < 3:
        return unchanged

    # The line y = a + b * row through the most boxes (ties: least error).
    best = None
    for (s1, y1, _), (s2, y2, _) in combinations(rows, 2):
        if s1 == s2 or (y2 - y1) / (s2 - s1) <= 0:
            continue
        b = (y2 - y1) / (s2 - s1)
        a = y1 - b * s1
        err = [abs(y - (a + b * s)) / b for s, y, _ in rows]
        fit = sum(e < ON_ROW for e in err), -sum(min(e, ON_ROW) for e in err)
        if best is None or fit > best[0]:
            best = (fit, a, b)
    if best is None or best[0][0] < 3:
        return unchanged
    _, a, b = best

    # Every box to the row it sits on; two on one row: the surer one stays.
    placed: dict[str, Detection] = {}
    moved: list[tuple[str, str]] = []
    for k in sorted((k for k in located if k in order), key=lambda k: -located[k].confidence):
        y = _centre(located[k])
        slot = round((y - a) / b)
        if not 0 <= slot < len(order) or abs(y - (a + b * slot)) / b >= ON_ROW:
            continue  # on no row: a stray box
        row = order[slot]
        if row in placed:
            continue
        placed[row] = replace(located[k], class_name=row, class_index=located[k].class_index)
        if row != k:
            moved.append((k, row))

    out = {k: d for k, d in located.items() if k not in order}
    out.update(placed)

    # Rows inside the found block that have no box: where they would be.
    found = sorted(order.index(k) for k in placed)
    empty, boxes = [], {}
    if found:
        dets = list(placed.values())
        half = float(np.median([d.box[3] - d.box[1] for d in dets])) / 2
        x1 = min(d.box[0] for d in dets)
        x2 = int(np.median([d.box[2] for d in dets]))
        for slot in range(found[0] + 1, found[-1]):
            row = order[slot]
            if row not in placed:
                y = a + b * slot
                empty.append(row)
                boxes[row] = (x1, int(y - half), x2, int(y + half))
    return RowPlacement(out, moved, empty, boxes)
