"""The genuine-card reference the forensic features are scored against.

Per feature, a region's deviation is how unusual its most unusual cell is
compared with genuine cards, in robust standard deviations:

    deviation_f = max over cells |value_f - genuine median_f| / genuine spread_f

Digit fields are split into one cell per character (see PhotoMaps.features);
the portrait is a single cell. Each feature has its own limit, the genuine
deviation at the 1 - FPR/len(features) quantile (so the region as a whole
stays near the target FPR), and the region score is the largest
deviation / limit: 1.0 is exactly at the warning line. The tails of the three
features differ a lot, so one shared limit would let the heaviest-tailed
feature set it for all. Measured by scripts/eval_forensics.py, stored in
models/forensics_calibration.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# region (OCR field key) -> cells. 10 = the national id's digits; dates are
# 10 characters (yyyy/mm/dd) too.
REGIONS = {"portrait": 1, "national_id": 10, "birth_date": 10, "expiry_date": 10}
# The whole card, scored with the recapture (screen / print) features.
CARD = "card"

Cells = list[dict[str, float]]


def fit(genuine: list[dict[str, Cells]], fpr: float, regions_to_fit=None) -> dict:
    """`genuine`: one {region: cells} per genuine photo. Regions with fewer
    than 20 photos are left out (not judged)."""
    regions = {}
    for region in regions_to_fit or REGIONS:
        rows = [g[region] for g in genuine if region in g]
        if len(rows) < 20:
            continue
        pooled = [cell for row in rows for cell in row]
        features = list(pooled[0])
        median, scale = {}, {}
        for feat in features:
            v = np.array([c[feat] for c in pooled])
            med = float(np.median(v))
            median[feat] = round(med, 5)
            scale[feat] = round(max(float(np.median(np.abs(v - med))) * 1.4826, 1e-3), 5)
        entry = {"cells": REGIONS.get(region, 1), "median": median, "scale": scale, "n": len(rows)}
        devs = [deviations({"regions": {region: entry}}, region, row) for row in rows]
        limits = {f: round(float(np.quantile([d[f] for d in devs], 1 - fpr / len(features))), 3) for f in features}
        # A feature that never varies on genuine cards (clone on the portrait,
        # a single cell) has nothing to say about this region.
        entry["warn_above"] = {f: v for f, v in limits.items() if v > 0}
        regions[region] = entry
    return {"version": 1, "target_fpr": fpr, "regions": regions}


def deviations(calib: dict, region: str, cells: Cells) -> dict[str, float]:
    """Per feature: the most unusual cell's distance from genuine, in spreads."""
    ref = calib["regions"][region]
    return {f: max(abs(c[f] - ref["median"][f]) / ref["scale"][f] for c in cells) for f in ref["median"]}


def region_score(calib: dict, region: str, cells: Cells) -> tuple[float, str]:
    """(largest deviation / its limit, the feature it came from). >= 1 warns."""
    limits = calib["regions"][region]["warn_above"]
    ratios = {f: d / limits[f] for f, d in deviations(calib, region, cells).items() if f in limits}
    feat = max(ratios, key=ratios.get)
    return ratios[feat], feat


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
