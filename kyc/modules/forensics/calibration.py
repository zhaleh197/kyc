"""The genuine-card reference the forensic features are scored against.

Per feature, a region's deviation is how unusual its most unusual cell is
compared with genuine cards, in robust standard deviations:

    deviation_f = max over cells |value_f - genuine median_f| / genuine spread_f

Digit fields are split into one cell per character (see PhotoMaps.features);
the portrait is a single cell. Each feature is divided by its own genuine
scale (the tails of the features differ a lot, so one shared scale would let
the heaviest-tailed feature set it for all), then one factor sets the warning
line for the whole photo at the target FPR (see fit). The region score is the
largest deviation / limit: 1.0 is exactly at the warning line. Measured by
scripts/calibrate_forensics.py, stored in
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
# The genuine-deviation quantile each feature is first divided by (see fit).
SCALE_QUANTILE = 0.9

Cells = list[dict[str, float]]


def fit(genuine: list[dict[str, Cells]], fpr: float, regions_to_fit=None) -> dict:
    """`genuine`: one {region: cells} per genuine photo. Regions with fewer
    than 20 photos are left out (not judged).

    Two steps. Per feature, the genuine deviation at SCALE_QUANTILE puts the
    features on one scale - a quantile a few unusual photos cannot move; set
    at the tail (1 - fpr/len(features)) it was those photos' own value, so
    they scored at most 1 in the fit set while new photos had no ceiling. Then
    all limits are multiplied by one factor so that a whole *photo* - its
    largest ratio over every region and feature - warns on `fpr` of genuine
    photos. Without the second step every region x feature would spend the
    full budget on its own: with about 15 of them, and limits at the
    largest value a few dozen photos show, 30-40% of genuine cards outside
    the fit set warned. The factor uses the conformal rank ceil((n+1)(1-fpr)),
    so it stays honest for photos not in the fit set; with fewer than
    1/fpr - 1 photos that rank is past the end and the largest is used,
    which bounds the FPR at about 1/(n+1) instead of `fpr`."""
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
        limits = {f: round(float(np.quantile([d[f] for d in devs], SCALE_QUANTILE)), 3) for f in features}
        # A feature that never varies on genuine cards (clone on the portrait,
        # a single cell) has nothing to say about this region.
        entry["warn_above"] = {f: v for f, v in limits.items() if v > 0}
        regions[region] = entry
    calib = {"version": 2, "target_fpr": fpr, "regions": regions}
    scores = [photo_score(calib, g) for g in genuine]
    scores = sorted(s for s in scores if s is not None)
    if scores:
        rank = int(np.ceil((len(scores) + 1) * (1 - fpr)))
        factor = max(scores[min(rank, len(scores)) - 1], 1e-3)
        for entry in regions.values():
            entry["warn_above"] = {f: round(v * factor, 3) for f, v in entry["warn_above"].items()}
        calib["photo_factor"] = round(factor, 4)
        calib["fit_photos"] = len(scores)
    return calib


def photo_score(calib: dict, feats: dict[str, Cells]) -> float | None:
    """A photo's largest region score; None when no fitted region is seen."""
    scores = [region_score(calib, r, cells)[0] for r, cells in feats.items()
              if r in calib["regions"] and calib["regions"][r]["warn_above"]]
    return max(scores) if scores else None


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
