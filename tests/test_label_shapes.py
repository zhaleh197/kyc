"""Polygon labels must be read as polygons, not as the first two vertices.

Roboflow exports a segmentation project as `class x1 y1 x2 y2 ... xn yn`.
Reading only the first four numbers would treat two vertices as a YOLO box and
put every annotation in the wrong place - with no error raised anywhere, which
is what makes this worth a test.
"""

from __future__ import annotations

import numpy as np

from kyc.modules.ocr import rectify
from scripts.rectify_dataset import label_to_points, xyxy_to_yolo


def test_four_values_are_read_as_a_yolo_box():
    points = label_to_points("0.5 0.5 0.4 0.2", 100, 100)
    assert points == [(30.0, 40.0), (70.0, 40.0), (70.0, 60.0), (30.0, 60.0)]


def test_polygon_is_read_as_all_of_its_vertices():
    # A triangle: three vertices, six numbers.
    points = label_to_points("0.1 0.1 0.9 0.2 0.5 0.8", 100, 200)
    assert points == [(10.0, 20.0), (90.0, 40.0), (50.0, 160.0)]


def test_polygon_bounding_box_covers_every_vertex():
    """The bug this guards: taking values[:4] would give a box around the first
    two vertices only, which is both wrong and plausible-looking."""
    label = "0.1 0.1 0.9 0.2 0.5 0.8"
    points = label_to_points(label, 100, 200)
    identity = np.eye(3, dtype=np.float64)
    box = rectify.transform_points(identity, points, (100, 200))
    assert box == (10.0, 20.0, 90.0, 160.0)

    naive = label_to_points(" ".join(label.split()[:4]), 100, 200)
    naive_box = rectify.transform_points(identity, naive, (100, 200))
    assert naive_box != box, "reading four values must not accidentally match"


def test_malformed_labels_are_rejected_rather_than_guessed():
    assert label_to_points("0.1 0.2 0.3", 100, 100) is None       # odd count, too short
    assert label_to_points("0.1 0.2 0.3 0.4 0.5", 100, 100) is None  # odd count
    assert label_to_points("a b c d", 100, 100) is None


def test_round_trip_through_yolo_format():
    points = label_to_points("0.5 0.5 0.4 0.2", 200, 100)
    box = rectify.transform_points(np.eye(3), points, (200, 100))
    assert xyxy_to_yolo(box, 200, 100).split() == ["0.500000", "0.500000", "0.400000", "0.200000"]
