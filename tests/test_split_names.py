"""Normalising an already-normalised dataset must not lose the val split.

Roboflow ships the validation images in `valid/`; ultralytics wants `val:` in
data.yaml. Running the normaliser over its own output once dropped the split
entirely, and the failure surfaced far downstream as ultralytics refusing the
data.yaml - so both spellings are covered here.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import yaml

from scripts.roboflow_pull import normalise


def build_dataset(root, val_dir_name: str) -> None:
    """A miniature YOLO dataset: one image and one label per split."""
    names = ["birthday", "exp", "fathername", "idcard", "idnumber", "img", "lastname", "name"]
    (root / "data.yaml").write_text(
        yaml.safe_dump({"train": "../train/images", "val": f"../{val_dir_name}/images", "nc": 8, "names": names}),
        encoding="utf-8",
    )
    for split in ("train", val_dir_name, "test"):
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        cv2.imwrite(str(images / "card.jpg"), np.full((64, 96, 3), 200, np.uint8))
        (labels / "card.txt").write_text("4 0.5 0.5 0.2 0.1\n", encoding="utf-8")


@pytest.mark.parametrize("val_dir_name", ["valid", "val"])
def test_both_validation_split_spellings_survive(tmp_path, val_dir_name):
    source = tmp_path / "src"
    source.mkdir()
    build_dataset(source, val_dir_name)

    out = tmp_path / "out"
    normalise(source, out, "ir_national_card_front")

    written = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert "val" in written, f"val split lost when the source used {val_dir_name}/"
    assert (out / "val" / "images" / "card.jpg").exists()
    assert (out / "train" / "images" / "card.jpg").exists()
