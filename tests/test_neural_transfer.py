"""Small split and target checks for the random-weight SSDLite path."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "transfer"))
from neural_detector import clipped_targets, fit_rows  # noqa: E402


def test_fit_rows_excludes_holdout_group_but_keeps_rows_without_scene():
    rows = [{"id": "fit", "group": "train"},
            {"id": "holdout", "group": "cross_pick"},
            {"id": "ungrouped"}]
    assert [r["id"] for r in fit_rows(rows)] == ["fit", "ungrouped"]
    assert [r["id"] for r in fit_rows(rows, excluded_group=None)] == ["fit", "holdout", "ungrouped"]


def test_targets_clip_and_map_sparse_category_ids_with_negative_image(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "image.png"
    Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(path)
    row = {"image": str(path), "objects": [
        {"category_id": 71, "bbox_xywh": [-5, 3, 20, 20]},
        {"category_id": 900, "bbox_xywh": [40, 30, 30, 20]},
        {"category_id": 71, "bbox_xywh": [100, 5, 20, 10]},
        {"category_id": 7, "bbox_xywh": [0, 0, 5, 5]},
    ]}
    image, target = clipped_targets(row, (71, 900), Image, np, torch)
    assert image.shape == (3, 40, 60)
    assert target["boxes"].tolist() == [[0, 3, 15, 23], [40, 30, 60, 40]]
    assert target["labels"].tolist() == [1, 2]
    row["objects"] = []
    _, empty = clipped_targets(row, (71, 900), Image, np, torch)
    assert empty["boxes"].shape == (0, 4)
    assert empty["labels"].shape == (0,)
