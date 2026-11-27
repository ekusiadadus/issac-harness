"""Small geometry check for the optional template detector."""

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="run with --group transfer")
pytest.importorskip("sklearn", reason="run with --group transfer")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "transfer"))

from detector import intersection_over_union  # noqa: E402
from sift_detector import detect, fit_templates, load_native_instance_mask  # noqa: E402


def test_sift_template_localizes_translated_printed_patch(tmp_path):
    rng = np.random.default_rng(71)
    patch = rng.integers(0, 255, (100, 120, 3), dtype=np.uint8)
    training = np.zeros((320, 320, 3), dtype=np.uint8)
    training[50:150, 60:180] = patch
    target = np.zeros_like(training)
    target[130:230, 170:290] = patch
    path = tmp_path / "synthetic.png"
    cv2.imwrite(str(path), training)
    row = {"image": str(path), "id": "synthetic", "objects": [
        {"category_id": 71, "bbox_xywh": [60, 50, 120, 100]}]}
    model = fit_templates([row], (71,), tmp_path, None)
    predictions = detect(target, model)
    assert predictions
    assert max(intersection_over_union(p["bbox_xywh"], [170, 130, 120, 100]) for p in predictions) > 0.8


def test_native_16bit_instance_ids_and_group_template_selection(tmp_path):
    rng = np.random.default_rng(72)
    rows = []
    for group in ("layout_a", "layout_b"):
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        image[50:150, 60:180] = rng.integers(0, 255, (100, 120, 3), dtype=np.uint8)
        ids = np.zeros((320, 320), dtype=np.uint16)
        ids[50:150, 60:180] = 257
        ids[180:220, 60:180] = 1
        image_path, mask_path = tmp_path / f"{group}.png", tmp_path / f"{group}_ids.png"
        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(mask_path), ids)
        row = {"id": group, "group": group, "image": str(image_path),
               "instance_mask": mask_path.name,
               "objects": [{"category_id": 71, "instance_id": 257, "bbox_xywh": [60, 50, 120, 100]}]}
        decoded = load_native_instance_mask(row, tmp_path)
        assert decoded.dtype == np.uint16
        assert decoded[60, 70] == 257
        assert decoded[190, 70] == 1
        rows.append(row)
    model = fit_templates(rows, (71,), tmp_path, None)
    assert {template["group"] for template in model[71]} == {"layout_a", "layout_b"}
