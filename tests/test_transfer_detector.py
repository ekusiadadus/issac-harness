"""Metric checks for the external transfer benchmark."""

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="run with --group transfer")
pytest.importorskip("sklearn", reason="run with --group transfer")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "transfer"))

from detector import (  # noqa: E402
    ap_for_class,
    evaluate,
    fit,
    input_audit,
    intersection_over_union,
    read_index,
)


def test_iou_xywh():
    assert intersection_over_union((0, 0, 10, 10), (5, 5, 10, 10)) == 25 / 175
    assert intersection_over_union((0, 0, 10, 10), (20, 20, 10, 10)) == 0


def test_ap_penalizes_ranked_false_positive_and_counts_negative_image():
    rows = [
        {"archive_member": "a", "objects": [{"category_id": 2, "bbox_xywh": [0, 0, 10, 10]}]},
        {"archive_member": "negative", "objects": []},
    ]
    predictions = [
        {"image": "negative", "detections": [{"category_id": 2, "bbox_xywh": [0, 0, 10, 10], "score": 2.0}]},
        {"image": "a", "detections": [{"category_id": 2, "bbox_xywh": [0, 0, 10, 10], "score": 1.0}]},
    ]
    result = ap_for_class(rows, predictions, 2, 0.5)
    assert result["ap"] == 0.5
    assert result["recall"] == 1
    assert result["gt"] == 1
    assert result["predictions"] == 2


def test_train_index_resolves_relative_image_from_index_directory(tmp_path):
    index = tmp_path / "frames.jsonl"
    index.write_text('{"image":"images/000001.png","objects":[{"category_id":2,"bbox_xywh":[1,2,3,4]}]}\n')
    assert read_index(index)[0]["image"] == str(tmp_path / "images" / "000001.png")


def test_arbitrary_class_and_optional_group_without_implicit_scene_exclusion(tmp_path):
    image = tmp_path / "a.png"
    cv2.imwrite(str(image), np.random.default_rng(5).integers(0, 255, (320, 320, 3), dtype=np.uint8))
    row = {"image": str(image), "id": "a", "scene": 40,
           "objects": [{"category_id": 71, "bbox_xywh": [80, 80, 64, 80]}]}
    models = fit([row], classes=(71,))
    assert models[71]["positive"] == 1
    with pytest.raises(ValueError, match="Insufficient synthetic samples"):
        fit([row], classes=(71,), exclude_train_scene=40)
    row["group"] = "native_camera_A"
    prediction = [{"image": "a", "detections": [{"category_id": 71, "bbox_xywh": [80, 80, 64, 80], "score": 1.0}]}]
    result = evaluate([row], prediction, classes=(71,))
    assert result["macro_ap50"] == 1.0
    assert "native_camera_A" in result["per_scene"]


def test_input_audit_hashes_actual_rgb_and_rejects_byte_identical_leakage(tmp_path):
    a, b = tmp_path / "train.png", tmp_path / "test.png"
    a.write_bytes(b"RGB-a")
    b.write_bytes(b"RGB-b")
    train_index, test_index = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    train_index.write_text('{"id":"train-a","image":"train.png","objects":[{"category_id":7,"bbox_xywh":[1,2,3,4]}]}\n')
    test_index.write_text('{"id":"test-b","image":"test.png","objects":[]}\n')
    train, test = read_index(train_index), read_index(test_index)
    result = input_audit(train_index, test_index, train, test, (7,), None)
    assert result["byte_identical_train_test_rgb_count"] == 0
    assert result["train_fit_counts"]["7"]["instances"] == 1
    assert result["train_fit_counts"]["7"]["image_examples"] == ["train-a"]
    b.write_bytes(a.read_bytes())
    with pytest.raises(ValueError, match="Byte-identical train/test RGB leakage"):
        input_audit(train_index, test_index, train, test, (7,), None)


def test_generic_evaluation_handles_all_negative_group():
    rows = [{"id": "negative-1", "image": "unused.png", "group": "empty", "objects": []}]
    predictions = [{"image": "negative-1", "detections": [{"category_id": 71, "bbox_xywh": [0, 0, 10, 10], "score": 1.0}]}]
    result = evaluate(rows, predictions, classes=(71,))
    assert result["macro_ap50"] is None
    assert result["per_scene"]["empty"]["71"]["predictions"] == 1


def test_group_holdout_audit_removes_group_from_fit(tmp_path):
    (tmp_path / "a.png").write_bytes(b"a")
    (tmp_path / "b.png").write_bytes(b"b")
    index = tmp_path / "both.jsonl"
    index.write_text('{"id":"a","image":"a.png","group":"holdout","objects":[]}\n'
                     '{"id":"b","image":"b.png","group":"fit","objects":[]}\n')
    rows = read_index(index)
    result = input_audit(index, index, rows, [rows[0]], (71,), None, "holdout")
    assert result["train_fit_images"] == 1
    assert result["test_images"] == 1
    assert result["byte_identical_train_test_rgb_count"] == 0
