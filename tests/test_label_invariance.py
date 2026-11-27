import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from harness import label_invariance


def _write_json(path, data):
    path.write_text(json.dumps(data, sort_keys=True))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal(folder):
    annotation = folder / "annotation.json"
    data = json.loads(annotation.read_text())
    for name in data["sha256"]:
        data["sha256"][name] = _hash(folder / name)
    _write_json(annotation, data)
    (folder / "record.sha256").write_text(_hash(annotation) + "\n")


def _frame(root, index, header, sample, readback, instance):
    folder = root / "frames" / f"{index:08d}"
    folder.mkdir(parents=True)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :, 0] = 25 + index * 10
    rgb[0, 0, 1] = 100
    cls = np.where(instance == 1, 21, 0).astype(np.uint8)
    for name, pixels in (("rgb.png", rgb), ("instances.png", instance), ("classes.png", cls)):
        Image.fromarray(pixels).save(folder / name)
    np.savez(folder / "range_m.npz", range_m=np.ones((4, 4), dtype=np.float32))
    yy, xx = np.where(instance == 1)
    annotation = {
        "campaign_key": header["campaign_key"], "index": index,
        "sample": sample, "readback": readback,
        "objects": [{"id": "crate", "instance_id": 1, "category_id": 21,
                     "class": "carton", "area": len(xx),
                     "bbox_xywh": [int(xx.min()), int(yy.min()), int(np.ptp(xx)) + 1, int(np.ptp(yy)) + 1]}],
        "native_audit": {"instance_path_pixels_different": 0,
                         "path_ids": {"0": "BACKGROUND", "2": "/World/Crate"},
                         "semantic_ids": {"0": {"class": "BACKGROUND"}, "2": {"class": "capture00001"}}},
        "sha256": {name: _hash(folder / name) for name in
                   ("rgb.png", "instances.png", "classes.png", "range_m.npz")},
    }
    _write_json(folder / "annotation.json", annotation)
    (folder / "record.sha256").write_text(_hash(folder / "annotation.json") + "\n")
    return folder


def _campaign(tmp_path, *, flip=False, pose_change=False):
    header = {"format": 2, "resolution": [4, 4], "instances": [{"id": "crate", "class": "carton", "prim": "/World/Crate"}],
              "classes": {"carton": 21}, "sources": [{"path": "scene.usda", "sha256": "same"}]}
    header["campaign_key"] = hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest()
    _write_json(tmp_path / "campaign.json", header)
    instance = np.zeros((4, 4), dtype=np.uint8)
    instance[1:3, 1:3] = 1
    for index in (0, 1):
        mask = instance.copy()
        if flip and index == 1:
            mask[1, 1] = 0
        sample = {"layout": "fixed", "camera": {"id": "front", "K": [1]},
                  "objects": {"crate": {"visible": True, "position": [0, 0, 0]}},
                  "edits": {"mark_visible": bool(index), "light": 1}}
        readback = {"camera": {"matrix_row": [1], "K": [1]},
                    "objects": {"crate": {"visible": True, "matrix_row": [0]}},
                    "edits": {"mark_visible": "invisible" if index == 0 else "inherited", "light": 1}}
        if pose_change and index == 1:
            sample["objects"]["crate"]["position"] = [1, 0, 0]
            readback["objects"]["crate"]["matrix_row"] = [1]
        _frame(tmp_path, index, header, sample, readback, mask)
    return tmp_path


def test_stable_masks_pass(tmp_path):
    root = _campaign(tmp_path)
    result = label_invariance.compare(root, 0, 1, edits=["mark_visible"], target="crate", min_rgb_pixels=1)
    assert result["passed"]
    assert result["instance_changed_pixels"] == result["class_changed_pixels"] == 0


def test_mask_flip_is_a_failure(tmp_path):
    root = _campaign(tmp_path, flip=True)
    result = label_invariance.compare(root, 0, 1, edits=["mark_visible"], target="crate")
    assert not result["passed"]
    assert result["instance_transitions"] == {"1->0": 1}
    assert result["target_class_lost_pixels"] == 1


def test_pose_change_is_rejected(tmp_path):
    root = _campaign(tmp_path, pose_change=True)
    try:
        label_invariance.compare(root, 0, 1, edits=["mark_visible"])
    except ValueError as exc:
        assert "pose/visibility" in str(exc)
    else:
        raise AssertionError("pose change passed the fixed-view control")


def test_tampered_record_is_rejected(tmp_path):
    root = _campaign(tmp_path)
    annotation = root / "frames" / "00000001" / "annotation.json"
    annotation.write_text(annotation.read_text() + " ")
    try:
        label_invariance.compare(root, 0, 1, edits=["mark_visible"])
    except ValueError as exc:
        assert "modified" in str(exc)
    else:
        raise AssertionError("unsealed frame passed")


def test_bounded_decal_range_and_target_rgb_threshold(tmp_path):
    root = _campaign(tmp_path)
    folder = root / "frames" / "00000001"
    np.savez(folder / "range_m.npz", range_m=np.full((4, 4), 1.0008, dtype=np.float32))
    _reseal(folder)
    result = label_invariance.compare(root, 0, 1, edits=["mark_visible"], target="crate",
                                      range_atol_m=0.002, rgb_delta=0.01,
                                      min_target_rgb_pixels=4)
    assert result["passed"]
    assert result["target_rgb_changed_pixels"] == 4
    assert result["protected_range_violating_pixels"] == 0
    assert result["protected_range_max_delta_m"] < 0.002
    exact = label_invariance.compare(root, 0, 1, edits=["mark_visible"], target="crate")
    assert not exact["passed"]
    assert exact["protected_range_violating_pixels"] == 16


def test_raw_native_ids_may_reindex(tmp_path):
    root = _campaign(tmp_path)
    folder = root / "frames" / "00000001"
    annotation = folder / "annotation.json"
    data = json.loads(annotation.read_text())
    data["native_audit"]["path_ids"] = {"0": "BACKGROUND", "7": "/World/Crate/Decal"}
    data["native_audit"]["semantic_ids"] = {"0": {"class": "BACKGROUND"}, "9": {"class": "capture00001"}}
    _write_json(annotation, data)
    _reseal(folder)
    assert label_invariance.compare(root, 0, 1, edits=["mark_visible"])["passed"]


def test_ambiguous_native_semantic_mapping_is_rejected(tmp_path):
    root = _campaign(tmp_path)
    folder = root / "frames" / "00000001"
    annotation = folder / "annotation.json"
    data = json.loads(annotation.read_text())
    data["native_audit"]["instance_path_pixels_different"] = 1
    _write_json(annotation, data)
    _reseal(folder)
    try:
        label_invariance.compare(root, 0, 1, edits=["mark_visible"])
    except ValueError as exc:
        assert "native instance path" in str(exc)
    else:
        raise AssertionError("native mapping disagreement passed")


def test_missing_native_owner_path_is_rejected(tmp_path):
    root = _campaign(tmp_path)
    folder = root / "frames" / "00000001"
    annotation = folder / "annotation.json"
    data = json.loads(annotation.read_text())
    data["native_audit"]["path_ids"] = {"0": "BACKGROUND", "7": "/World/Other"}
    _write_json(annotation, data)
    _reseal(folder)
    try:
        label_invariance.compare(root, 0, 1, edits=["mark_visible"])
    except ValueError as exc:
        assert "native path mapping missing" in str(exc)
    else:
        raise AssertionError("missing native target path passed")


def test_infinite_background_range_handling(tmp_path):
    root = _campaign(tmp_path)
    folders = [root / "frames" / f"{index:08d}" for index in (0, 1)]
    for folder in folders:
        depth = np.ones((4, 4), dtype=np.float32)
        depth[0, 0] = np.inf
        np.savez(folder / "range_m.npz", range_m=depth)
        _reseal(folder)
    assert label_invariance.compare(root, 0, 1, edits=["mark_visible"])["passed"]
    depth[0, 0] = 1
    np.savez(folders[1] / "range_m.npz", range_m=depth)
    _reseal(folders[1])
    result = label_invariance.compare(root, 0, 1, edits=["mark_visible"], range_atol_m=1)
    assert not result["passed"]
    assert result["protected_range_violating_pixels"] == 1
    assert result["protected_range_max_delta_m"] == "inf"


def test_target_rgb_floor_uses_stable_target_only(tmp_path):
    root = _campaign(tmp_path)
    result = label_invariance.compare(root, 0, 1, edits=["mark_visible"], target="crate",
                                      rgb_delta=0.1, min_target_rgb_pixels=1)
    assert not result["passed"]
    assert result["target_rgb_changed_pixels"] == 0
    assert not result["target_rgb_floor_met"]


def test_observed_native_mark_failure():
    root = Path(__file__).resolve().parents[1] / "evidence/campaign-l4/mark-comparison"
    if not (root / "campaign.json").exists():
        return
    result = label_invariance.compare(root, 0, 1, edits=["mark_visible"], target="crate")
    assert result["instance_changed_pixels"] == 13738
    assert result["class_changed_pixels"] == 13738
    assert result["target_instance_lost_pixels"] == 13738


def test_release_enforces_fixed_view_and_binds_both_records(tmp_path):
    from harness.qualification import release

    root = _campaign(tmp_path, flip=True)
    policy = tmp_path / "policy.toml"
    policy.write_text('[[appearance_pairs]]\nfirst = 0\nsecond = 1\nedits = ["mark_visible"]\n')
    report = release(root, policy, tmp_path / "result")
    assert report["status"] == "fail"
    assert report["checks"][0]["observed"]["instance_changed_pixels"] == 1
    for i in (0, 1):
        assert report["evidence_sha256"][f"appearance_frame_{i}"] == _hash(
            root / "frames" / f"{i:08d}" / "annotation.json"
        )


def test_coverage_counts_bytes_for_selected_bundles(tmp_path):
    from harness.campaign import texture_registry
    from harness.qualification import release

    root = _campaign(tmp_path)
    for path, content in (("one", b"texture bytes"), ("alias", b"texture bytes"), ("other", b"other bytes")):
        (root / path).write_bytes(content)
    spec = {"id": "paper", "kind": "texture_set", "choices": ["one", "alias", "other"],
            "bundles": {key: {"base": key} for key in ("one", "alias", "other")}}
    head = json.loads((root / "campaign.json").read_text())
    head["texture_sets"] = texture_registry(root, [spec])
    head.pop("campaign_key")
    head["campaign_key"] = hashlib.sha256(json.dumps(head, sort_keys=True).encode()).hexdigest()
    _write_json(root / "campaign.json", head)
    policy = tmp_path / "policy.toml"
    policy.write_text("[min_distinct_texture_sets]\npaper = 2\n")
    for selection, expected in (("alias", "fail"), ("other", "pass")):
        for i, choice in enumerate(("one", selection)):
            folder = root / "frames" / f"{i:08d}"
            a = json.loads((folder / "annotation.json").read_text())
            a["campaign_key"] = head["campaign_key"]
            for field in ("sample", "readback"):
                a[field]["edits"]["paper"] = choice
            _write_json(folder / "annotation.json", a)
            _reseal(folder)
        assert release(root, policy, tmp_path / "result")["status"] == expected
