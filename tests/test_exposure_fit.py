"""Native bracket selection and calibration leakage checks."""

import json

import numpy as np
from PIL import Image
import pytest

from harness.campaign import digest
from harness.exposure_fit import fit


def bracket(tmp_path, *, duplicate=False, incomplete=False, heldout=False):
    campaign = tmp_path / "campaign"
    frames = campaign / "frames"
    frames.mkdir(parents=True)
    references = tmp_path / "real"
    classes = np.zeros((12, 12), np.uint8)
    classes[2:10, 2:10] = 1
    instances = classes.astype(np.uint16)
    reference_ids = ["cal_a", "cal_b"] + (["heldout"] if heldout else [])
    for ref in reference_ids:
        path = references / ref
        path.mkdir(parents=True)
        Image.fromarray(np.full((12, 12, 3), 100, np.uint8)).save(path / "rgb.png")
        Image.fromarray(classes).save(path / "classes.png")
    header = {"campaign_key": "bracket", "resolution": [12, 12],
              "classes": {"target": 1}, "instances": [{"id": "obj", "class": "target"}]}
    (campaign / "campaign.json").write_text(json.dumps(header))
    cases = [(r, ev, sim) for ev, sim in [(-2, 160), (-3, 101), (-4, 50)] for r in reference_ids[:2]]
    if incomplete:
        cases.remove(("cal_b", -3, 101))
    if duplicate:
        cases.append(("cal_a", -3, 101))
    if heldout:
        cases.append(("heldout", 99, 255))
    for i, (ref, ev, sim) in enumerate(cases):
        frame = frames / f"{i:08d}"
        frame.mkdir()
        pixels = np.full((12, 12, 3), sim, np.uint8)
        pixels[0, 0] = 0
        Image.fromarray(pixels).save(frame / "rgb.png")
        Image.fromarray(classes).save(frame / "classes.png")
        Image.fromarray(instances).save(frame / "instances.png")
        np.savez_compressed(frame / "range_m.npz", range_m=np.ones((12, 12), np.float32))
        ann = {"campaign_key": "bracket", "index": i,
               "sample": {"reference": ref, "edits": {"light_gain": ev}, "layout": "scene"},
               "objects": [{"instance_id": 1, "category_id": 1, "class": "target",
                            "bbox_xywh": [2, 2, 8, 8], "area": 64}],
               "sha256": {n: digest(frame / n) for n in ("rgb.png", "classes.png", "instances.png", "range_m.npz")}}
        (frame / "annotation.json").write_text(json.dumps(ann))
    return campaign, references


def test_selects_observed_scalar_and_freezes_evidence(tmp_path):
    campaign, references = bracket(tmp_path, heldout=True)
    result = fit(campaign, references, ["cal_a", "cal_b"], tmp_path / "freeze.json")
    assert result["chosen_gain"] == -3
    assert result["candidate_count"] == 3
    assert {r["reference"] for c in result["candidates"] for r in c["per_reference"]} == {"cal_a", "cal_b"}
    assert all(r["sha256"]["real_rgb"] for c in result["candidates"] for r in c["per_reference"])
    assert result["candidates"][1]["matched_pixels"] == 32  # 4x4 interior per view
    assert json.loads((tmp_path / "freeze.json").read_text())["chosen_gain"] == -3


def test_rejects_incomplete_bracket(tmp_path):
    campaign, references = bracket(tmp_path, incomplete=True)
    with pytest.raises(ValueError, match="incomplete bracket"):
        fit(campaign, references, ["cal_a", "cal_b"], tmp_path / "freeze.json")


def test_rejects_duplicate_reference_at_gain(tmp_path):
    campaign, references = bracket(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate calibration reference"):
        fit(campaign, references, ["cal_a", "cal_b"], tmp_path / "freeze.json")


def test_rejects_missing_gain_and_invalid_reference_list(tmp_path):
    campaign, references = bracket(tmp_path)
    with pytest.raises(ValueError, match="distinct calibration"):
        fit(campaign, references, ["cal_a", "cal_a"], tmp_path / "freeze.json")
    with pytest.raises(ValueError, match="missing sample.edits.ev"):
        fit(campaign, references, ["cal_a", "cal_b"], tmp_path / "freeze.json", gain_field="edits.ev")
