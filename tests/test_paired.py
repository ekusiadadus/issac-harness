"""Registered scoring math and fail-closed campaign integration."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from harness.campaign import digest
from harness.paired import evaluate, radial_to_z, score_frame


def fixture(tmp_path, *, missing_reference=False, sim_depth_m=1.0, sim_shift=0):
    campaign = tmp_path / "campaign"
    frame = campaign / "frames" / "00000000"
    frame.mkdir(parents=True)
    refs = tmp_path / "real"
    ref = refs / "sample_1"
    if not missing_reference:
        ref.mkdir(parents=True)
    h = w = 12
    real_cls = np.zeros((h, w), np.uint8)
    real_cls[2:10, 2:10] = 1
    sim_cls = np.zeros_like(real_cls)
    sim_cls[2:10, 2 + sim_shift:10 + sim_shift] = 1
    sim_inst = sim_cls.astype(np.uint16)
    rgb = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
    rgb = np.stack((rgb, rgb, rgb), axis=-1)
    Image.fromarray(rgb).save(frame / "rgb.png")
    Image.fromarray(sim_cls).save(frame / "classes.png")
    Image.fromarray(sim_inst).save(frame / "instances.png")
    K = [10, 0, 6, 0, 10, 6, 0, 0, 1]
    y, x = np.indices((h, w))
    radial = sim_depth_m * np.sqrt(1 + ((x + .5 - 6) / 10) ** 2 + ((y + .5 - 6) / 10) ** 2)
    np.savez_compressed(frame / "range_m.npz", range_m=radial.astype(np.float32))
    if not missing_reference:
        Image.fromarray(rgb).save(ref / "rgb.png")
        Image.fromarray(real_cls).save(ref / "classes.png")
        Image.fromarray(np.full((h, w), 1000, np.uint16)).save(ref / "z-depth.png")
    head = {"campaign_key": "test", "resolution": [w, h], "classes": {"target": 1},
            "instances": [{"id": "x", "class": "target"}]}
    (campaign / "campaign.json").write_text(json.dumps(head))
    yy, xx = np.where(sim_cls == 1)
    ann = {"campaign_key": "test", "index": 0,
           "sample": {"reference": "sample_1", "depth_scale_mm": 1, "layout": "scene_1", "camera": {"K": K}},
           "readback": {"camera": {"K": K}},
           "objects": [{"instance_id": 1, "category_id": 1, "class": "target",
                        "bbox_xywh": [int(xx.min()), int(yy.min()), int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1)],
                        "area": len(xx)}],
           "sha256": {n: digest(frame / n) for n in ("rgb.png", "classes.png", "instances.png", "range_m.npz")}}
    (frame / "annotation.json").write_text(json.dumps(ann))
    config = tmp_path / "quality.toml"
    config.write_text("[scenes.scene_1]\niou_macro_min=0.80\nz_median_abs_m_max=0.015\nrgb_mae_max=0.10\n")
    return campaign, refs, config, frame, head


def test_radial_depth_is_optical_z_at_pixel_centres():
    K = [100, 0, 1, 0, 100, 1, 0, 0, 1]
    y, x = np.indices((2, 2))
    radial = np.sqrt(1 + ((x + .5 - 1) / 100) ** 2 + ((y + .5 - 1) / 100) ** 2)
    assert np.allclose(radial_to_z(radial, K), 1)


def test_identical_registered_pair_passes(tmp_path):
    campaign, refs, config, frame, head = fixture(tmp_path)
    item = score_frame(frame, head, refs)
    assert item["classes"]["target"]["iou"] == 1
    assert item["z_median_abs_m"] < 1e-7
    assert item["rgb_mae"] == 0
    result = evaluate(campaign, refs, config, tmp_path / "out")
    assert result["status"] == "pass"
    assert (tmp_path / "out/strips/00000000.png").exists()


def test_depth_and_mask_mismatch_fail_gate(tmp_path):
    campaign, refs, config, _, _ = fixture(tmp_path, sim_depth_m=1.04, sim_shift=2)
    report = evaluate(campaign, refs, config, tmp_path / "out")
    assert report["status"] == "fail"
    assert report["scenes"]["scene_1"]["metrics"]["iou_macro"] < .80
    assert report["scenes"]["scene_1"]["metrics"]["z_median_abs_m"] > .015


def test_missing_reference_fails_closed(tmp_path):
    campaign, refs, config, _, _ = fixture(tmp_path, missing_reference=True)
    report = evaluate(campaign, refs, config, tmp_path / "out")
    assert report["status"] == "fail"
    assert "missing reference" in report["failures"][0]["reason"]


def test_unsupported_real_class_fails_closed(tmp_path):
    campaign, refs, config, _, _ = fixture(tmp_path)
    path = refs / "sample_1/classes.png"
    labels = np.asarray(Image.open(path)).copy()
    labels[0, 0] = 99
    Image.fromarray(labels).save(path)
    report = evaluate(campaign, refs, config, tmp_path / "out")
    assert report["status"] == "fail"
    assert "unsupported class IDs" in report["failures"][0]["reason"]


def test_missing_real_depth_fails_closed(tmp_path):
    campaign, refs, config, _, _ = fixture(tmp_path)
    (refs / "sample_1/z-depth.png").unlink()
    report = evaluate(campaign, refs, config, tmp_path / "out")
    assert report["status"] == "fail"
    assert "z-depth.png" in report["failures"][0]["reason"]


def test_recorded_real_rgb_hash_is_checked_and_all_real_hashes_reported(tmp_path):
    campaign, refs, config, frame, _ = fixture(tmp_path)
    ann_path = frame / "annotation.json"
    ann = json.loads(ann_path.read_text())
    ann["sample"]["rgb_sha256"] = "0" * 64
    ann_path.write_text(json.dumps(ann))
    bad = evaluate(campaign, refs, config, tmp_path / "bad")
    assert bad["status"] == "fail"
    assert "real RGB hash mismatch" in bad["failures"][0]["reason"]
    ann["sample"]["rgb_sha256"] = digest(refs / "sample_1/rgb.png")
    ann_path.write_text(json.dumps(ann))
    good = evaluate(campaign, refs, config, tmp_path / "good")
    assert good["status"] == "pass"
    assert set(good["frames"][0]["real_sha256"]) == {"rgb.png", "classes.png", "z-depth.png"}
