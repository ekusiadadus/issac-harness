import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from harness.campaign import digest, inspect_frame, read, verify, write
from harness.sensor_dataset import derive, validate_valid_mask


def fixture(tmp_path):
    source = tmp_path / "native"
    source.mkdir()
    head = {"format": 2, "resolution": [12, 8], "seed": 8,
            "classes": {"part": 7}, "instances": [{"id": "p", "class": "part"}]}
    head["campaign_key"] = hashlib.sha256(json.dumps(head, sort_keys=True).encode()).hexdigest()
    write(source / "campaign.json", head)
    for index in (0, 1):
        frame = source / "frames" / f"{index:08d}"
        frame.mkdir(parents=True)
        yy, xx = np.indices((8, 12))
        rgb = np.stack([xx * 12 + index, yy * 20, xx * 6 + yy], axis=2).astype(np.uint8)
        inst = np.zeros((8, 12), np.uint16)
        inst[2:6, 2:6] = 1
        Image.fromarray(rgb).save(frame / "rgb.png")
        Image.fromarray((inst > 0).astype(np.uint8) * 7).save(frame / "classes.png")
        Image.fromarray(inst).save(frame / "instances.png")
        np.savez_compressed(frame / "range_m.npz", range_m=np.full((8, 12), 2.0, np.float32))
        write(frame / "annotation.json", {
            "campaign_key": head["campaign_key"], "index": index,
            "sample": {"camera": {"id": "native", "K": [10, 0, 6, 0, 10, 4, 0, 0, 1]},
                       "layout": "fixture"}, "readback": {},
            "objects": [{"id": "p", "class": "part", "category_id": 7,
                         "instance_id": 1, "bbox_xywh": [2, 2, 4, 4], "area": 16}],
            "sha256": {name: digest(frame / name) for name in
                       ("rgb.png", "classes.png", "instances.png", "range_m.npz")},
        })
        (frame / "record.sha256").write_text(digest(frame / "annotation.json") + "\n")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"name": "fixture lens", "provenance": "synthetic calibration",
                                   "width": 12, "height": 8,
                                   "K": [10, 0, 4, 0, 10, 4, 0, 0, 1],
                                   "D": [0, 0, 0, 0, 0]}))
    return source, profile


def test_batch_clips_boxes_and_exports_standard_training(tmp_path):
    source, profile = fixture(tmp_path)
    out = tmp_path / "derived"
    report = derive(source, profile, out, 51)
    assert report["frames"] == 2
    head = read(out / "campaign.json")
    assert head["kind"] == "sensor-derived"
    frame = out / "frames/00000000"
    annotation = inspect_frame(frame, head)
    validate_valid_mask(frame, annotation, head)
    assert annotation["objects"][0]["bbox_xywh"] == [0, 2, 4, 4]
    assert annotation["objects"][0]["area"] == 16
    assert annotation["sensor_derivative"]["source_annotation_sha256"] == digest(source / "frames/00000000/annotation.json")
    assert annotation["sensor_derivative"]["source_K"] != annotation["sensor_derivative"]["target_K"]
    assert annotation["sample"]["camera"]["K"] == annotation["sensor_derivative"]["target_K"]
    assert annotation["sample"]["camera"]["id"] == "native"
    assert annotation["sensor_derivative"]["seed"] != read(out / "frames/00000001/annotation.json")["sensor_derivative"]["seed"]
    assert verify(out, tmp_path / "export", 0, 2, coco=True)["boxes"] == 2


def test_resume_and_valid_tampering(tmp_path):
    source, profile = fixture(tmp_path)
    out = tmp_path / "derived"
    derive(source, profile, out, 52, start=0, count=1)
    original = digest(out / "frames/00000000/annotation.json")
    derive(source, profile, out, 52)
    assert digest(out / "frames/00000000/annotation.json") == original
    with pytest.raises(ValueError, match="different"):
        derive(source, profile, out, 53)
    valid = out / "frames/00000000/valid.png"
    valid.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="valid mask"):
        derive(source, profile, out, 52, start=0, count=1)


def test_resume_rejects_source_change_and_invalid_fov_plane(tmp_path):
    source, profile = fixture(tmp_path)
    out = tmp_path / "derived"
    derive(source, profile, out, 52, start=0, count=1)
    frame = out / "frames/00000000"
    head = read(out / "campaign.json")
    annotation = read(frame / "annotation.json")
    valid = np.asarray(Image.open(frame / "valid.png")).copy()
    valid[2, 2] = 0
    Image.fromarray(valid).save(frame / "valid.png")
    annotation["sensor_derivative"]["valid_sha256"] = digest(frame / "valid.png")
    annotation["sensor_derivative"]["valid_pixel_fraction"] = float((valid == 255).mean())
    write(frame / "annotation.json", annotation)
    (frame / "record.sha256").write_text(digest(frame / "annotation.json") + "\n")
    with pytest.raises(ValueError, match="outside valid"):
        validate_valid_mask(frame, annotation, head)

    # A changed source annotation is caught by its own seal before reuse.
    source_annotation = source / "frames/00000000/annotation.json"
    source_annotation.write_text(source_annotation.read_text() + " ")
    with pytest.raises(ValueError):
        derive(source, profile, out, 52, start=0, count=1)
