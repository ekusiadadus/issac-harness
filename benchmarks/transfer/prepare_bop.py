"""Extract a deterministic, small BOP YCB-V subset via HTTP ranges.

The image/annotation manifest is also the optional input format for native Isaac
frames: one JSON object per line with absolute ``image`` and an ``objects`` list
of ``category_id`` and ``bbox_xywh``. Only synthetic frames belong in train.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from range_zip import open_zip


ROOT = "https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/5c2c4aa229800355648cd268040aa814f8dc94f0"
CLASSES = (2, 3, 5)
TRAIN_SCENES = (0, 10, 20, 30, 40)
TEST_SCENES = (49, 52, 53, 54, 55, 56, 58, 59)


def extract(zip_file, member: str, destination: Path) -> str:
    data = zip_file.read(member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def prepare(part: str, root: Path, local_archive: Path | None = None) -> None:
    synthetic = part == "train"
    archive = "ycbv_train_pbr.zip" if synthetic else "ycbv_test_bop19.zip"
    split = "train_pbr" if synthetic else "test"
    scenes = TRAIN_SCENES if synthetic else TEST_SCENES
    if local_archive:
        zf, remote = zipfile.ZipFile(local_archive), None
    else:
        zf, remote = open_zip(f"{ROOT}/{archive}")
    records = []
    hashes = {}
    for scene in scenes:
        prefix = f"{split}/{scene:06d}"
        gt_member = f"{prefix}/scene_gt.json"
        info_member = f"{prefix}/scene_gt_info.json"
        gt = json.loads(zf.read(gt_member))
        info = json.loads(zf.read(info_member))
        camera_member = f"{prefix}/scene_camera.json"
        camera = json.loads(zf.read(camera_member)) if not synthetic else None
        for member in (gt_member, info_member):
            hashes[member] = extract(zf, member, root / member)
        if camera is not None:
            hashes[camera_member] = extract(zf, camera_member, root / camera_member)
        selected = sorted(gt, key=int)
        selected = selected[::50][:20] if synthetic else selected[::7][:11]
        for image_id in selected:
            objects = []
            for idx, (g, inf) in enumerate(zip(gt[image_id], info[image_id])):
                box = inf["bbox_visib"]
                if box[2] <= 0 or box[3] <= 0:
                    continue
                objects.append({
                    "category_id": g["obj_id"],
                    "bbox_xywh": box,
                    "instance_index": idx,
                    "pose_R": g["cam_R_m2c"],
                    "pose_t_mm": g["cam_t_m2c"],
                    "visibility": inf["visib_fract"],
                })
            if synthetic and not any(o["category_id"] in CLASSES for o in objects):
                continue
            ext = "jpg" if synthetic else "png"
            member = f"{prefix}/rgb/{int(image_id):06d}.{ext}"
            path = root / member
            hashes[member] = extract(zf, member, path)
            records.append({
                "image": member,
                "archive_member": member,
                "scene": scene,
                "image_id": int(image_id),
                "objects": objects,
                **({"camera": camera[image_id]} if camera is not None else {}),
            })
        print(f"{part} scene {scene}: total records {len(records)}"
              + (f"; transferred {remote.transferred / 1e6:.1f} MB" if remote else ""), flush=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{part}.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records))
    (root / f"{part}_sha256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"part": part, "archive": archive,
                      "archive_etag": remote.etag if remote else None,
                      "records": len(records), "requests": remote.requests if remote else 0,
                      "bytes": remote.transferred if remote else 0}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=("train", "test"))
    parser.add_argument("--root", type=Path, default=Path("/tmp/issac-transfer-bench/data"))
    parser.add_argument("--local-archive", type=Path)
    args = parser.parse_args()
    prepare(args.part, args.root, args.local_archive)
