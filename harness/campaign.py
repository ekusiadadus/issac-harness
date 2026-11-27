"""Plan, verify and export portable, sharded Isaac Sim capture campaigns."""

import argparse
import hashlib
import json
import math
import os
import tomllib
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def texture_registry(root, edits):
    """Identify a finish by all its texture bytes, including the channel names."""
    hashes = {}
    registry = {}
    for spec in edits:
        if spec["kind"] != "texture_set":
            continue
        choices = {}
        for name in spec["choices"]:
            channels = {}
            for channel, filename in spec["bundles"][name].items():
                path = (root / filename).resolve()
                if path not in hashes:
                    hashes[path] = digest(path)
                channels[channel] = hashes[path]
            choices[name] = channels
        registry[spec["id"]] = choices
    return registry


def write(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )


def read(path):
    return json.loads(path.read_text())


def prescribed(root, cfg):
    return read(root / cfg["poses"]) if "poses" in cfg else None


def load(path):
    root = path.resolve().parent
    with path.open("rb") as src:
        cfg = tomllib.load(src)
    instances = cfg["instances"]
    ids = [i["id"] for i in instances]
    prims = [i["prim"] for i in instances]
    categories = cfg["classes"]
    if (
        not ids
        or len(set(ids)) != len(ids)
        or len(set(prims)) != len(prims)
        or any(not p.startswith("/") for p in prims)
        or any(p.startswith(q + "/") for p in prims for q in prims if p != q)
    ):
        raise ValueError("instances need unique ids and disjoint absolute USD prims")
    if (
        not categories
        or len(set(categories.values())) != len(categories)
        or any(type(n) is not int or not 1 <= n < 255 for n in categories.values())
        or any(i["class"] not in categories for i in instances)
        or len(ids) >= 65535
    ):
        raise ValueError("invalid class ids or instance ontology")
    if (
        any(int(cfg[k]) < 1 for k in ("width", "height"))
        or cfg.get("meters_per_unit", 1) <= 0
    ):
        raise ValueError("invalid image or world units")
    if not (root / cfg["stage"]).is_file():
        raise ValueError("stage file missing")
    for i in instances:
        if "asset" in i and not (root / i["asset"]).is_file():
            raise ValueError(f"asset missing for {i['id']}: {i['asset']}")
    for field in ("edits", "cameras", "layouts"):
        names = [v["id"] for v in cfg.get(field, [])]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate {field} ID")
    for obj in instances:
        if not 0 <= obj.get("probability", 1) <= 1:
            raise ValueError("probability must be in [0,1]")
    cases = prescribed(root, cfg)
    if cases is None:
        if not cfg.get("cameras") or not cfg.get("layouts"):
            raise ValueError("campaign requires cameras and layouts, or recorded poses")
        for layout in cfg["layouts"]:
            if set(layout["anchors"]) != set(ids):
                raise ValueError("each layout must place every configured instance")
        for cam in cfg["cameras"]:
            for key in ("azimuth_deg", "elevation_deg", "distance_m", "focal_px"):
                bounds = cam[key]
                if (
                    len(bounds) != 2
                    or not all(math.isfinite(v) for v in bounds)
                    or bounds[0] > bounds[1]
                ):
                    raise ValueError(f"invalid {key} range for {cam['id']}")
            if min(cam["distance_m"]) <= 0 or min(cam["focal_px"]) <= 0:
                raise ValueError("distance and focal length must be positive")
            if len(cam["target"]) != 3 or len(cam["focal_px"]) != 2:
                raise ValueError("camera target and focal range are required")
    else:
        if not cases or any(set(c["objects"]) != set(ids) for c in cases):
            raise ValueError("recorded poses must match the configured instance ids")
    for edit in cfg.get("edits", []):
        kinds = [k for k in ("uniform", "choices") if k in edit]
        if len(kinds) != 1 or not edit[kinds[0]]:
            raise ValueError(f"invalid property sampling: {edit}")
    return root, cfg


def sample(cfg, cases, seed, index):
    """Per-index RNG gives identical interventions for every shard layout or retry."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    if cases is not None:
        row = dict(cases[index])
        return {
            **row,
            "edits": row.get("edits", {}),
            "layout": row.get("layout", "recorded"),
        }
    # Cycle all camera/layout combinations before repeating a stratum.
    cams, layouts = cfg["cameras"], cfg["layouts"]
    camera = cams[(index + seed) % len(cams)]
    layout = layouts[(index // len(cams) + seed) % len(layouts)]

    def uniform(bounds):
        return float(rng.uniform(*bounds))

    look = np.asarray(camera["target"]) + rng.uniform(-1, 1, 3) * camera.get(
        "target_jitter", [0, 0, 0]
    )
    az, el = np.deg2rad(
        [uniform(camera["azimuth_deg"]), uniform(camera["elevation_deg"])]
    )
    distance = uniform(camera["distance_m"])
    eye = look + distance * np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    )
    focal = uniform(camera["focal_px"])
    w, h = cfg["width"], cfg["height"]
    objects = {}
    for inst in cfg["instances"]:
        jitter = np.array(inst.get("jitter_xy_m", [0, 0])) * rng.uniform(-1, 1, 2)
        x, y, z = layout["anchors"][inst["id"]]
        objects[inst["id"]] = {
            "bottom_xyz_m": [x + jitter[0], y + jitter[1], z],
            "yaw_deg": uniform(inst.get("yaw_deg", [0, 0])),
            "visible": bool(rng.random() < inst.get("probability", 1.0)),
        }
    edits = {
        edit["id"]: uniform(edit["uniform"])
        if "uniform" in edit
        else edit["choices"][int(rng.integers(len(edit["choices"])))]
        for edit in cfg.get("edits", [])
    }
    scaled = camera.get("principal_point", [0.5, 0.5])
    return {
        "layout": layout["id"],
        "camera": {
            "id": camera["id"],
            "eye_m": eye.tolist(),
            "target_m": look.tolist(),
            "azimuth_deg": float(np.rad2deg(az)),
            "elevation_deg": float(np.rad2deg(el)),
            "K": [
                focal,
                0.0,
                w * scaled[0],
                0.0,
                focal * camera.get("fy_over_fx", 1.0),
                h * scaled[1],
                0.0,
                0.0,
                1.0,
            ],
        },
        "objects": objects,
        "edits": edits,
    }


def inspect_frame(folder, header, *, pixels=True):
    data = read(folder / "annotation.json")
    if data["campaign_key"] != header["campaign_key"] or data["index"] != int(
        folder.name
    ):
        raise ValueError(f"{folder}: wrong capture or index")
    if header.get("format", 1) >= 2:
        unsealed = {k: v for k, v in header.items() if k != "campaign_key"}
        key = hashlib.sha256(json.dumps(unsealed, sort_keys=True).encode()).hexdigest()
        if key != header["campaign_key"] or (
            folder / "record.sha256"
        ).read_text().strip() != digest(folder / "annotation.json"):
            raise ValueError(f"{folder}: capture header or frame record modified")
    expected = {"rgb.png", "classes.png", "instances.png", "range_m.npz"}
    if set(data["sha256"]) != expected or any(
        digest(folder / n) != h for n, h in data["sha256"].items()
    ):
        raise ValueError(f"{folder}: incomplete frame or content hash mismatch")
    if header.get("kind") == "sensor-derived":
        from .sensor_dataset import validate_valid_mask
        validate_valid_mask(folder, data, header)
    if not pixels:
        return data

    def plane(n):
        with Image.open(folder / n) as im:
            return np.asarray(im)

    rgb, cls, inst = (plane(n) for n in ("rgb.png", "classes.png", "instances.png"))
    with np.load(folder / "range_m.npz", allow_pickle=False) as f:
        depth = f["range_m"]
    w, h = header["resolution"]
    if (
        rgb.shape[:2] != (h, w)
        or cls.shape != (h, w)
        or inst.shape != (h, w)
        or depth.shape != (h, w)
        or depth.dtype != np.float32
        or rgb[:, :, :3].max() == rgb[:, :, :3].min()
    ):
        raise ValueError(f"{folder}: invalid image planes")
    labels = {obj["instance_id"]: obj for obj in data["objects"]}
    if len(labels) != len(data["objects"]) or any(
        not 1 <= i <= len(header["instances"]) for i in labels
    ):
        raise ValueError(f"{folder}: duplicate or invalid instance ID")
    if set(np.unique(inst)).difference({0, *labels}) or np.any(cls[inst == 0] != 0):
        raise ValueError(f"{folder}: unmapped instance/class pixels")
    for ident, obj in labels.items():
        cat = obj["category_id"]
        if (
            header["instances"][ident - 1]["class"] != obj["class"]
            or (
                obj.get("id") is not None
                and header["instances"][ident - 1]["id"] != obj["id"]
            )
            or header["classes"][obj["class"]] != cat
        ):
            raise ValueError(f"{folder}: ontology mismatch")
        yy, xx = np.where(inst == ident)
        if not len(xx):
            raise ValueError(f"{folder}: missing pixels for declared instance {ident}")
        expected_box = [
            int(xx.min()),
            int(yy.min()),
            int(xx.max() - xx.min() + 1),
            int(yy.max() - yy.min() + 1),
        ]
        mask = inst == ident
        if (
            obj["bbox_xywh"] != expected_box
            or len(xx) != obj["area"]
            or np.any(cls[mask] != cat)
            or not np.all(np.isfinite(depth[mask]) & (depth[mask] > 0))
        ):
            raise ValueError(f"{folder}: mask, rectangle or metric range disagree")
    return data


def folders(root):
    return sorted((root / "frames").glob("[0-9]" * 8))


def mask_rle(mask):
    """COCO uncompressed RLE starts with background and traverses columns first."""
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    splits = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.r_[0, splits, len(flat)]).tolist()
    if flat[0]:
        runs.insert(0, 0)
    return {"size": list(mask.shape), "counts": runs}


def write_json_array(destination, spool):
    """Stream a JSONL spool into one standard JSON array."""
    destination.write("[")
    with spool.open() as rows:
        for index, line in enumerate(rows):
            if index:
                destination.write(",")
            destination.write(line)
    destination.write("]")


def verify(root, out, start, count, coco=False):
    head = read(root / "campaign.json")
    dirs = (
        (root / "frames" / f"{n:08d}" for n in range(start, start + count))
        if count is not None
        else folders(root)
    )
    out.mkdir(parents=True, exist_ok=True)
    cams, layouts, classes = Counter(), Counter(), Counter()
    seen = set()
    n_images = n_annotations = empty = 0
    with TemporaryDirectory(prefix=".verify-", dir=out) as working:
        staging = Path(working)
        with ExitStack() as stack:
            receipts = stack.enter_context((staging / "frames.sha256.jsonl").open("w"))
            if coco:
                images = stack.enter_context((staging / "image-rows").open("w"))
                annotations = stack.enter_context(
                    (staging / "annotation-rows").open("w")
                )
                training = stack.enter_context((staging / "training.jsonl").open("w"))
            for folder in dirs:
                frame = inspect_frame(folder, head)
                i = frame["index"]
                rgb_hash = frame["sha256"]["rgb.png"]
                if rgb_hash in seen:
                    raise ValueError(f"{folder}: duplicate RGB in requested sample")
                seen.add(rgb_hash)
                cams[frame["sample"]["camera"]["id"]] += 1
                layouts[frame["sample"]["layout"]] += 1
                n_images += 1
                empty += not frame["objects"]
                receipt = {
                    "index": i,
                    "record_sha256": digest(folder / "annotation.json"),
                    "planes": frame["sha256"],
                }
                receipts.write(json.dumps(receipt, sort_keys=True) + "\n")
                if coco:
                    relative_image = os.path.relpath(folder / "rgb.png", out)
                    images.write(
                        json.dumps(
                            {
                                "id": i + 1,
                                "file_name": relative_image,
                                "width": head["resolution"][0],
                                "height": head["resolution"][1],
                            }
                        )
                        + "\n"
                    )
                    with Image.open(folder / "instances.png") as source:
                        instance_pixels = np.asarray(source)
                    row = {
                        "id": f"{head['campaign_key']}:{i}",
                        "image": relative_image,
                        "instance_mask": os.path.relpath(folder / "instances.png", out),
                        "objects": [],
                        "frame_id": i,
                        "group": frame["sample"]["layout"],
                    }
                for obj in frame["objects"]:
                    classes[obj["class"]] += 1
                    n_annotations += 1
                    if coco:
                        ann = {
                            "id": n_annotations,
                            "image_id": i + 1,
                            "category_id": obj["category_id"],
                            "bbox": obj["bbox_xywh"],
                            "area": obj["area"],
                            "iscrowd": 0,
                            "segmentation": mask_rle(
                                instance_pixels == obj["instance_id"]
                            ),
                        }
                        annotations.write(json.dumps(ann, separators=(",", ":")) + "\n")
                        row["objects"].append(
                            {
                                k: obj[k]
                                for k in ("category_id", "bbox_xywh", "instance_id")
                            }
                        )
                if coco:
                    training.write(json.dumps(row) + "\n")
        if not n_images:
            raise ValueError("empty campaign")
        outputs = ["frames.sha256.jsonl"]
        if coco:
            categories = [{"id": v, "name": k} for k, v in head["classes"].items()]
            with (staging / "coco.json").open("w") as target:
                target.write('{"images":')
                write_json_array(target, staging / "image-rows")
                target.write(',"annotations":')
                write_json_array(target, staging / "annotation-rows")
                target.write(',"categories":' + json.dumps(categories) + "}\n")
            outputs += ["coco.json", "training.jsonl"]
        result = {
            "status": "pass",
            "frames": n_images,
            "resolution": head["resolution"],
            "distinct_rgb": len(seen),
            "instances_count_by_class": dict(classes),
            "camera_samples": dict(cams),
            "layouts": dict(layouts),
            "empty_frames": empty,
            "boxes": n_annotations,
            "campaign_key": head["campaign_key"],
            "export_sha256": {n: digest(staging / n) for n in outputs},
        }
        write(staging / "audit.json", result)
        for name in [*outputs, "audit.json"]:
            (staging / name).replace(out / name)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("config", type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--shard-size", type=int, default=128)
    p.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("check")
    q.add_argument("root", type=Path)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--start", type=int, default=0)
    q.add_argument("--count", type=int, required=True)
    q.add_argument("--coco", action="store_true")
    a = cli.parse_args()
    if a.mode == "plan":
        root, cfg = load(a.config)
        cases = prescribed(root, cfg)
        if (
            a.seed < 0
            or a.frames < 1
            or a.shard_size < 1
            or (cases is not None and a.frames > len(cases))
        ):
            cli.error("invalid sample or shard size")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        write(
            a.out,
            {
                "frames": a.frames,
                "seed": a.seed,
                "config_sha256": digest(a.config),
                "status": "planned",
                "shards": [
                    {"start": i, "count": min(a.shard_size, a.frames - i)}
                    for i in range(0, a.frames, a.shard_size)
                ],
            },
        )
    else:
        if a.start < 0 or a.count < 1:
            cli.error("invalid range")
        print(json.dumps(verify(a.root, a.out, a.start, a.count, a.coco), indent=2))


if __name__ == "__main__":
    main()
