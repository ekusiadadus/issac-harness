"""Make sealed, registered sensor derivatives of verified native campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from .campaign import digest, folders, inspect_frame, read, write
from .color_sensor import ColorProfile, _camera_K, transfer


KIND = "sensor-derived"
PLANES = ("rgb.png", "classes.png", "instances.png", "range_m.npz")


def _key(header: dict) -> str:
    return hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest()


def _seed(seed: int, index: int) -> int:
    # Index based, so shards, interrupted runs and retries produce identical noise.
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])


def validate_valid_mask(folder: Path, annotation: dict, header: dict) -> None:
    """Validate the fifth plane without calling inspect_frame (safe for its hook)."""
    if header.get("kind") != KIND:
        return
    sensor = annotation.get("sensor_derivative")
    if not isinstance(sensor, dict) or sensor.get("kind") != KIND:
        raise ValueError(f"{folder}: missing sensor derivative provenance")
    path = folder / "valid.png"
    if not path.is_file() or digest(path) != sensor.get("valid_sha256"):
        raise ValueError(f"{folder}: valid mask hash mismatch")
    with Image.open(path) as image:
        valid = np.asarray(image)
    w, h = header["resolution"]
    if valid.shape != (h, w) or valid.dtype != np.uint8 or not np.isin(valid, [0, 255]).all():
        raise ValueError(f"{folder}: invalid valid mask")
    inside = valid == 255
    with Image.open(folder / "instances.png") as image:
        instances = np.asarray(image)
    with Image.open(folder / "classes.png") as image:
        classes = np.asarray(image)
    with Image.open(folder / "rgb.png") as image:
        rgb = np.asarray(image)
    with np.load(folder / "range_m.npz", allow_pickle=False) as data:
        ranges = data["range_m"]
    if (np.any(instances[~inside] != 0) or np.any(classes[~inside] != 0)
            or np.any(rgb[~inside] != 0) or not np.all(np.isnan(ranges[~inside]))):
        raise ValueError(f"{folder}: nonempty pixels outside valid field of view")
    if sensor.get("valid_pixel_fraction") != float(inside.mean()):
        raise ValueError(f"{folder}: valid coverage disagrees")
    for field in ("source_campaign_key", "source_annotation_sha256", "profile_sha256", "seed"):
        if field not in sensor:
            raise ValueError(f"{folder}: missing {field}")
    if sensor["source_campaign_key"] != header["source_campaign_key"] or sensor["profile_sha256"] != header["profile_sha256"]:
        raise ValueError(f"{folder}: sensor provenance disagrees with campaign")


def _objects(instances: np.ndarray, source: dict, header: dict) -> list[dict]:
    source_objects = {obj["instance_id"]: obj for obj in source["objects"]}
    result = []
    for ident in sorted(int(x) for x in np.unique(instances) if x):
        if ident not in source_objects:
            raise ValueError(f"unexpected instance {ident} after remap")
        yy, xx = np.where(instances == ident)
        obj = source_objects[ident]
        result.append({**{k: obj[k] for k in ("id", "class", "category_id", "instance_id") if k in obj},
                       "bbox_xywh": [int(xx.min()), int(yy.min()), int(np.ptp(xx)) + 1,
                                     int(np.ptp(yy)) + 1], "area": int(xx.size)})
    return result


def _frame(source_folder: Path, destination: Path, source_header: dict, header: dict,
           profile: ColorProfile, noise: dict | None, seed: int) -> None:
    source = inspect_frame(source_folder, source_header)
    index = source["index"]
    annotation_hash = digest(source_folder / "annotation.json")
    if destination.exists():
        existing = inspect_frame(destination, header)
        meta = existing["sensor_derivative"]
        if meta["source_annotation_sha256"] != annotation_hash or meta["seed"] != seed:
            raise ValueError(f"{destination}: resume source or seed differs")
        return
    def plane(name: str) -> np.ndarray:
        with Image.open(source_folder / name) as image:
            return np.asarray(image)
    with np.load(source_folder / "range_m.npz", allow_pickle=False) as data:
        ranges = data["range_m"]
    source_K = _camera_K(source)
    rgb, classes, instances, radial, valid = transfer(
        plane("rgb.png"), plane("classes.png"), plane("instances.png"), ranges,
        source_K, profile, seed=seed, noise=noise)
    if radial.dtype != np.float32:
        radial = radial.astype(np.float32)
    sample = json.loads(json.dumps(source["sample"]))
    sample["camera"]["K"] = profile.K.reshape(-1).tolist()
    sample["camera"]["D"] = profile.D.tolist()
    sample["camera"]["sensor_profile"] = profile.name
    sample["camera"]["model"] = "plumb_bob"
    readback = json.loads(json.dumps(source.get("readback", {})))
    if isinstance(readback.get("camera"), dict):
        readback["camera"].update({"K": profile.K.reshape(-1).tolist(), "D": profile.D.tolist(),
                                   "model": "plumb_bob"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pending-sensor-", dir=destination.parent) as temporary:
        folder = Path(temporary) / destination.name
        folder.mkdir()
        Image.fromarray(rgb).save(folder / "rgb.png")
        Image.fromarray(classes).save(folder / "classes.png")
        Image.fromarray(instances).save(folder / "instances.png")
        Image.fromarray(valid.astype(np.uint8) * 255).save(folder / "valid.png")
        np.savez_compressed(folder / "range_m.npz", range_m=radial)
        sensor = {
            "kind": KIND, "source_campaign_key": source_header["campaign_key"],
            "source_index": index, "source_annotation_sha256": annotation_hash,
            "source_frame": f"frames/{index:08d}", "source_K": source_K.reshape(-1).tolist(),
            "target_K": profile.K.reshape(-1).tolist(), "target_D": profile.D.tolist(),
            "profile_sha256": header["profile_sha256"], "profile_provenance": profile.provenance,
            "noise_fit_sha256": header.get("noise_fit_sha256"),
            "noise_provenance": noise.get("provenance", "unspecified") if noise else "disabled",
            "seed": seed, "valid_sha256": digest(folder / "valid.png"),
            "valid_pixel_fraction": float(valid.mean()),
            "range_unit": "radial metres, nearest registered source pixel",
            "emission": "derived from native capture; no Isaac sensor emitted these planes",
        }
        write(folder / "annotation.json", {
            "campaign_key": header["campaign_key"], "index": index,
            "sample": sample, "readback": readback, "objects": _objects(instances, source, header),
            "sha256": {name: digest(folder / name) for name in PLANES},
            "sensor_derivative": sensor,
        })
        (folder / "record.sha256").write_text(digest(folder / "annotation.json") + "\n")
        inspect_frame(folder, header)
        try:
            folder.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            existing = inspect_frame(destination, header)
            if existing != read(folder / "annotation.json"):
                raise ValueError(f"{destination}: conflicting concurrent derivative")


def derive(source: Path, profile_path: Path, output: Path, seed: int, *,
           start: int = 0, count: int | None = None, noise_path: Path | None = None) -> dict:
    source, profile_path, output = source.resolve(), profile_path.resolve(), output.resolve()
    if seed < 0 or start < 0 or count is not None and count < 1:
        raise ValueError("seed/start must be nonnegative and count positive")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and derivative directories must be separate")
    source_header = read(source / "campaign.json")
    if source_header.get("format", 1) < 2 or source_header.get("kind") == KIND:
        raise ValueError("derivation needs a sealed undistorted source campaign")
    profile = ColorProfile.from_json(profile_path)
    noise = read(noise_path) if noise_path else None
    header = {**source_header, "format": 2, "kind": KIND,
              "source_campaign_key": source_header["campaign_key"],
              "source_campaign_sha256": digest(source / "campaign.json"),
              "profile_sha256": digest(profile_path), "profile_name": profile.name,
              "derivation_code": {name: digest(Path(__file__).with_name(name))
                                  for name in ("color_sensor.py", "sensor_dataset.py")},
              "profile_provenance": profile.provenance,
              "target_K": profile.K.reshape(-1).tolist(), "target_D": profile.D.tolist(),
              "noise_fit_sha256": digest(noise_path) if noise_path else None,
              "seed": seed, "resolution": [profile.width, profile.height],
              "depth": "derived radial metres; outside valid field of view is NaN"}
    header.pop("campaign_key", None)
    header["campaign_key"] = _key(header)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "campaign.json"
    if not target.exists():
        with TemporaryDirectory(prefix=".pending-header-", dir=output) as temporary:
            pending = Path(temporary) / "campaign.json"
            write(pending, header)
            try:
                os.link(pending, target)
            except FileExistsError:
                pass
    if read(target) != header:
        raise ValueError("existing derivative campaign has different source, profile, code, noise or seed")
    selected = (source / "frames" / f"{i:08d}" for i in range(start, start + count)) if count is not None else (f for f in folders(source) if int(f.name) >= start)
    total = 0
    for folder in selected:
        _frame(folder, output / "frames" / folder.name, source_header, header,
               profile, noise, _seed(seed, int(folder.name)))
        total += 1
    if not total:
        raise ValueError("no source frames selected")
    return {"kind": KIND, "frames": total, "campaign_key": header["campaign_key"], "output": str(output)}


def main(argv: list[str] | None = None) -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("--profile", required=True, type=Path)
    cli.add_argument("--out", required=True, type=Path)
    cli.add_argument("--seed", required=True, type=int)
    cli.add_argument("--start", type=int, default=0)
    cli.add_argument("--count", type=int)
    cli.add_argument("--noise-fit", type=Path)
    args = cli.parse_args(argv)
    print(json.dumps(derive(args.source, args.profile, args.out, args.seed,
                            start=args.start, count=args.count, noise_path=args.noise_fit), indent=2))


if __name__ == "__main__":
    main()
