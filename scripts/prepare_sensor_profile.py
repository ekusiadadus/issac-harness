"""Run the TUM mono sensor example on a saved renderer frame."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from harness.sensor_model import apply_sensor, inverse_fov_map, load_tum_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="RTX PNG or campaign frame directory")
    parser.add_argument("--profile", type=Path, default=Path("examples/sensors/tum_mono_vo_sequence_01"))
    parser.add_argument("--output", type=Path, default=Path("evidence/sensor-calibration/example"))
    parser.add_argument("--exposure-ms", type=float, default=3.6060349941)
    parser.add_argument("--reference-exposure-ms", type=float, default=3.6060349941)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    profile = load_tum_profile(args.profile)
    campaign = args.image.is_dir()
    source_image = args.image / "rgb.png" if campaign else args.image
    with Image.open(source_image) as im:
        original_size = im.size
        rgb = np.asarray(im.convert("RGB"))
    source_K = None
    input_annotation = None
    if campaign:
        input_annotation = json.loads((args.image / "annotation.json").read_text())
        source_K = np.asarray(input_annotation["sample"]["camera"]["K"], dtype=float).reshape(3, 3)
        with Image.open(args.image / "classes.png") as im:
            ids = np.asarray(im)
        with np.load(args.image / "range_m.npz", allow_pickle=False) as data:
            distance = data["range_m"]
    else:
        rgb = np.asarray(Image.fromarray(rgb).resize((profile.width, profile.height), Image.Resampling.BICUBIC))
        ids = np.zeros(rgb.shape[:2], dtype=np.uint32)
        distance = np.full(rgb.shape[:2], np.nan, dtype=np.float32)
    sensor_rgb, sensor_ids, sensor_distance, metadata = apply_sensor(rgb, ids, distance, profile,
        exposure_ms=args.exposure_ms, reference_exposure_ms=args.reference_exposure_ms,
        seed=args.seed, source_K=source_K)
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sensor_rgb).save(args.output / "rgb.png")
    if campaign:
        Image.fromarray(sensor_ids).save(args.output / "classes.png")
        np.savez_compressed(args.output / "range_m.npz", range_m=sensor_distance)
        with Image.open(args.image / "instances.png") as im:
            instances = np.asarray(im)
        if instances.shape != ids.shape:
            raise ValueError("instances.png must match classes.png")
        sx, sy = inverse_fov_map(profile, source_K)
        h, w = ids.shape
        valid = (sx >= 0) & (sy >= 0) & (sx <= w - 1) & (sy <= h - 1)
        ix = np.rint(np.where(valid, sx, 0)).clip(0, w - 1).astype(np.int32)
        iy = np.rint(np.where(valid, sy, 0)).clip(0, h - 1).astype(np.int32)
        sensor_instances = np.where(valid, instances[iy, ix], 0).astype(instances.dtype)
        Image.fromarray(sensor_instances).save(args.output / "instances.png")
    metadata.update({
        "input": str(args.image),
        "input_sha256": hashlib.sha256(source_image.read_bytes()).hexdigest(),
        "input_size": original_size,
        "sensor_size": [profile.width, profile.height],
        "input_mode": "campaign_registered" if campaign else "RGB_visual_demo_resized",
        "semantic_and_distance": "warped from campaign" if campaign else "unavailable",
        "output_mean_code": float(sensor_rgb[:, :, 0].mean()),
        "output_std_code": float(sensor_rgb[:, :, 0].std()),
    })
    (args.output / "annotation.json").write_text(json.dumps({
        "sensor_model": metadata, "source_annotation": input_annotation,
    }, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
