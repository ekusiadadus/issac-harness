"""Score camera-registered real and native Isaac campaign frames.

Run: python -m harness.paired CAMPAIGN --references ROOT --config quality.toml --out OUT
"""

import argparse
import json
from pathlib import Path
import sys
import tomllib

import numpy as np
from PIL import Image

from harness.campaign import digest, folders, inspect_frame, read, write


METRICS = ("iou_macro_min", "z_median_abs_m_max", "rgb_mae_max")


def radial_to_z(radial, K):
    """Convert Euclidean camera range to optical-axis Z at pixel centres."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("invalid effective camera intrinsics")
    h, w = radial.shape
    y, x = np.indices((h, w), dtype=np.float64)
    xy1 = np.stack((x + 0.5, y + 0.5, np.ones_like(x)), axis=-1)
    rays = xy1 @ np.linalg.inv(K).T
    return radial.astype(np.float64) * rays[..., 2] / np.linalg.norm(rays, axis=-1)


def interior(mask, radius=2):
    if radius == 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.pad(mask, radius, constant_values=False)
    result = np.ones_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result &= padded[dy:dy + h, dx:dx + w]
    return result


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def plane(path):
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim != 2:
        raise ValueError(f"expected one channel: {path}")
    return a


def thresholds(config):
    with config.open("rb") as stream:
        cfg = tomllib.load(stream)
    scenes = cfg.get("scenes", {})
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError("quality config needs explicit [scenes.NAME] thresholds")
    for name, limits in scenes.items():
        if set(limits) != set(METRICS) or any(not np.isfinite(limits[k]) for k in METRICS):
            raise ValueError(f"{name}: all three finite metric limits are required")
        if not (0 < limits["iou_macro_min"] <= 1 and limits["z_median_abs_m_max"] > 0 and 0 < limits["rgb_mae_max"] <= 1):
            raise ValueError(f"{name}: invalid metric limits")
    return scenes


def score_frame(folder, head, references, *, erosion=2, strip_dir=None):
    if head.get("kind") == "sensor-derived":
        raise ValueError("registered comparison requires undistorted images and matching pinhole intrinsics")
    ann = inspect_frame(folder, head)
    sample = ann["sample"]
    refname = sample.get("reference")
    if not refname or "depth_scale_mm" not in sample:
        raise ValueError(f"{folder}: recorded reference and depth_scale_mm required")
    reference = references / refname
    if not reference.is_dir():
        raise ValueError(f"{folder}: missing reference {reference}")
    real_hashes = {name: digest(reference / name) for name in ("rgb.png", "classes.png", "z-depth.png")}
    if sample.get("rgb_sha256") is not None and sample["rgb_sha256"] != real_hashes["rgb.png"]:
        raise ValueError(f"{folder}: recorded real RGB hash mismatch")
    real_rgb = rgb(reference / "rgb.png")
    sim_rgb = rgb(folder / "rgb.png")
    real_cls = plane(reference / "classes.png")
    sim_cls = plane(folder / "classes.png")
    real_raw = plane(reference / "z-depth.png")
    with np.load(folder / "range_m.npz", allow_pickle=False) as data:
        sim_range = data["range_m"]
    shape = sim_cls.shape
    if any(a.shape[:2] != shape for a in (real_rgb, sim_rgb, real_cls, real_raw, sim_range)) or sim_rgb.shape[2] != 3:
        raise ValueError(f"{folder}: real and simulated planes differ in resolution")
    scale = float(sample["depth_scale_mm"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{folder}: invalid real depth scale")
    real_z = real_raw.astype(np.float64) * scale * 0.001
    effective_K = ann.get("readback", {}).get("camera", {}).get("K")
    if effective_K is None:
        raise ValueError(f"{folder}: effective camera K missing")
    sim_z = radial_to_z(sim_range, effective_K)
    ontology = set(head["classes"].values())
    unexpected = (set(np.unique(real_cls)) | set(np.unique(sim_cls))) - ontology - {0}
    if unexpected:
        raise ValueError(f"{folder}: unsupported class IDs {sorted(unexpected)}")
    classes = {}
    z_errors, rgb_errors = [], []
    for name, ident in head["classes"].items():
        a, b = real_cls == ident, sim_cls == ident
        union = int(np.count_nonzero(a | b))
        if not union:
            continue
        intersection = int(np.count_nonzero(a & b))
        matched = interior(a, erosion) & interior(b, erosion)
        valid = matched & np.isfinite(real_z) & (real_z > 0) & np.isfinite(sim_z) & (sim_z > 0)
        dz = np.abs(real_z[valid] - sim_z[valid])
        drgb = np.abs(real_rgb[matched].astype(np.float64) - sim_rgb[matched].astype(np.float64)) / 255
        z_errors.extend(dz.tolist())
        rgb_errors.extend(drgb.ravel().tolist())
        classes[name] = {
            "id": ident,
            "real_pixels": int(a.sum()), "sim_pixels": int(b.sum()),
            "intersection_pixels": intersection, "union_pixels": union,
            "iou": intersection / union,
            "matched_interior_pixels": int(matched.sum()),
            "valid_depth_pixels": int(valid.sum()),
            "z_median_abs_m": float(np.median(dz)) if dz.size else None,
            "rgb_mae": float(np.mean(drgb)) if drgb.size else None,
        }
    if strip_dir is not None:
        strip_dir.mkdir(parents=True, exist_ok=True)
        # Red/cyan contours expose either mask even where the two masks overlap.
        overlay = sim_rgb.copy()
        real_target, sim_target = real_cls != 0, sim_cls != 0
        real_edge = real_target & ~interior(real_target, 1)
        sim_edge = sim_target & ~interior(sim_target, 1)
        overlay[real_edge] = [255, 32, 32]
        overlay[sim_edge] = [0, 255, 255]
        overlay[real_edge & sim_edge] = [255, 255, 0]
        Image.fromarray(np.concatenate((real_rgb, sim_rgb, overlay), axis=1)).save(strip_dir / f"{folder.name}.png")
    return {
        "frame": int(folder.name), "scene": sample["layout"], "reference": refname,
        "real_sha256": real_hashes,
        "classes": classes, "iou_macro": float(np.mean([v["iou"] for v in classes.values()])) if classes else None,
        "z_median_abs_m": float(np.median(z_errors)) if z_errors else None,
        "rgb_mae": float(np.mean(rgb_errors)) if rgb_errors else None,
        "valid_depth_pixels": len(z_errors), "matched_rgb_channels": len(rgb_errors),
        "depth_abs_errors_m": z_errors, "rgb_abs_errors_normalized": rgb_errors,
    }


def evaluate(campaign, references, config, out, *, erosion=2):
    limits = thresholds(config)
    head = read(campaign / "campaign.json")
    records, failures = [], []
    for folder in folders(campaign):
        try:
            records.append(score_frame(folder, head, references, erosion=erosion, strip_dir=out / "strips"))
        except (ValueError, OSError, KeyError, IndexError) as exc:
            failures.append({"frame": folder.name, "reason": str(exc)})
    scenes = {}
    for name in sorted(set(limits) | {r["scene"] for r in records}):
        rows = [r for r in records if r["scene"] == name]
        class_ious = [c["iou"] for r in rows for c in r["classes"].values()]
        dz = [v for r in rows for v in r.pop("depth_abs_errors_m")]
        drgb = [v for r in rows for v in r.pop("rgb_abs_errors_normalized")]
        metrics = {
            "iou_macro": float(np.mean(class_ious)) if class_ious else None,
            "z_median_abs_m": float(np.median(dz)) if dz else None,
            "rgb_mae": float(np.mean(drgb)) if drgb else None,
        }
        bound = limits.get(name)
        passed = bool(bound and rows and all(v is not None for v in metrics.values())
                      and metrics["iou_macro"] >= bound["iou_macro_min"]
                      and metrics["z_median_abs_m"] <= bound["z_median_abs_m_max"]
                      and metrics["rgb_mae"] <= bound["rgb_mae_max"])
        scenes[name] = {"frames": len(rows), "class_samples": len(class_ious), "valid_depth_pixels": len(dz),
                        "matched_rgb_channels": len(drgb), "metrics": metrics, "thresholds": bound,
                        "status": "pass" if passed else "fail"}
    if not records and not failures:
        failures.append({"frame": None, "reason": "empty campaign"})
    report = {"status": "pass" if not failures and all(v["status"] == "pass" for v in scenes.values()) else "fail",
              "campaign_key": head["campaign_key"], "config": str(config), "config_sha256": digest(config),
              "references": str(references),
              "erosion_px": erosion, "scenes": scenes, "frames": records, "failures": failures}
    out.mkdir(parents=True, exist_ok=True)
    write(out / "quality.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--erosion-px", type=int, default=2)
    a = p.parse_args()
    if a.erosion_px < 0:
        p.error("erosion must be nonnegative")
    try:
        result = evaluate(a.campaign, a.references, a.config, a.out, erosion=a.erosion_px)
    except (ValueError, OSError, KeyError) as exc:
        p.error(str(exc))
    print(json.dumps({"status": result["status"], "scenes": result["scenes"], "failures": result["failures"]}, indent=2))
    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
