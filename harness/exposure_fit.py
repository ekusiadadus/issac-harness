"""Freeze one observed native RTX exposure bracket using registered real RGB.

Example: python -m harness.exposure_fit CAPTURE --references REAL \
    --calibration-ref A --calibration-ref B --gain-field edits.light_gain --out freeze.json
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from harness.campaign import digest, folders, inspect_frame, read, write
from harness.paired import interior, plane, rgb


def field(record, dotted):
    value = record
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing sample.{dotted}")
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"sample.{dotted} must be a finite numeric scalar")
    return float(value)


def aligned_rgb_error(folder, head, reference, *, erosion=2):
    if head.get("kind") == "sensor-derived":
        raise ValueError("radiometric calibration requires undistorted, camera-registered RGB")
    actual = rgb(folder / "rgb.png")
    desired = rgb(reference / "rgb.png")
    sim_cls = plane(folder / "classes.png")
    real_cls = plane(reference / "classes.png")
    if actual.shape != desired.shape or actual.shape[:2] != sim_cls.shape or sim_cls.shape != real_cls.shape:
        raise ValueError(f"{folder}: RGB or class plane resolution mismatch")
    allowed = set(head["classes"].values()) | {0}
    extra = (set(np.unique(real_cls)) | set(np.unique(sim_cls))) - allowed
    if extra:
        raise ValueError(f"{folder}: unsupported class IDs {sorted(extra)}")
    matched = np.zeros(sim_cls.shape, bool)
    class_counts = {}
    for name, ident in head["classes"].items():
        area = interior(real_cls == ident, erosion) & interior(sim_cls == ident, erosion)
        if area.any():
            matched |= area
            class_counts[name] = int(area.sum())
    if not matched.any():
        raise ValueError(f"{folder}: no aligned target interiors")
    sim_values = actual[matched]
    real_values = desired[matched]
    return {
        "matched_pixels": int(matched.sum()),
        "matched_channels": int(sim_values.size),
        "classes": class_counts,
        "srgb_mae": float(np.abs(sim_values.astype(np.float64) - real_values.astype(np.float64)).mean() / 255),
        "sim_clipped_channel_fraction": float(np.mean(sim_values >= 250)),
        "real_clipped_channel_fraction": float(np.mean(real_values >= 250)),
    }


def fit(campaign, references, calibration_refs, output, *, gain_field="edits.light_gain", erosion=2):
    """Select only among observed bracket values; never alter captured pixels."""
    refs = list(calibration_refs)
    if not refs or len(set(refs)) != len(refs) or any(not v or "/" in v or v in (".", "..") for v in refs):
        raise ValueError("provide distinct calibration reference directory names")
    if erosion < 0:
        raise ValueError("erosion must be nonnegative")
    head = read(campaign / "campaign.json")
    by_gain = {}
    for folder in folders(campaign):
        sample = read(folder / "annotation.json")["sample"]
        refname = sample.get("reference")
        if refname not in refs:
            continue  # Held-out captures are neither inspected nor scored.
        ann = inspect_frame(folder, head)
        gain = field(sample, gain_field)
        reference = references / refname
        if not reference.is_dir():
            raise ValueError(f"missing calibration reference {reference}")
        group = by_gain.setdefault(gain, {})
        if refname in group:
            raise ValueError(f"duplicate calibration reference {refname} at {gain_field}={gain}")
        stats = aligned_rgb_error(folder, head, reference, erosion=erosion)
        group[refname] = {
            "frame": ann["index"], "reference": refname, "gain": gain,
            **stats,
            "sha256": {
                "real_rgb": digest(reference / "rgb.png"),
                "real_classes": digest(reference / "classes.png"),
                "sim_rgb": ann["sha256"]["rgb.png"],
                "sim_classes": ann["sha256"]["classes.png"],
                "annotation": digest(folder / "annotation.json"),
            },
        }
    if not by_gain:
        raise ValueError("no calibration bracket frames found")
    for gain, items in by_gain.items():
        missing = set(refs) - set(items)
        if missing:
            raise ValueError(f"incomplete bracket {gain_field}={gain}: missing {sorted(missing)}")
    candidates = []
    for gain, items in sorted(by_gain.items()):
        rows = [items[r] for r in refs]
        channels = sum(r["matched_channels"] for r in rows)
        candidates.append({
            "gain": gain,
            "mean_reference_srgb_mae": float(np.mean([r["srgb_mae"] for r in rows])),
            "pooled_srgb_mae": float(sum(r["srgb_mae"] * r["matched_channels"] for r in rows) / channels),
            "matched_pixels": sum(r["matched_pixels"] for r in rows),
            "matched_channels": channels,
            "sim_clipped_channel_fraction": float(sum(r["sim_clipped_channel_fraction"] * r["matched_channels"] for r in rows) / channels),
            "real_clipped_channel_fraction": float(sum(r["real_clipped_channel_fraction"] * r["matched_channels"] for r in rows) / channels),
            "per_reference": rows,
        })
    # Equal reference weight; deterministic numeric gain tie-break.
    best = min(candidates, key=lambda c: (c["mean_reference_srgb_mae"], c["gain"]))
    artifact = {
        "format": 1, "method": "observed_native_bracket_equal_reference_mean_srgb_mae",
        "campaign_key": head["campaign_key"],
        "campaign_header_sha256": digest(campaign / "campaign.json"),
        "gain_field": gain_field, "chosen_gain": best["gain"],
        "calibration_references": refs, "erosion_px": erosion,
        "candidate_count": len(candidates), "candidates": candidates,
        "held_out_policy": "Only explicitly named calibration references were scored; no held-out reference was fitted.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, artifact)
    return artifact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--calibration-ref", action="append", required=True)
    p.add_argument("--gain-field", default="edits.light_gain", help="Dot path within sample, e.g. edits.light_gain or edits.ev")
    p.add_argument("--erosion-px", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    try:
        result = fit(a.campaign, a.references, a.calibration_ref, a.out, gain_field=a.gain_field, erosion=a.erosion_px)
    except (ValueError, OSError, KeyError, IndexError) as exc:
        p.error(str(exc))
    print(json.dumps({"gain_field": result["gain_field"], "chosen_gain": result["chosen_gain"], "output": str(a.out)}))
    sys.exit(0)


if __name__ == "__main__":
    main()
