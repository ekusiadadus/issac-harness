"""Check that a fixed view's labels survive a declared appearance edit."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import campaign


def _plane(folder, name):
    with Image.open(folder / name) as image:
        return np.asarray(image)


def _without_edits(record, allowed):
    if not isinstance(record, dict) or not isinstance(record.get("edits"), dict):
        raise ValueError("missing saved appearance edits")
    return {**record, "edits": {k: v for k, v in record["edits"].items() if k not in allowed}}


def _validate_native_audit(record, header):
    """Check each frame's raw annotator mapping without assuming stable raw IDs."""
    audit = record.get("native_audit")
    if not isinstance(audit, dict) or not isinstance(audit.get("path_ids"), dict) or not isinstance(audit.get("semantic_ids"), dict):
        raise ValueError("missing native instance/semantic audit")
    if audit.get("instance_path_pixels_different") != 0:
        raise ValueError("native instance path and semantic masks disagree")
    paths = [spec["prim"] for spec in header["instances"]]
    tokens = {f"capture{i:05d}" for i in range(1, len(paths) + 1)}
    path_owners = set()
    for path in audit["path_ids"].values():
        owners = [owner for owner in paths if path == owner or path.startswith(owner + "/")]
        if len(owners) > 1:
            raise ValueError(f"native path has ambiguous target ownership: {path}")
        path_owners.update(owners)
    observed = set()
    for labels in audit["semantic_ids"].values():
        if not isinstance(labels, dict):
            raise ValueError("invalid native semantic labels")
        hits = set(str(labels.get("class", "")).replace(",", " ").split()) & tokens
        if len(hits) > 1:
            raise ValueError("native semantic label crosses registered targets")
        observed.update(hits)
    for obj in record["objects"]:
        if paths[obj["instance_id"] - 1] not in path_owners:
            raise ValueError(f"native path mapping missing target {obj['id']}")
        if f"capture{obj['instance_id']:05d}" not in observed:
            raise ValueError(f"native semantic mapping missing target {obj['id']}")


def compare(directory, first, second, *, edits, min_rgb_pixels=0, target=None,
            range_atol_m=0.0, rgb_delta=0.0, min_target_rgb_pixels=0):
    """Compare two sealed native frames from one campaign; raise on invalid controls."""
    directory = Path(directory)
    if first == second:
        raise ValueError("choose two different frame indices")
    allowed = set(edits)
    if not allowed or any(not key or "." in key for key in allowed):
        raise ValueError("declare at least one simple appearance edit with --edit")
    if min_rgb_pixels < 0:
        raise ValueError("--min-rgb-pixels must be nonnegative")
    if min_target_rgb_pixels < 0:
        raise ValueError("--min-target-rgb-pixels must be nonnegative")
    if not math.isfinite(range_atol_m) or range_atol_m < 0:
        raise ValueError("--range-atol-m must be finite and nonnegative")
    if not math.isfinite(rgb_delta) or not 0 <= rgb_delta <= 1:
        raise ValueError("--rgb-delta must be finite and between 0 and 1")
    if min_target_rgb_pixels and target is None:
        raise ValueError("--min-target-rgb-pixels requires --target")
    header = campaign.read(directory / "campaign.json")
    if header.get("format", 1) < 2 or not isinstance(header.get("campaign_key"), str):
        raise ValueError("a sealed format-2 native campaign is required")
    folders = [directory / "frames" / f"{index:08d}" for index in (first, second)]
    records = [campaign.inspect_frame(folder, header) for folder in folders]
    a, b = records
    for record in records:
        _validate_native_audit(record, header)
    for field in ("layout", "camera", "objects", "edits"):
        if field not in a["sample"] or field not in b["sample"]:
            raise ValueError(f"missing sampled {field}")
    for field in ("camera", "objects", "edits"):
        if field not in a.get("readback", {}) or field not in b.get("readback", {}):
            raise ValueError(f"missing native readback {field}")
    for key in allowed:
        if key not in a["sample"]["edits"] or key not in b["sample"]["edits"]:
            raise ValueError(f"declared edit {key!r} missing from sample")
        if key not in a["readback"]["edits"] or key not in b["readback"]["edits"]:
            raise ValueError(f"declared edit {key!r} missing from native readback")
    if _without_edits(a["sample"], allowed) != _without_edits(b["sample"], allowed):
        raise ValueError("sampled camera, object pose/visibility, layout, or undeclared edit changed")
    if _without_edits(a["readback"], allowed) != _without_edits(b["readback"], allowed):
        raise ValueError("native camera, object pose/visibility, or undeclared edit changed")
    changed = sorted(key for key in allowed if a["sample"]["edits"][key] != b["sample"]["edits"][key])
    if not changed:
        raise ValueError("declared appearance edit did not change")
    for key in allowed:
        native_changed = a["readback"]["edits"][key] != b["readback"]["edits"][key]
        if native_changed != (key in changed):
            raise ValueError(f"declared edit {key!r} disagrees with native readback")
    ids_a = {(obj["id"], obj["instance_id"], obj["category_id"], obj["class"]) for obj in a["objects"]}
    ids_b = {(obj["id"], obj["instance_id"], obj["category_id"], obj["class"]) for obj in b["objects"]}
    if ids_a != ids_b:
        raise ValueError("visible object identities or class IDs changed")
    rgb_a, rgb_b = (_plane(folder, "rgb.png") for folder in folders)
    inst_a, inst_b = (_plane(folder, "instances.png") for folder in folders)
    cls_a, cls_b = (_plane(folder, "classes.png") for folder in folders)
    if rgb_a.shape != rgb_b.shape or inst_a.shape != inst_b.shape or cls_a.shape != cls_b.shape:
        raise ValueError("image shape mismatch")
    # Mean per-channel sRGB absolute difference, normalized to [0, 1].
    # A nonzero threshold filters sparse Monte Carlo/denoiser fluctuations.
    rgb_mad = np.mean(np.abs(rgb_a.astype(np.float32) - rgb_b.astype(np.float32)), axis=2) / 255.0
    rgb_changed = rgb_mad > rgb_delta
    rgb_count = int(np.count_nonzero(rgb_changed))
    inst_changed = inst_a != inst_b
    cls_changed = cls_a != cls_b
    with np.load(folders[0] / "range_m.npz", allow_pickle=False) as source, np.load(
        folders[1] / "range_m.npz", allow_pickle=False
    ) as comparison:
        depth_a, depth_b = source["range_m"], comparison["range_m"]
    # A real decal can protrude slightly from its owner. Protect the unchanged
    # mask area and let the caller bound the permitted range displacement there.
    protected = ~inst_changed & ~cls_changed
    depth_changed = ~np.isclose(depth_a, depth_b, rtol=0, atol=0, equal_nan=True)
    depth_violation = ~np.isclose(depth_a, depth_b, rtol=0, atol=range_atol_m, equal_nan=True)
    depth_elsewhere = int(np.count_nonzero(depth_violation & protected))
    finite_protected = protected & np.isfinite(depth_a) & np.isfinite(depth_b)
    nonfinite_mismatch = bool(np.any(depth_changed & protected & ~finite_protected))
    max_protected_delta = ("inf" if nonfinite_mismatch else
                           float(np.max(np.abs(depth_a[finite_protected] - depth_b[finite_protected])))
                           if np.any(finite_protected) else 0.0)
    result = {
        "campaign": str(directory), "frames": [first, second], "edits_changed": changed,
        "rgb_changed_pixels": rgb_count,
        "min_rgb_pixels": min_rgb_pixels,
        "rgb_floor_met": rgb_count >= min_rgb_pixels,
        "rgb_delta": rgb_delta,
        "instance_changed_pixels": int(np.count_nonzero(inst_changed)),
        "class_changed_pixels": int(np.count_nonzero(cls_changed)),
        "range_changed_pixels": int(np.count_nonzero(depth_changed)),
        "range_atol_m": range_atol_m,
        "protected_range_max_delta_m": max_protected_delta,
        "protected_range_violating_pixels": depth_elsewhere,
        "instance_transitions": {}, "class_transitions": {},
    }
    for name, before, after, changed_mask in (
        ("instance", inst_a, inst_b, inst_changed), ("class", cls_a, cls_b, cls_changed)
    ):
        if np.any(changed_mask):
            pairs, counts = np.unique(np.stack((before[changed_mask], after[changed_mask]), axis=1), axis=0, return_counts=True)
            result[f"{name}_transitions"] = {f"{int(x)}->{int(y)}": int(n) for (x, y), n in zip(pairs, counts)}
    if target is not None:
        matches = [obj for obj in a["objects"] if obj["id"] == target]
        if len(matches) != 1:
            raise ValueError(f"target {target!r} is not a visible instance")
        ident, category = matches[0]["instance_id"], matches[0]["category_id"]
        result["target"] = target
        target_intersection = (inst_a == ident) & (inst_b == ident)
        result["target_intersection_pixels"] = int(np.count_nonzero(target_intersection))
        result["target_rgb_changed_pixels"] = int(np.count_nonzero(rgb_changed & target_intersection))
        result["min_target_rgb_pixels"] = min_target_rgb_pixels
        result["target_rgb_floor_met"] = result["target_rgb_changed_pixels"] >= min_target_rgb_pixels
        result["target_instance_lost_pixels"] = int(np.count_nonzero((inst_a == ident) & (inst_b != ident)))
        result["target_class_lost_pixels"] = int(np.count_nonzero((cls_a == category) & (cls_b != category)))
    result["passed"] = (result["instance_changed_pixels"] == 0 and
                        result["class_changed_pixels"] == 0 and depth_elsewhere == 0 and
                        result["rgb_floor_met"] and result.get("target_rgb_floor_met", True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, help="directory containing campaign.json and frames/")
    parser.add_argument("first", type=int)
    parser.add_argument("second", type=int)
    parser.add_argument("--edit", action="append", required=True, help="appearance edit allowed to change")
    parser.add_argument("--target", help="stable instance ID to count lost pixels for")
    parser.add_argument("--min-rgb-pixels", type=int, default=0)
    parser.add_argument("--rgb-delta", type=float, default=0.0,
                        help="minimum mean absolute per-channel sRGB change, normalized to [0, 1]")
    parser.add_argument("--min-target-rgb-pixels", type=int, default=0,
                        help="required changed RGB pixels inside stable target-mask intersection")
    parser.add_argument("--range-atol-m", type=float, default=0.0,
                        help="maximum allowed range change in unchanged mask pixels, metres")
    parser.add_argument("--out", type=Path, help="write JSON result, including failures")
    args = parser.parse_args(argv)
    try:
        result = compare(args.campaign, args.first, args.second, edits=args.edit,
                         min_rgb_pixels=args.min_rgb_pixels, target=args.target,
                         range_atol_m=args.range_atol_m, rgb_delta=args.rgb_delta,
                         min_target_rgb_pixels=args.min_target_rgb_pixels)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        result = {"passed": False, "error": str(exc)}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
