"""Compare independent RTX shards and gate a dataset against a project's measured targets."""

import argparse
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image

from .campaign import digest, folders, inspect_frame, read, verify, write
from .label_invariance import compare as compare_appearance


def replay(original, repeat, start, count, max_rgb_rmse, min_mask_iou, max_range_error):
    if original.resolve() == repeat.resolve() or count < 1 or start < 0:
        raise ValueError("use two distinct campaigns and a positive frame count")
    header = read(original / "campaign.json")
    if header != read(repeat / "campaign.json"):
        raise ValueError(
            "independent captures must have identical source, sensor, seed and renderer conditions"
        )
    if not 0 <= max_rgb_rmse <= 1 or not 0 <= min_mask_iou <= 1 or max_range_error < 0:
        raise ValueError("invalid replay limits")
    result = {
        "campaign_key": header["campaign_key"],
        "frames": [],
        "status": "pass",
        "thresholds": {
            "max_rgb_rmse": max_rgb_rmse,
            "min_instance_iou": min_mask_iou,
            "max_range_mae_m": max_range_error,
        },
    }
    for idx in range(start, start + count):
        aa, bb = [root / "frames" / f"{idx:08d}" for root in (original, repeat)]
        a, b = [inspect_frame(f, header) for f in (aa, bb)]
        if a["sample"] != b["sample"] or a["readback"] != b["readback"]:
            raise ValueError(f"scene interventions differ at {idx}")
        values = []
        for name in ("rgb.png", "instances.png"):
            pair = []
            for f in (aa, bb):
                with Image.open(f / name) as im:
                    pair.append(np.array(im))
            values.append(pair)
        (rgb1, rgb2), (m1, m2) = values
        rmse = float(np.mean(((rgb1.astype(np.float32) - rgb2) / 255.0) ** 2) ** 0.5)
        labels = set(np.unique(m1)) | set(np.unique(m2))
        labels.discard(0)
        minimum = min(
            (
                float(np.sum((m1 == n) & (m2 == n)) / np.sum((m1 == n) | (m2 == n)))
                for n in labels
            ),
            default=1.0,
        )
        with np.load(aa / "range_m.npz") as x, np.load(bb / "range_m.npz") as y:
            p, q = x["range_m"], y["range_m"]
        common = np.isfinite(p) & np.isfinite(q) & (m1 == m2) & (m1 != 0)
        depth = float(np.mean(np.abs(p[common] - q[common]))) if common.any() else None
        passed = (
            rmse <= max_rgb_rmse
            and minimum >= min_mask_iou
            and (not labels or depth is not None and depth <= max_range_error)
        )
        if not passed:
            result["status"] = "fail"
        result["frames"].append(
            {
                "index": idx,
                "rgb_rmse_srgb": rmse,
                "min_instance_iou": minimum,
                "range_mae_m": depth,
                "pass": passed,
                "rgb_sha256": [a["sha256"]["rgb.png"], b["sha256"]["rgb.png"]],
            }
        )
    return result


def registered_inputs(policy_dir, spec, report):
    """Bind a separate registration campaign to the project's chosen photographs and limits."""
    if set(spec) != {"poses", "quality_policy"}:
        raise ValueError("registered_inputs requires poses and quality_policy paths")
    pose_path = policy_dir / spec["poses"]
    threshold_path = policy_dir / spec["quality_policy"]
    poses = read(pose_path)
    expected = {(r["layout"], r["reference"], r["rgb_sha256"]) for r in poses}
    frames = report.get("frames", [])
    observed = {
        (r.get("scene"), r.get("reference"), r.get("real_sha256", {}).get("rgb.png"))
        for r in frames
    }
    hashes = {"poses_sha256": digest(pose_path), "quality_policy_sha256": digest(threshold_path)}
    valid = (
        bool(expected) and len(expected) == len(poses) == len(frames)
        and expected == observed and report.get("config_sha256") == hashes["quality_policy_sha256"]
    )
    return {**hashes, "expected_views": len(poses), "measured_views": len(frames)}, valid


def release(
    root, policy, out, paired=None, transfer=None, reviews=None, replay_report=None
):
    rules = tomllib.loads(policy.read_text())
    allowed = {
        "minimum_frames",
        "min_visible_per_class",
        "minimum_cameras",
        "minimum_layouts",
        "required_edits",
        "min_distinct_texture_sets",
        "min_macro_ap50_real",
        "real_classes",
        "require_registered_quality",
        "registered_inputs",
        "required_visual_frames",
        "require_replay",
        "appearance_pairs",
    }
    unknown = set(rules) - allowed
    if unknown:
        raise ValueError(f"unrecognized policy fields: {sorted(unknown)}")
    if not rules:
        raise ValueError("empty release policy")
    if any(
        type(rules[k]) is not int or rules[k] < 1
        for k in (
            "minimum_frames",
            "minimum_cameras",
            "minimum_layouts",
            "min_visible_per_class",
        )
        if k in rules
    ):
        raise ValueError("count targets must be positive integers")
    if "min_macro_ap50_real" in rules and not 0 <= rules["min_macro_ap50_real"] <= 1:
        raise ValueError("AP50 threshold must lie in [0,1]")
    paired, transfer, reviews, replay_report = (
        p if p and p.is_file() else None
        for p in (paired, transfer, reviews, replay_report)
    )
    out.mkdir(parents=True, exist_ok=True)
    dirs = folders(root)
    if not dirs:
        raise ValueError("empty capture")
    first, last = int(dirs[0].name), int(dirs[-1].name)
    audit = verify(root, out, first, last - first + 1, True)
    header = read(root / "campaign.json")
    checks = []

    def check(name, value, passed):
        checks.append({"criterion": name, "observed": value, "pass": bool(passed)})

    for key, observed in (
        ("minimum_frames", audit["frames"]),
        ("minimum_cameras", len(audit["camera_samples"])),
        ("minimum_layouts", len(audit["layouts"])),
    ):
        if key in rules:
            check(key, observed, observed >= rules[key])
    if "min_visible_per_class" in rules:
        counts = {
            k: audit["instances_count_by_class"].get(k, 0) for k in header["classes"]
        }
        check(
            "min_visible_per_class",
            counts,
            min(counts.values()) >= rules["min_visible_per_class"],
        )
    import json
    from collections import defaultdict

    edits = defaultdict(set)
    actual_edits = defaultdict(set)
    for f in dirs:
        record = read(f / "annotation.json")
        for name, val in record["sample"]["edits"].items():
            edits[name].add(json.dumps(val, sort_keys=True))
        for name in rules.get("min_distinct_texture_sets", {}):
            choice = record.get("readback", {}).get("edits", {}).get(name)
            channels = header.get("texture_sets", {}).get(name, {}).get(choice)
            if channels:
                actual_edits[name].add(tuple(sorted(channels.items())))
    for k, n in rules.get("required_edits", {}).items():
        check("distinct_" + k, len(edits[k]), len(edits[k]) >= n)
    for name, n in rules.get("min_distinct_texture_sets", {}).items():
        if type(n) is not int or n < 1:
            raise ValueError("texture-set coverage must be a positive integer")
        check("distinct_texture_content_" + name, len(actual_edits[name]), len(actual_edits[name]) >= n)
    proofs = {}
    if "appearance_pairs" in rules:
        pairs = rules["appearance_pairs"]
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("appearance_pairs must declare at least one fixed-view pair")
        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict) or not {"first", "second", "edits"} <= pair.keys():
                raise ValueError("each appearance pair requires first, second and edits")
            if (any(type(pair[k]) is not int or pair[k] < 0 for k in ("first", "second"))
                    or not isinstance(pair["edits"], list)
                    or not all(isinstance(v, str) for v in pair["edits"])):
                raise ValueError("appearance pairs require nonnegative indices and a list of edit IDs")
            try:
                measured = compare_appearance(root, **pair)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                measured = {"passed": False, "error": str(exc)}
            for index in (pair["first"], pair["second"]):
                record = root / "frames" / f"{index:08d}" / "annotation.json"
                if record.is_file():
                    proofs[f"appearance_frame_{index}"] = digest(record)
            check(f"appearance_pair_{i}", measured, measured["passed"])
    if rules.get("require_registered_quality") or "registered_inputs" in rules:
        data = read(paired) if paired else {}
        # May use a separately captured, pose-registered qualification scene.
        check(
            "registered_real_views",
            data.get("status", "missing"),
            data.get("status") == "pass"
            and bool(data.get("campaign_key"))
            and bool(data.get("scenes"))
            and bool(data.get("frames"))
            and all(row.get("real_sha256") for row in data["frames"]),
        )
        if paired:
            proofs["registered_quality"] = digest(paired)
        if "registered_inputs" in rules:
            observed, matches = registered_inputs(policy.parent, rules["registered_inputs"], data)
            check("registered_input_matches_policy", observed, matches)
    if "min_macro_ap50_real" in rules or rules.get("real_classes"):
        data = read(transfer) if transfer else {}
        lineage = (
            read(transfer.parent / "lineage.json")
            if transfer and (transfer.parent / "lineage.json").is_file()
            else {}
        )
        if "min_macro_ap50_real" in rules:
            measurement = data.get("macro_ap50")
            check(
                "held_out_real_AP50",
                measurement,
                isinstance(measurement, (int, float))
                and measurement >= rules["min_macro_ap50_real"],
            )
        for code, limits in rules.get("real_classes", {}).items():
            if (
                set(limits) != {"min_ap50", "min_gt"}
                or not 0 <= limits["min_ap50"] <= 1
                or type(limits["min_gt"]) is not int
                or limits["min_gt"] < 1
            ):
                raise ValueError(f"invalid class-specific evaluation threshold: {code}")
            if int(code) not in header["classes"].values():
                raise ValueError(f"class {code} does not belong to this dataset")
            cls = data.get("classes", {}).get(code, {})
            check(
                f"real_class_{code}",
                {"ap50": cls.get("ap"), "ground_truth": cls.get("gt", 0)},
                cls.get("gt", 0) >= limits["min_gt"]
                and isinstance(cls.get("ap"), (int, float))
                and cls["ap"] >= limits["min_ap50"],
            )
        audit_file = transfer.parent / "input_audit.json" if transfer else None
        fit_audit = read(audit_file) if audit_file and audit_file.is_file() else {}
        source_ids = {
            f"{header['campaign_key']}:{int(f.name)}": read(f / "annotation.json")[
                "sha256"
            ]["rgb.png"]
            for f in dirs
        }
        fitted = fit_audit.get("train_rgb_sha256_by_id", {})
        independent = fit_audit.get("test_rgb_sha256_by_id", {})
        linked = bool(fitted) and all(
            source_ids.get(key) == sha for key, sha in fitted.items()
        )
        independent_real = bool(independent) and not set(fitted.values()).intersection(
            independent.values()
        )
        frozen = (
            bool(transfer)
            and lineage.get("scope") == "real"
            and lineage.get("campaign_key") == header["campaign_key"]
            and lineage.get("metrics_sha256") == digest(transfer)
            and bool(fit_audit)
            and lineage.get("input_audit_sha256") == digest(audit_file)
            and lineage.get("real_index_sha256") == fit_audit.get("test_index_sha256")
        )
        check(
            "training_input_matches_capture",
            {
                "native_fit": len(fitted),
                "real_test": len(independent),
                "campaign_key": lineage.get("campaign_key"),
            },
            linked and independent_real and frozen,
        )
        if transfer:
            proofs["transfer"] = digest(transfer)
        if lineage:
            proofs["training_lineage"] = digest(transfer.parent / "lineage.json")
    if rules.get("require_replay"):
        data = read(replay_report) if replay_report else {}
        check(
            "independent_replay",
            data.get("status", "missing"),
            bool(data.get("frames"))
            and data.get("status") == "pass"
            and data.get("campaign_key") == header["campaign_key"],
        )
        if replay_report:
            proofs["replay"] = digest(replay_report)
    reviewed = rules.get("required_visual_frames", [])
    if reviewed:
        data = read(reviews) if reviews else {}
        check(
            "visual_review_capture",
            data.get("campaign_key"),
            data.get("campaign_key") == header["campaign_key"],
        )
        for index in reviewed:
            row = data.get("frames", {}).get(str(index), {})
            reference = root / "frames" / f"{index:08d}" / "rgb.png"
            matches = reference.is_file() and row.get("rgb_sha256") == digest(reference)
            ok = (
                matches
                and row.get("decision") == "pass"
                and bool(str(row.get("observation", "")).strip())
            )
            check(f"visual_review_{index}", row or "missing", ok)
        if reviews:
            proofs["review"] = digest(reviews)
    if not checks:
        raise ValueError("the policy has no active measurable criteria")
    result = {
        "status": "pass" if all(c["pass"] for c in checks) else "fail",
        "campaign_key": header["campaign_key"],
        "policy_sha256": digest(policy),
        "evidence_sha256": proofs,
        "checks": checks,
    }
    write(out / "qualification.json", result)
    return result


def main(operation, argv):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("capture", type=Path)
    if operation == "replay":
        cli.add_argument("independent", type=Path)
        cli.add_argument("--start", type=int, default=0)
        cli.add_argument("--count", type=int, required=True)
        cli.add_argument("--max-rgb-rmse", type=float, default=0.035)
        cli.add_argument("--min-mask-iou", type=float, default=0.97)
        cli.add_argument("--max-range-mae-m", type=float, default=0.005)
        cli.add_argument("--out", type=Path, required=True)
        a = cli.parse_args(argv)
        result = replay(
            a.capture,
            a.independent,
            a.start,
            a.count,
            a.max_rgb_rmse,
            a.min_mask_iou,
            a.max_range_mae_m,
        )
        a.out.parent.mkdir(parents=True, exist_ok=True)
        write(a.out, result)
    else:
        cli.add_argument("--policy", type=Path, required=True)
        cli.add_argument("--out", type=Path, required=True)
        for name in ("paired", "transfer", "reviews", "replay"):
            cli.add_argument("--" + name, type=Path)
        a = cli.parse_args(argv)
        result = release(
            a.capture, a.policy, a.out, a.paired, a.transfer, a.reviews, a.replay
        )
    print(f"{operation}: {result['status']}")
    return int(result["status"] != "pass")
