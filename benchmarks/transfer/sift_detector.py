"""Masked synthetic SIFT templates: no pretrained weights, no real fitting.

Train once, inspect only independent synthetic holdout, freeze the model file,
then evaluate the unchanged model on real images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from detector import evaluate, input_audit, intersection_over_union, read_index, row_id


cv2.setNumThreads(1)
DEFAULT_CLASSES = (2, 3, 5)
SIFT_NFEATURES = 900
RATIO = 0.72
MIN_MATCHES = 7
MIN_INLIERS = 6
RANSAC_PIXELS = 5.0
MAX_PER_SCENE = 5
MAX_PER_CLASS = 32
NMS_IOU = 0.40


def mask_path(row: dict, obj: dict, index_dir: Path) -> Path | None:
    if "mask" in obj:
        path = Path(obj["mask"])
        return path if path.is_absolute() else index_dir / path
    member = row.get("archive_member", "")
    if member.startswith("train_pbr/") and "instance_index" in obj and "image_id" in row:
        scene = int(row["scene"])
        return index_dir / "train_pbr" / f"{scene:06d}" / "mask_visib" / f"{row['image_id']:06d}_{obj['instance_index']:06d}.png"
    return None


def load_native_instance_mask(row: dict, index_dir: Path) -> np.ndarray | None:
    """Preserve native integer IDs, including 16-bit values above 255."""
    if "instance_mask" not in row:
        return None
    path = Path(row["instance_mask"])
    if not path.is_absolute():
        path = index_dir / path
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"Instance mask must be one-channel integer PNG: {path}")
    return mask


def fit_templates(rows: list[dict], classes: tuple[int, ...], index_dir: Path,
                  exclude_train_scene: int | None, exclude_train_group: str | None = None) -> dict:
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    candidates = defaultdict(list)
    for row in rows:
        if exclude_train_scene is not None and row.get("scene") == exclude_train_scene:
            continue
        if exclude_train_group is not None and row.get("group") == exclude_train_group:
            continue
        image = cv2.imread(row["image"])
        if image is None:
            raise FileNotFoundError(row["image"])
        ih, iw = image.shape[:2]
        native_ids = load_native_instance_mask(row, index_dir)
        if native_ids is not None and native_ids.shape != (ih, iw):
            raise ValueError(f"Instance mask/image dimensions differ: {row_id(row)}")
        for obj in row["objects"]:
            category = obj["category_id"]
            if category not in classes or obj.get("visibility", 1.0) < 0.5:
                continue
            x, y, w, h = map(int, obj["bbox_xywh"])
            if w < 50 or h < 50:
                continue
            x0, y0, x1, y1 = max(0, x), max(0, y), min(iw, x + w), min(ih, y + h)
            crop = image[y0:y1, x0:x1]
            mask = None
            if native_ids is not None:
                if "instance_id" not in obj:
                    raise ValueError(f"Object lacks instance_id with instance_mask: {row_id(row)}")
                mask = np.uint8(native_ids[y0:y1, x0:x1] == int(obj["instance_id"])) * 255
            else:
                path = mask_path(row, obj, index_dir)
                if path is not None and path.exists():
                    full = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if full is not None:
                        mask = full[y0:y1, x0:x1]
            points, descriptors = sift.detectAndCompute(crop, mask)
            if descriptors is None or len(points) < MIN_MATCHES:
                continue
            group = row.get("group") if row.get("group") is not None else row.get("scene")
            if group is None:
                group = "all"
            candidates[category].append({"group": group, "source_id": row_id(row),
                                         "visibility": obj.get("visibility", 1.0),
                                         "size": (x1 - x0, y1 - y0),
                                         "points": np.array([p.pt for p in points], np.float32),
                                         "descriptors": descriptors})
    models = {}
    for category in classes:
        grouped = defaultdict(list)
        for item in candidates[category]:
            grouped[item["group"]].append(item)
        selected = []
        for group in sorted(grouped, key=str):
            ranked = sorted(grouped[group], key=lambda t: (len(t["points"]) * t["visibility"], t["source_id"]), reverse=True)
            selected += ranked[:MAX_PER_SCENE if len(grouped) > 1 else MAX_PER_CLASS]
        selected = sorted(selected, key=lambda t: len(t["points"]) * t["visibility"], reverse=True)[:MAX_PER_CLASS]
        if not selected:
            raise ValueError(f"No SIFT templates for class {category}")
        models[category] = selected
    return models


def detect(image: np.ndarray, models: dict) -> list[dict]:
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    scene_points, scene_desc = sift.detectAndCompute(image, None)
    if scene_desc is None:
        return []
    coordinates = np.array([p.pt for p in scene_points], np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    ih, iw = image.shape[:2]
    result = []
    for category, templates in models.items():
        candidates = []
        for template in templates:
            pairs = matcher.knnMatch(template["descriptors"], scene_desc, k=2)
            good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < RATIO * b.distance]
            if len(good) < MIN_MATCHES:
                continue
            src = template["points"][[m.queryIdx for m in good]]
            dst = coordinates[[m.trainIdx for m in good]]
            homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PIXELS)
            if homography is None or inliers is None:
                continue
            n_inliers = int(inliers.sum())
            if n_inliers < MIN_INLIERS:
                continue
            w, h = template["size"]
            corners = np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]])
            projected = cv2.perspectiveTransform(corners, homography).reshape(4, 2)
            if not np.isfinite(projected).all():
                continue
            min_xy, max_xy = projected.min(0), projected.max(0)
            x0, y0 = np.maximum(min_xy, [0, 0])
            x1, y1 = np.minimum(max_xy, [iw, ih])
            bw, bh = x1 - x0, y1 - y0
            if bw < 20 or bh < 20 or bw > iw * 0.9 or bh > ih * 0.9:
                continue
            score = n_inliers * (n_inliers / len(good))
            candidates.append({"category_id": category, "bbox_xywh": [int(x0), int(y0), int(bw), int(bh)],
                               "score": float(score), "inliers": n_inliers, "source_id": template["source_id"]})
        candidates.sort(key=lambda d: d["score"], reverse=True)
        kept = []
        for item in candidates:
            if all(intersection_over_union(item["bbox_xywh"], k["bbox_xywh"]) < NMS_IOU for k in kept):
                kept.append(item)
        result.extend(kept[:30])
    return result


def predict(rows: list[dict], models: dict) -> list[dict]:
    predictions = []
    for row in rows:
        image = cv2.imread(row["image"])
        if image is None:
            raise FileNotFoundError(row["image"])
        predictions.append({"image": row_id(row), "scene": row.get("scene"),
                            "image_id": row.get("image_id"), "detections": detect(image, models)})
        print(f"predicted {row_id(row)}", flush=True)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--train-index", type=Path, required=True)
    train_cmd.add_argument("--classes", type=int, nargs="+", default=DEFAULT_CLASSES)
    train_cmd.add_argument("--exclude-train-scene", type=int)
    train_cmd.add_argument("--exclude-train-group")
    train_cmd.add_argument("--output", type=Path, required=True)
    eval_cmd = sub.add_parser("evaluate")
    eval_cmd.add_argument("--model", type=Path, required=True)
    eval_cmd.add_argument("--train-index", type=Path, required=True)
    eval_cmd.add_argument("--test-index", type=Path, required=True)
    eval_cmd.add_argument("--exclude-train-scene", type=int)
    eval_cmd.add_argument("--exclude-train-group")
    eval_cmd.add_argument("--only-test-scene", type=int)
    eval_cmd.add_argument("--only-test-group")
    eval_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "train":
        classes = tuple(args.classes)
        if len(set(classes)) != len(classes):
            parser.error("class IDs must be unique")
        rows = read_index(args.train_index)
        models = fit_templates(rows, classes, args.train_index.parent,
                               args.exclude_train_scene, args.exclude_train_group)
        path = args.output / "model.pkl"
        path.write_bytes(pickle.dumps(models))
        mask_manifest = args.train_index.parent / "template_mask_sha256.json"
        manifest = {"classes": classes, "train_index_sha256": hashlib.sha256(args.train_index.read_bytes()).hexdigest(),
                    "excluded_train_scene": args.exclude_train_scene,
                    "excluded_train_group": args.exclude_train_group,
                    "template_counts": {str(k): len(v) for k, v in models.items()},
                    "template_source_ids": {str(k): [item["source_id"] for item in v] for k, v in models.items()},
                    "mask_manifest_sha256": hashlib.sha256(mask_manifest.read_bytes()).hexdigest() if mask_manifest.exists() else None,
                    "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "settings": {"sift_features": SIFT_NFEATURES, "ratio": RATIO, "min_matches": MIN_MATCHES,
                                 "min_inliers": MIN_INLIERS, "ransac_pixels": RANSAC_PIXELS,
                                 "max_per_scene": MAX_PER_SCENE, "max_per_class": MAX_PER_CLASS, "nms_iou": NMS_IOU}}
        (args.output / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return
    with args.model.open("rb") as f:
        models = pickle.load(f)
    classes = tuple(models)
    train, test = read_index(args.train_index), read_index(args.test_index)
    if args.only_test_scene is not None:
        test = [r for r in test if r.get("scene") == args.only_test_scene]
        if not test:
            parser.error(f"No test records for scene {args.only_test_scene}")
    if args.only_test_group is not None:
        test = [r for r in test if r.get("group") == args.only_test_group]
        if not test:
            parser.error(f"No test records for group {args.only_test_group}")
    audit = input_audit(args.train_index, args.test_index, train, test, classes,
                        args.exclude_train_scene, args.exclude_train_group)
    audit["model_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
    (args.output / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    preds = predict(test, models)
    metrics = evaluate(test, preds, classes)
    (args.output / "predictions.json").write_text(json.dumps(preds))
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
