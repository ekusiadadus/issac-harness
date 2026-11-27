"""Small synthetic-only HOG/SVM detector and held-out real BOP evaluation.

No real image or annotation enters fit(), aspect selection, or a threshold choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from functools import cache
from pathlib import Path

import numpy as np

DEFAULT_CLASSES = (2, 3, 5)


@cache
def hog_descriptor():
    import cv2
    return cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


def read_index(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    for row in rows:
        image = Path(row["image"])
        row["_id"] = str(row.get("id") or row.get("archive_member") or row["image"])
        row["_group"] = row.get("group", row.get("scene", "all"))
        if not image.is_absolute():
            row["image"] = str((path.parent / image).resolve())
    return rows


def row_id(row: dict) -> str:
    return str(row.get("_id") or row.get("id") or row.get("archive_member") or row["image"])


def input_audit(train_index: Path, test_index: Path, train: list[dict], test: list[dict],
                classes: tuple[int, ...], exclude_train_scene: int | None,
                exclude_train_group: str | None = None) -> dict:
    """Hash actual RGB bytes before fit; raise on any cross-split identical image."""
    def counts(rows: list[dict]) -> dict:
        return {str(category): {
            "instances": sum(o["category_id"] == category for row in rows for o in row["objects"]),
            "image_examples": [row_id(row) for row in rows if any(o["category_id"] == category for o in row["objects"])][:5],
        } for category in classes}

    def hashes(rows: list[dict]) -> dict:
        result = {}
        for row in rows:
            key = row_id(row)
            if key in result:
                raise ValueError(f"Duplicate image ID in one index: {key}")
            result[key] = hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest()
        return result

    fit_rows = [row for row in train
                if (exclude_train_scene is None or row.get("scene") != exclude_train_scene)
                and (exclude_train_group is None or row.get("group") != exclude_train_group)]
    train_hashes, test_hashes = hashes(fit_rows), hashes(test)
    reverse_train = defaultdict(list)
    for key, digest in train_hashes.items():
        reverse_train[digest].append(key)
    overlap = [{"train_ids": reverse_train[digest], "test_id": key, "sha256": digest}
               for key, digest in test_hashes.items() if digest in reverse_train]
    if overlap:
        raise ValueError(f"Byte-identical train/test RGB leakage: {overlap[:5]}")
    return {
        "train_index_sha256": hashlib.sha256(train_index.read_bytes()).hexdigest(),
        "test_index_sha256": hashlib.sha256(test_index.read_bytes()).hexdigest(),
        "train_index_images": len(train), "train_fit_images": len(fit_rows), "test_images": len(test),
        "classes": list(classes), "excluded_train_scene": exclude_train_scene,
        "excluded_train_group": exclude_train_group,
        "train_fit_counts": counts(fit_rows), "test_counts": counts(test),
        "train_rgb_sha256_by_id": train_hashes, "test_rgb_sha256_by_id": test_hashes,
        "byte_identical_train_test_rgb_count": 0,
    }


def feature(image: np.ndarray) -> np.ndarray:
    import cv2
    patch = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    hog = hog_descriptor().compute(patch).reshape(-1)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    color = np.concatenate([hsv.mean((0, 1)) / [180, 255, 255], hsv.std((0, 1)) / [180, 255, 255]])
    return np.concatenate([hog, color.astype(np.float32) * 5]).astype(np.float32)


def intersection_over_union(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    area = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return area / (aw * ah + bw * bh - area) if area else 0.0


def fit(index: list[dict], classes: tuple[int, ...] = DEFAULT_CLASSES,
        exclude_train_scene: int | None = None) -> dict:
    import cv2
    from sklearn.svm import LinearSVC

    rng = np.random.default_rng(7319)
    models = {}
    for category in classes:
        def usable(o: dict) -> bool:
            box = o["bbox_xywh"]
            return (o["category_id"] == category and box[2] >= 24 and box[3] >= 24
                    and o.get("visibility", 1.0) >= 0.25)

        boxes = [o["bbox_xywh"] for row in index
                 if exclude_train_scene is None or row.get("scene") != exclude_train_scene
                 for o in row["objects"] if usable(o)]
        if not boxes:
            raise ValueError(f"Insufficient synthetic samples for category {category}")
        aspect = float(np.median([b[2] / b[3] for b in boxes]))
        xs, ys = [], []
        for row in index:
            if exclude_train_scene is not None and row.get("scene") == exclude_train_scene:
                continue
            image = cv2.imread(row["image"])
            if image is None:
                raise FileNotFoundError(row["image"])
            h, w = image.shape[:2]
            positives = [o["bbox_xywh"] for o in row["objects"] if usable(o)]
            for x, y, bw, bh in positives:
                margin_x, margin_y = bw * 0.08, bh * 0.08
                x0, y0 = max(0, int(x - margin_x)), max(0, int(y - margin_y))
                x1, y1 = min(w, int(x + bw + margin_x)), min(h, int(y + bh + margin_y))
                if x1 > x0 and y1 > y0:
                    xs.append(feature(image[y0:y1, x0:x1])); ys.append(1)
            # Image backgrounds and other objects are synthetic negatives.
            for _ in range(4):
                bh = int(rng.choice((64, 96, 128, 192, 256)))
                bw = int(bh * aspect)
                if bw >= w or bh >= h:
                    continue
                for _attempt in range(20):
                    x, y = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
                    box = (x, y, bw, bh)
                    if max((intersection_over_union(box, b) for b in positives), default=0) < 0.1:
                        xs.append(feature(image[y:y + bh, x:x + bw])); ys.append(0)
                        break
        if not xs or len(set(ys)) < 2:
            raise ValueError(f"Insufficient synthetic samples for category {category}")
        classifier = LinearSVC(C=0.1, class_weight="balanced", max_iter=3000, random_state=7319)
        classifier.fit(np.stack(xs), ys)
        models[category] = {"classifier": classifier, "aspect": aspect, "positive": sum(ys), "negative": len(ys) - sum(ys)}
    return models


def detect_one(image: np.ndarray, model: dict) -> list[dict]:
    import cv2
    classifier, aspect = model["classifier"], model["aspect"]
    h, w = image.shape[:2]
    candidates = []
    for box_h in (64, 96, 128, 192, 256):
        box_w = int(box_h * aspect)
        if box_w >= w or box_h >= h:
            continue
        sx, sy = 64 / box_w, 64 / box_h
        scaled = cv2.resize(image, (max(64, int(w * sx)), max(64, int(h * sy))))
        sh, sw = scaled.shape[:2]
        locations = [(x, y) for y in range(0, sh - 63, 16) for x in range(0, sw - 63, 16)]
        if not locations:
            continue
        descriptors = hog_descriptor().compute(scaled, winStride=(8, 8), locations=locations)
        hogs = descriptors.reshape(len(locations), -1)
        hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV).astype(np.float32)
        colors = np.empty((len(locations), 6), dtype=np.float32)
        for i, (x, y) in enumerate(locations):
            patch = hsv[y:y + 64, x:x + 64]
            colors[i, :3] = patch.mean((0, 1)) / [180, 255, 255]
            colors[i, 3:] = patch.std((0, 1)) / [180, 255, 255]
        scores = classifier.decision_function(np.concatenate([hogs, colors * 5], axis=1))
        for i in np.argsort(scores)[-100:]:
            x, y = locations[i]
            candidates.append({"bbox_xywh": [int(x / sx), int(y / sy), box_w, box_h], "score": float(scores[i])})
    candidates.sort(key=lambda d: d["score"], reverse=True)
    kept = []
    for candidate in candidates:
        if all(intersection_over_union(candidate["bbox_xywh"], other["bbox_xywh"]) < 0.4 for other in kept):
            kept.append(candidate)
            if len(kept) == 30:
                break
    return kept


def predict(index: list[dict], models: dict) -> list[dict]:
    import cv2
    result = []
    for row in index:
        image = cv2.imread(row["image"])
        detections = []
        for category, model in models.items():
            detections += [{"category_id": category, **d} for d in detect_one(image, model)]
        result.append({"image": row_id(row), "scene": row.get("scene"), "image_id": row.get("image_id"), "detections": detections})
        print(f"predicted {row_id(row)}", flush=True)
    return result


def ap_for_class(rows: list[dict], predictions: list[dict], category: int, threshold: float) -> dict:
    ground_truth = {}
    count = 0
    for row in rows:
        key = row_id(row)
        ground_truth[key] = [o["bbox_xywh"] for o in row["objects"] if o["category_id"] == category]
        count += len(ground_truth[key])
    ranked = sorted(((d["score"], pred["image"], d["bbox_xywh"]) for pred in predictions for d in pred["detections"] if d["category_id"] == category), reverse=True)
    matched = defaultdict(set)
    tp, fp = [], []
    for _, key, box in ranked:
        candidates = [(intersection_over_union(box, gt), i) for i, gt in enumerate(ground_truth[key]) if i not in matched[key]]
        best = max(candidates, default=(0, -1))
        hit = best[0] >= threshold
        if hit:
            matched[key].add(best[1])
        tp.append(int(hit)); fp.append(int(not hit))
    if count == 0:
        return {"ap": None, "recall": None, "gt": 0, "predictions": len(ranked)}
    recalls = np.cumsum(tp) / count
    precisions = np.cumsum(tp) / np.maximum(1, np.cumsum(tp) + np.cumsum(fp))
    ap = np.mean([max(precisions[recalls >= r], default=0) for r in np.linspace(0, 1, 101)])
    return {"ap": float(ap), "recall": float(recalls[-1]) if len(recalls) else 0.0,
            "gt": count, "predictions": len(ranked)}


def evaluate(rows: list[dict], predictions: list[dict], classes: tuple[int, ...] = DEFAULT_CLASSES) -> dict:
    thresholds = np.arange(0.5, 1.0, 0.05)
    overall = {}
    for category in classes:
        at50 = ap_for_class(rows, predictions, category, 0.5)
        aps = [ap_for_class(rows, predictions, category, float(t))["ap"] for t in thresholds]
        overall[str(category)] = {**at50, "ap50_95": float(np.mean(aps)) if at50["gt"] else None}
    per_scene = {}
    for scene in sorted({r.get("_group", r.get("group", r.get("scene", "all"))) for r in rows}, key=str):
        selected_rows = [r for r in rows if r.get("_group", r.get("group", r.get("scene", "all"))) == scene]
        names = {row_id(r) for r in selected_rows}
        selected_preds = [p for p in predictions if p["image"] in names]
        per_scene[str(scene)] = {str(c): ap_for_class(selected_rows, selected_preds, c, 0.5) for c in classes}
    valid = [x for x in overall.values() if x["gt"]]
    return {"classes": overall, "macro_ap50": float(np.mean([x["ap"] for x in valid])) if valid else None,
            "macro_ap50_95": float(np.mean([x["ap50_95"] for x in valid])) if valid else None,
            "macro_recall50": float(np.mean([x["recall"] for x in valid])) if valid else None,
            "per_scene": per_scene, "images": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, nargs="+", default=DEFAULT_CLASSES,
                        help="BOP/category IDs to fit and evaluate (default: 2 3 5)")
    parser.add_argument("--exclude-train-scene", type=int,
                        help="Explicit synthetic holdout scene; published baseline used 40")
    parser.add_argument("--only-test-scene", type=int,
                        help="Evaluate only this scene from the test index, e.g. synthetic holdout 40")
    parser.add_argument("--audit-only", action="store_true",
                        help="Verify inputs and write input_audit.json without training")
    args = parser.parse_args()
    classes = tuple(args.classes)
    if len(set(classes)) != len(classes):
        parser.error("--classes must contain unique category IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    train, test = read_index(args.train_index), read_index(args.test_index)
    if args.only_test_scene is not None:
        test = [row for row in test if row.get("scene") == args.only_test_scene]
        if not test:
            parser.error(f"No test records for scene {args.only_test_scene}")
    audit = input_audit(args.train_index, args.test_index, train, test, classes, args.exclude_train_scene)
    (args.output / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if args.audit_only:
        print(json.dumps({k: v for k, v in audit.items() if not k.endswith("_by_id")}, indent=2))
        return
    models = fit(train, classes, args.exclude_train_scene)
    with (args.output / "model.pkl").open("wb") as f:
        pickle.dump(models, f)
    test_pred = predict(test, models)
    metrics = evaluate(test, test_pred, classes)
    (args.output / "predictions.json").write_text(json.dumps(test_pred))
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
