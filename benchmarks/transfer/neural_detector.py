"""Train a random-initialized TorchVision SSDLite on synthetic JSONL images.

The training split is selected before fitting. No pretrained detector or backbone
weights are loaded; evaluation uses the saved state_dict without fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from detector import evaluate, input_audit, read_index, row_id

DEFAULT_CLASSES = (2, 3, 5)
DEFAULT_EXCLUDED_GROUP = "cross_pick"


def dependencies():
    """Delay heavyweight imports so CLI help and data checks work without Torch."""
    try:
        import numpy as np
        import torch
        import torchvision
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install matching torch and torchvision builds, plus Pillow and NumPy") from exc
    return Image, np, torch, torchvision


def fit_rows(rows, excluded_scene=None, excluded_group=DEFAULT_EXCLUDED_GROUP):
    return [row for row in rows
            if (excluded_scene is None or row.get("scene") != excluded_scene)
            and (excluded_group is None or row.get("group") != excluded_group)]


def clipped_targets(row, classes, Image, np, torch):
    with Image.open(row["image"]) as source:
        image = source.convert("RGB")
        width, height = image.size
    boxes, labels = [], []
    registry = {category: index + 1 for index, category in enumerate(classes)}
    for obj in row["objects"]:
        if obj["category_id"] not in registry:
            continue
        x, y, w, h = map(float, obj["bbox_xywh"])
        x0, y0 = max(0.0, min(float(width), x)), max(0.0, min(float(height), y))
        x1, y1 = max(0.0, min(float(width), x + w)), max(0.0, min(float(height), y + h))
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append([x0, y0, x1, y1])
        labels.append(registry[obj["category_id"]])
    # TorchVision's SSD transform applies the model's mean/std and 320 resize.
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255.0)
    target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
              "labels": torch.tensor(labels, dtype=torch.int64)}
    return tensor, target


def make_model(classes, torchvision):
    return torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=len(classes) + 1)


def seed_everything(seed, torch):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Disable changing benchmark algorithms across epochs/runs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(rows, classes, output, *, epochs=24, batch_size=8, learning_rate=0.001,
          seed=42, device="auto", num_workers=0):
    Image, np, torch, torchvision = dependencies()
    if not rows:
        raise ValueError("No images remain in the training split")
    present = {o["category_id"] for row in rows for o in row["objects"]}
    absent = set(classes) - present
    if absent:
        raise ValueError(f"No training instances for categories: {sorted(absent)}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed_everything(seed, torch)

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            return clipped_targets(rows[index], classes, Image, np, torch)

    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        Dataset(), batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=num_workers, collate_fn=lambda batch: tuple(zip(*batch)))
    model = make_model(classes, torchvision).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    started = time.monotonic()
    for epoch in range(epochs):
        total, batches = 0.0, 0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at epoch {epoch + 1}: {losses}")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        item = {"epoch": epoch + 1, "mean_batch_loss": total / batches, "batches": batches}
        history.append(item)
        print(json.dumps(item), flush=True)
    checkpoint = output / "model.pt"
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint)
    return {"model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "fit_elapsed_seconds": time.monotonic() - started,
            "history": history, "device": str(device),
            "torch_version": torch.__version__, "torchvision_version": torchvision.__version__}


def predict(rows, model, classes, device, score_threshold=0.05):
    Image, np, torch, _ = dependencies()
    reverse_registry = {index + 1: category for index, category in enumerate(classes)}
    model.eval()
    result = []
    with torch.inference_mode():
        for row in rows:
            image, _ = clipped_targets(row, classes, Image, np, torch)
            output = model([image.to(device)])[0]
            detections = []
            for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
                category = reverse_registry.get(int(label))
                if category is None or float(score) < score_threshold:
                    continue
                x0, y0, x1, y1 = box.tolist()
                x, y, w, h = int(x0), int(y0), int(x1 - x0), int(y1 - y0)
                if w > 0 and h > 0:
                    detections.append({"category_id": category, "bbox_xywh": [x, y, w, h],
                                       "score": float(score)})
            result.append({"image": row_id(row), "scene": row.get("scene"),
                           "image_id": row.get("image_id"), "detections": detections})
            print(f"predicted {row_id(row)}", flush=True)
    return result


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def image_set_sha256(rows):
    """Bind the fit to ordered image IDs and actual encoded image bytes."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row_id(row).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(Path(row["image"]).read_bytes()).digest())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--train-index", type=Path, required=True)
    train_cmd.add_argument("--classes", type=int, nargs="+", default=DEFAULT_CLASSES)
    train_cmd.add_argument("--exclude-train-scene", type=int)
    train_cmd.add_argument("--exclude-train-group", default=DEFAULT_EXCLUDED_GROUP)
    train_cmd.add_argument("--output", type=Path, required=True)
    train_cmd.add_argument("--epochs", type=int, default=24)
    train_cmd.add_argument("--batch-size", type=int, default=8)
    train_cmd.add_argument("--learning-rate", type=float, default=0.001)
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument("--device", default="auto")
    train_cmd.add_argument("--num-workers", type=int, default=0)
    eval_cmd = sub.add_parser("evaluate")
    eval_cmd.add_argument("--model", type=Path, required=True)
    eval_cmd.add_argument("--train-index", type=Path, required=True)
    eval_cmd.add_argument("--test-index", type=Path, required=True)
    eval_cmd.add_argument("--exclude-train-scene", type=int)
    eval_cmd.add_argument("--exclude-train-group", default=DEFAULT_EXCLUDED_GROUP)
    eval_cmd.add_argument("--only-test-scene", type=int)
    eval_cmd.add_argument("--only-test-group")
    eval_cmd.add_argument("--output", type=Path, required=True)
    eval_cmd.add_argument("--device", default="auto")
    eval_cmd.add_argument("--score-threshold", type=float, default=0.05)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "train":
        classes = tuple(args.classes)
        if len(set(classes)) != len(classes):
            parser.error("--classes must be unique category IDs")
        if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
            parser.error("epochs, batch-size and learning-rate must be positive")
        rows = read_index(args.train_index)
        selected = fit_rows(rows, args.exclude_train_scene, args.exclude_train_group)
        fit_images_sha256 = image_set_sha256(selected)
        result = train(selected, classes, args.output, epochs=args.epochs,
                       batch_size=args.batch_size, learning_rate=args.learning_rate,
                       seed=args.seed, device=args.device, num_workers=args.num_workers)
        manifest = {"architecture": "ssdlite320_mobilenet_v3_large", "weights": None,
                    "weights_backbone": None, "classes": list(classes),
                    "category_to_label": {str(c): i + 1 for i, c in enumerate(classes)},
                    "train_index_sha256": hashlib.sha256(args.train_index.read_bytes()).hexdigest(),
                    "train_index_images": len(rows), "train_fit_images": len(selected),
                    "train_fit_images_sha256": fit_images_sha256,
                    "excluded_train_scene": args.exclude_train_scene,
                    "excluded_train_group": args.exclude_train_group,
                    "epochs": args.epochs, "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate, "optimizer": "AdamW",
                    "seed": args.seed, "normalization": "RGB float [0,1]; TorchVision SSD transform",
                    **result}
        write_json(args.output / "model_manifest.json", manifest)
        print(json.dumps({k: v for k, v in manifest.items() if k != "history"}, indent=2))
        return
    manifest_path = args.model.parent / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    classes = tuple(manifest["classes"])
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if digest != manifest["model_sha256"]:
        raise ValueError("Checkpoint SHA256 differs from model manifest")
    if args.exclude_train_scene != manifest["excluded_train_scene"] or args.exclude_train_group != manifest["excluded_train_group"]:
        raise ValueError("Evaluation train exclusion differs from fitted model")
    train_rows, test_rows = read_index(args.train_index), read_index(args.test_index)
    if hashlib.sha256(args.train_index.read_bytes()).hexdigest() != manifest["train_index_sha256"]:
        raise ValueError("Evaluation train index differs from fitted model")
    if image_set_sha256(fit_rows(train_rows, args.exclude_train_scene, args.exclude_train_group)) != manifest["train_fit_images_sha256"]:
        raise ValueError("Fitted training image bytes differ from model manifest")
    if args.only_test_scene is not None:
        test_rows = [row for row in test_rows if row.get("scene") == args.only_test_scene]
    if args.only_test_group is not None:
        test_rows = [row for row in test_rows if row.get("group") == args.only_test_group]
    if not test_rows:
        parser.error("No test records match selection")
    audit = input_audit(args.train_index, args.test_index, train_rows, test_rows, classes,
                        args.exclude_train_scene, args.exclude_train_group)
    audit["model_sha256"] = digest
    write_json(args.output / "input_audit.json", audit)
    _, _, torch, torchvision = dependencies()
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(classes, torchvision)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.to(device)
    predictions = predict(test_rows, model, classes, device, args.score_threshold)
    metrics = evaluate(test_rows, predictions, classes)
    metrics["model_sha256"] = digest
    metrics["score_threshold"] = args.score_threshold
    write_json(args.output / "predictions.json", predictions)
    write_json(args.output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
