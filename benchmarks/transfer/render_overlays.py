"""Render fixed examples from the measured test split for visual inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from detector import read_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_index(args.test_index)
    preds = {p["image"]: p for p in json.loads(args.predictions.read_text())}
    selected = {}
    for row in rows:
        if row["archive_member"] in preds:
            selected.setdefault(row["scene"], row)
    for scene, row in selected.items():
        image = cv2.imread(row["image"])
        for o in row["objects"]:
            if o["category_id"] not in (2, 3, 5):
                continue
            x, y, w, h = o["bbox_xywh"]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"GT {o['category_id']}", (x, max(14, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        for category, color in ((2, (0, 0, 255)), (3, (255, 0, 0)), (5, (0, 255, 255))):
            ds = [d for d in preds[row["archive_member"]]["detections"] if d["category_id"] == category]
            for d in sorted(ds, key=lambda x: x["score"], reverse=True)[:1]:
                x, y, w, h = d["bbox_xywh"]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, f"{category}:{d['score']:.2f}", (x, min(image.shape[0] - 4, y + h + 13)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imwrite(str(args.output / f"scene_{scene}_frame_{row['image_id']:06d}.jpg"), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
