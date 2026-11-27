"""Fetch only the visible synthetic masks used to build SIFT templates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from range_zip import open_zip


def main() -> None:
    root = Path("/tmp/issac-transfer-bench/data")
    rows = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
    zf, remote = open_zip("https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/5c2c4aa229800355648cd268040aa814f8dc94f0/ycbv_train_pbr.zip")
    hashes = {}
    for row in rows:
        if row["scene"] == 40:
            continue
        for obj in row["objects"]:
            x, y, w, h = obj["bbox_xywh"]
            if obj["category_id"] not in (2, 3, 5) or obj["visibility"] < 0.5 or w < 50 or h < 50:
                continue
            member = (f"train_pbr/{row['scene']:06d}/mask_visib/"
                      f"{row['image_id']:06d}_{obj['instance_index']:06d}.png")
            content = zf.read(member)
            path = root / member
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            hashes[member] = hashlib.sha256(content).hexdigest()
        print(f"masks through {row['archive_member']}: {len(hashes)}", flush=True)
    (root / "template_mask_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({"masks": len(hashes), "transferred": remote.transferred,
                      "requests": remote.requests, "etag": remote.etag}), flush=True)


if __name__ == "__main__":
    main()
