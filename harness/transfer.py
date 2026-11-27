"""Fit a synthetic-only baseline, freeze it, then score a layout holdout and independent real images."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from .campaign import digest, folders, read, verify, write


def transfer(capture, real_index, output, categories, holdout, method="sift"):
    if output.exists():
        raise FileExistsError(f"use a new model output directory: {output}")
    if method == "sift":
        script, checkpoint, modules = "sift_detector.py", "model.pkl", ["cv2"]
    elif method == "ssdlite":
        script, checkpoint, modules = "neural_detector.py", "model.pt", ["torch", "torchvision"]
    else:
        raise ValueError(f"unsupported detection method {method}")
    if any(importlib.util.find_spec(p) is None for p in modules):
        raise RuntimeError(f"{method} training requires {', '.join(modules)} in this Python environment")
    head = read(capture / "campaign.json")
    if not categories or set(categories) - set(head["classes"].values()):
        raise ValueError("choose class IDs in the campaign registry")
    output.mkdir(parents=True)
    labels = output / "inputs"
    dirs = folders(capture)
    if not dirs:
        raise ValueError("empty capture")
    verify(
        capture,
        labels,
        int(dirs[0].name),
        int(dirs[-1].name) - int(dirs[0].name) + 1,
        True,
    )
    native = labels / "training.jsonl"
    rows = [json.loads(line) for line in native.read_text().splitlines()]
    groups = {row["group"] for row in rows}
    if holdout not in groups or groups == {holdout}:
        raise ValueError("holdout must be one of at least two native layout groups")
    detector = Path(__file__).resolve().parents[1] / "benchmarks/transfer" / script
    model = output / checkpoint
    common = ["--train-index", str(native), "--exclude-train-group", holdout]

    def invoke(name, args):
        with (output / (name + ".log")).open("w") as log:
            subprocess.run(
                [sys.executable, str(detector), *args],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    invoke(
        "train",
        ["train", *common, "--classes", *map(str, categories), "--output", str(output)],
    )
    freeze = {
        "campaign_key": head["campaign_key"],
        "model_sha256": digest(model),
        "implementation_sha256": digest(detector),
        "evaluator_sha256": digest(detector.with_name("detector.py")),
        "method": method,
        "native_index_sha256": digest(native),
        "holdout_layout": holdout,
        "real_index_sha256": digest(real_index),
        "evaluated_classes": categories,
    }
    write(output / "frozen.json", freeze)
    for split, index, selection in (
        ("synthetic-holdout", native, ["--only-test-group", holdout]),
        ("real", real_index, []),
    ):
        if digest(model) != freeze["model_sha256"]:
            raise ValueError("model changed after freeze")
        folder = output / split
        invoke(
            split,
            [
                "evaluate",
                *common,
                "--model",
                str(model),
                "--test-index",
                str(index),
                *selection,
                "--output",
                str(folder),
            ],
        )
        audit = read(folder / "input_audit.json")
        if audit["train_index_sha256"] != freeze["native_index_sha256"] or audit[
            "test_index_sha256"
        ] != digest(index):
            raise ValueError("input audit does not match evaluated indexes")
        write(
            folder / "lineage.json",
            {
                **freeze,
                "scope": split,
                "input_audit_sha256": digest(folder / "input_audit.json"),
                "metrics_sha256": digest(folder / "metrics.json"),
                "fit_images": audit["train_fit_images"],
                "evaluated_images": audit["test_images"],
            },
        )
    return read(output / "real/metrics.json")


def main(argv):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("capture", type=Path)
    cli.add_argument("--real-index", required=True, type=Path)
    cli.add_argument("--classes", nargs="+", type=int, required=True)
    cli.add_argument("--holdout-layout", required=True)
    cli.add_argument("--detector", choices=["sift", "ssdlite"], default="sift")
    cli.add_argument("--out", type=Path, required=True)
    a = cli.parse_args(argv)
    result = transfer(
        a.capture.resolve(),
        a.real_index.resolve(),
        a.out.resolve(),
        a.classes,
        a.holdout_layout, a.detector,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("images", "macro_ap50", "macro_ap50_95", "macro_recall50")
            },
            indent=2,
        )
    )
