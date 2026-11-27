"""Verify a copied Replicator pilot and, optionally, its independent seeded replay."""

import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .project import file_hash, inside, write_json
from .review import read


def verify(directory):
    directory = directory.resolve()
    manifest = read(directory / "manifest.json")
    frames = manifest["frames"]
    if manifest["status"] != "captured" or not frames or len(frames) != manifest["frames_requested"]:
        raise ValueError("incomplete dataset run")
    width, height = manifest["render"]["resolution"]
    stats = []
    for i, meta in enumerate(frames):
        folder = directory / f"frame_{i:04d}"
        expected = {
            folder / f for f in ("annotation.json", "rgb.png", "distance_to_camera_m.npy", "semantic_ids.npy")
        }
        files = {inside(directory, f["path"]): f["sha256"] for f in meta["files"]}
        if files.keys() != expected or any(file_hash(p) != sha for p, sha in files.items()):
            raise ValueError(f"frame {i}: incomplete files or content hash mismatch")
        annotation = read(folder / "annotation.json")
        for key in (
            "boxes",
            "sampled",
            "readback",
            "index",
            "segmentation_id_to_labels",
            "bbox_id_to_labels",
        ):
            if meta[key] != annotation[key]:
                raise ValueError(f"frame {i}: manifest and annotation disagree on {key}")
        ids = np.load(folder / "semantic_ids.npy", allow_pickle=False)
        depth = np.load(folder / "distance_to_camera_m.npy", allow_pickle=False)
        with Image.open(folder / "rgb.png") as im:
            if im.size != (width, height) or np.asarray(im.convert("L")).std() == 0:
                raise ValueError(f"frame {i}: invalid RGB")
        if (
            ids.shape != (height, width)
            or depth.shape != ids.shape
            or ids.dtype != np.uint32
            or depth.dtype != np.float32
            or i != meta["index"]
        ):
            raise ValueError(f"frame {i}: unexpected annotation shape, dtype or index")
        classes = {}
        for box in meta["boxes"]:
            label = box["class"]
            for semantic_id in box["segmentation_ids"]:
                if meta["segmentation_id_to_labels"][str(semantic_id)]["class"] != label:
                    raise ValueError(f"frame {i}: incorrect ID for {label}")
            mask = np.isin(ids, box["segmentation_ids"])
            y, x = np.where(mask)
            left, top, right, bottom = box["xyxy_px"]
            if (
                not x.size
                or int(x.size) != box["visible_pixels"]
                or not (0 <= left <= right < width and 0 <= top <= bottom < height)
                or x.min() < left - 2
                or x.max() >= right + 2
                or y.min() < top - 2
                or y.max() >= bottom + 2
            ):
                raise ValueError(f"frame {i}: mask disagrees with {label} box")
            if not (np.isfinite(depth[mask]) & (depth[mask] > 0)).all():
                raise ValueError(f"frame {i}: invalid {label} range")
            classes[label] = {"pixels": int(mask.sum()), "median_range_m": float(np.median(depth[mask]))}
        if classes.keys() != {"carton", "handheld_scanner"}:
            raise ValueError(f"frame {i}: missing pilot class")
        stats.append(classes)
    return manifest, stats


def dataset_check(directory, replay=None):
    manifest, stats = verify(directory)
    report = {
        "status": "pass",
        "seed": manifest["seed"],
        "frames": stats,
        "manifest_sha256": file_hash(directory / "manifest.json"),
    }
    if replay is None:
        return report
    if directory.resolve() == replay.resolve():
        raise ValueError("use a separately captured replay")
    other, _ = verify(replay)
    for key in ("seed", "camera_model", "render", "frames_requested", "versions", "source_files"):
        if other[key] != manifest[key]:
            raise ValueError(f"replay conditions differ: {key}")
    metrics = []
    for a, b in zip(manifest["frames"], other["frames"], strict=True):
        if a["sampled"] != b["sampled"] or a["readback"] != b["readback"]:
            raise ValueError(f"replay sampled or effective scene state differs at frame {a['index']}")
        folder = f"frame_{a['index']:04d}"
        with Image.open(directory / folder / "rgb.png") as im, Image.open(replay / folder / "rgb.png") as ref:
            delta = np.asarray(im.convert("RGB"), dtype=np.float32) - np.asarray(
                ref.convert("RGB"), dtype=np.float32
            )
        ids = np.load(directory / folder / "semantic_ids.npy")
        replay_ids = np.load(replay / folder / "semantic_ids.npy")
        masks_equal = np.array_equal(ids, replay_ids)
        depth = np.load(directory / folder / "distance_to_camera_m.npy")
        replay_depth = np.load(replay / folder / "distance_to_camera_m.npy")
        ranges_equal = np.array_equal(depth, replay_depth, equal_nan=True)
        boxes_equal = {v["class"]: v["xyxy_px"] for v in a["boxes"]} == {
            v["class"]: v["xyxy_px"] for v in b["boxes"]
        }
        target_masks_equal = True
        range_error_m = 0.0
        replay_boxes = {box["class"]: box for box in b["boxes"]}
        for box in a["boxes"]:
            mask = np.isin(ids, box["segmentation_ids"])
            other_mask = np.isin(replay_ids, replay_boxes[box["class"]]["segmentation_ids"])
            target_masks_equal &= np.array_equal(mask, other_mask)
            shared = mask & other_mask
            if shared.any():
                range_error_m = max(
                    range_error_m, float(np.max(np.abs(depth[shared] - replay_depth[shared])))
                )
        rmse = float(np.sqrt(np.mean(delta**2)) / 255)
        metrics.append(
            {
                "rgb_rmse_srgb": rmse,
                "semantic_ids_exact": masks_equal,
                "range_exact": ranges_equal,
                "target_masks_exact": target_masks_equal,
                "max_target_range_error_m": range_error_m,
                "boxes_exact": boxes_equal,
            }
        )
        if not boxes_equal or not target_masks_equal or range_error_m > 0.001 or rmse > 0.025:
            report["status"] = "fail"
    report["replay_manifest_sha256"] = file_hash(replay / "manifest.json")
    report["replay_thresholds"] = {"rgb_rmse_srgb": 0.025, "max_target_range_error_m": 0.001}
    report["replay"] = metrics
    return report


TRAINING_CLASSES = {"background/other": 0, "carton": 1, "handheld_scanner": 2}


def _class_mask(ids, mapping):
    """Use semantic class names: Replicator's numeric IDs can change between runs."""
    result = np.zeros(ids.shape, dtype=np.uint8)
    for raw_id, labels in mapping.items():
        name = labels["class"]
        if name in ("carton", "handheld_scanner"):
            result[ids == int(raw_id)] = TRAINING_CLASSES[name]
    return result


def _exclusive_box(native_xyxy):
    left, top, right, bottom = native_xyxy
    return [left, top, right + 1, bottom + 1]


def export_training(directory, destination):
    """Create a small, stable-class index without copying the original RGB/range arrays."""
    directory = directory.resolve()
    destination = destination.resolve()
    manifest, _ = verify(directory)
    if destination.exists():
        raise FileExistsError(f"training destination already exists: {destination}")
    destination.mkdir(parents=True)
    try:
        return _write_training(directory, destination, manifest)
    except BaseException:
        shutil.rmtree(destination)
        raise


def _write_training(directory, destination, manifest):
    relative_source = os.path.relpath(directory, destination)
    frames = []
    for meta in manifest["frames"]:
        index = meta["index"]
        source = directory / f"frame_{index:04d}"
        ids = np.load(source / "semantic_ids.npy", allow_pickle=False)
        mask = _class_mask(ids, meta["segmentation_id_to_labels"])
        if not all(np.any(mask == class_id) for class_id in (1, 2)):
            raise ValueError(f"frame {index}: empty target class mask")
        mask_path = destination / f"frame_{index:04d}_mask.png"
        Image.fromarray(mask).save(mask_path)
        source_frame = f"{relative_source}/frame_{index:04d}"
        label_path = destination / f"frame_{index:04d}.json"
        labels = {
            "index": index,
            "rgb": f"{source_frame}/rgb.png",
            "distance_to_camera_m": f"{source_frame}/distance_to_camera_m.npy",
            "mask": mask_path.name,
            "boxes_xyxy_px": [
                {
                    "class_id": TRAINING_CLASSES[box["class"]],
                    "xyxy": _exclusive_box(box["xyxy_px"]),
                    "visible_pixels": box["visible_pixels"],
                    "median_range_m": box["range_depth_m_median"],
                }
                for box in meta["boxes"]
            ],
            "source_annotation": f"{source_frame}/annotation.json",
            "source_segmentation_id_to_labels": meta["segmentation_id_to_labels"],
        }
        write_json(label_path, labels)
        frames.append({"index": index, "labels": label_path.name, "mask": mask_path.name})
    write_json(destination / "training.json", {
        "format": "isaac-harness-training-v1",
        "classes": TRAINING_CLASSES,
        "non_targets": "BACKGROUND, UNLABELLED, parts_tray and every other class map to background/other (0); they are not detection targets.",
        "mask": "single-channel uint8 PNG; pixel values are stable class IDs",
        "distance_to_camera_m": "source float32 NPY, camera range in metres; non-target pixels may be non-finite",
        "boxes_xyxy_px": "[left, top, right, bottom], integer pixels; right and bottom exclusive",
        "source_boxes_xyxy_px": "native Replicator integer pixels; right and bottom inclusive",
        "source_manifest": f"{relative_source}/manifest.json",
        "source_manifest_sha256": file_hash(directory / "manifest.json"),
        "frames": frames,
    })
    verify_training(destination)
    return destination / "training.json"


def verify_training(destination):
    """Roundtrip-check every exported mask, label and original file reference."""
    destination = destination.resolve()
    training = read(destination / "training.json")
    if training["format"] != "isaac-harness-training-v1" or training["classes"] != TRAINING_CLASSES:
        raise ValueError("invalid training format or class map")
    source_manifest = (destination / training["source_manifest"]).resolve()
    if file_hash(source_manifest) != training["source_manifest_sha256"]:
        raise ValueError("source manifest changed")
    source = source_manifest.parent
    manifest, _ = verify(source)
    if len(training["frames"]) != len(manifest["frames"]) or not training["frames"]:
        raise ValueError("incomplete training frames")
    for expected, entry in zip(manifest["frames"], training["frames"], strict=True):
        index = expected["index"]
        if entry != {"index": index, "labels": f"frame_{index:04d}.json", "mask": f"frame_{index:04d}_mask.png"}:
            raise ValueError(f"frame {index}: invalid training index")
        labels = read(destination / entry["labels"])
        source_frame = source / f"frame_{index:04d}"
        if labels["index"] != index or labels["mask"] != entry["mask"]:
            raise ValueError(f"frame {index}: label index or mask differs")
        for key, filename in (("rgb", "rgb.png"), ("distance_to_camera_m", "distance_to_camera_m.npy"), ("source_annotation", "annotation.json")):
            if (destination / labels[key]).resolve() != source_frame / filename:
                raise ValueError(f"frame {index}: invalid {key} reference")
        if labels["source_segmentation_id_to_labels"] != expected["segmentation_id_to_labels"]:
            raise ValueError(f"frame {index}: changed class provenance")
        expected_boxes = [
            {
                "class_id": TRAINING_CLASSES[box["class"]],
                "xyxy": _exclusive_box(box["xyxy_px"]),
                "visible_pixels": box["visible_pixels"],
                "median_range_m": box["range_depth_m_median"],
            }
            for box in expected["boxes"]
        ]
        if labels["boxes_xyxy_px"] != expected_boxes:
            raise ValueError(f"frame {index}: changed boxes")
        ids = np.load(source_frame / "semantic_ids.npy", allow_pickle=False)
        with Image.open(destination / entry["mask"]) as im:
            if im.mode != "L" or not np.array_equal(np.asarray(im), _class_mask(ids, expected["segmentation_id_to_labels"])):
                raise ValueError(f"frame {index}: changed or invalid class mask")
    return {"status": "pass", "frames": len(training["frames"])}


def gallery(directory, destination):
    """Display saved labels and metric range at the same camera pixels as RGB."""
    from html import escape
    import os
    from PIL import ImageDraw

    manifest, _ = verify(directory)
    destination.mkdir(parents=True, exist_ok=False)
    colors = {"carton": (243, 161, 53), "handheld_scanner": (20, 180, 207)}
    sections = []
    for frame in manifest["frames"]:
        index = frame["index"]
        folder = directory / f"frame_{index:04d}"
        ids = np.load(folder / "semantic_ids.npy", allow_pickle=False)
        distance = np.load(folder / "distance_to_camera_m.npy", allow_pickle=False)
        colored = np.zeros((*ids.shape, 3), dtype=np.uint8)
        with Image.open(folder / "rgb.png") as im:
            annotated = im.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        for box in frame["boxes"]:
            label = box["class"]
            colored[np.isin(ids, box["segmentation_ids"])] = colors[label]
            draw.rectangle(box["xyxy_px"], outline=colors[label], width=2)
            x, y = box["xyxy_px"][:2]
            draw.text((x, y - 14), label, fill=colors[label], stroke_width=1, stroke_fill="black")
        valid = np.isfinite(distance) & (distance > 0)
        depth = np.zeros_like(distance, dtype=np.uint8)
        depth[valid] = np.clip(distance[valid] / 5.0 * 255, 1, 255).astype(np.uint8)
        for suffix, picture in [
            ("boxes", annotated),
            ("classes", Image.fromarray(colored)),
            ("range", Image.fromarray(depth)),
        ]:
            picture.save(destination / f"{index}-{suffix}.png")
        paths = [
            (f"{index}-boxes.png", "RGB と tight bbox"),
            (f"{index}-classes.png", "セマンティック ID"),
            (f"{index}-range.png", "距離 0–5 m"),
        ]
        figures = "".join(
            f'<figure><a href="{p}"><img src="{p}" alt="{title}"></a><figcaption>{title}</figcaption></figure>'
            for p, title in paths
        )
        annotation = escape(os.path.relpath(folder / "annotation.json", destination))
        sections.append(
            f'<section><h2>Frame {index}</h2><div>{figures}</div><a href="{annotation}">座標・クラス・抽選値</a></section>'
        )
    html = """<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>梱包作業台の教師データ</title><style>body{max-width:1600px;margin:2em auto;padding:0 3vw;font:16px system-ui;color:#222}section{padding:1em 0;border-top:1px solid #bbb}section div{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(280px,100%),1fr));gap:1em}figure{margin:0}img{width:100%;display:block}h2{font-size:1.1em}</style><h1>梱包作業台の教師データ</h1><p>橙は段ボール、青は端末。距離画像は明るいほど遠く、黒は無効値。元の float32 配列はメートル単位。</p>"""
    (destination / "index.html").write_text(html + "\n".join(sections) + "</html>\n", encoding="utf-8")
