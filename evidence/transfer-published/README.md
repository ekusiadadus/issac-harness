# Published synthetic to real detector benchmark

## Scope and sources

This is a deliberately small, **measured** detector baseline on BOP YCB-V. It is not an official BOP challenge submission, an Isaac Sim result, or an industrial transfer result. The source dataset is [BOP YCB-V](https://bop.felk.cvut.cz/datasets/#YCB-V), mirrored by [BOP on Hugging Face](https://huggingface.co/datasets/bop-benchmark/ycbv) under MIT; the exact dataset notice is copied in [ycbv_dataset_info.md](ycbv_dataset_info.md). BOP's `train_pbr` images were rendered with BlenderProc4BOP. The held-out `test` images are real YCB-Video camera images. The BOP page warns against the original YCB-Video synthetic 80K images because those repeat test poses; this run used only BOP `train_pbr`.

The three labels use **BOP IDs**: 2 = `003_cracker_box`, 3 = `004_sugar_box`, 5 = `006_mustard_bottle`. The original YCB numeric prefix differs from the BOP ID. The base archive's `dataset_info.md` confirms this and says BOP meshes are in millimeters, recentered at the 3D bounding-box center, with ground-truth poses adjusted accordingly.

| Archive | URL | SHA-256 |
| --- | --- | --- |
| BOP BlenderProc synthetic | [ycbv_train_pbr.zip](https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_train_pbr.zip) | `cd64d9d343520af1334fd44729230207099a2e528c99b64851ef2d0ce0844770` (Hugging Face linked etag) |
| BOP real test | [ycbv_test_bop19.zip](https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_test_bop19.zip) | `c0c11251849877e7b2f373f6c7acf54739dc69d0fb2649c20050ca977bf5513d` (locally verified) |
| BOP base, model files | [base](https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_base.zip), [models](https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_models.zip) | See source archive |

Selected member names and SHA-256 hashes are in `train_frames.txt`, `test_frames.txt`, and the `*_member_sha256.json` files. These files specify exact frames, annotations, and camera metadata; no full synthetic archive was downloaded. HTTP range requests read 256.6 MB of the 20.98 GB synthetic archive, including its 215 MB central directory. The local extracted files used under 200 MB.

## Protocol

- Synthetic candidate scenes: 0, 10, 20, 30, 40. From each, sort image IDs numerically and take every 50th frame. Keep frames containing at least one of the three labels: 98 frames. Scenes 0, 10, 20, 30 (78 frames) fit the detector. Scene 40 (20 frames) is a synthetic holdout.
- Real test scenes: 49, 52, 53, 54, 55, 56, 58, 59. From each 75-frame BOP subset, sort image IDs numerically and take every 7th frame, capped at 11: 88 frames total. There are 22 cracker boxes, 44 sugar boxes, 11 mustard bottles. Scenes 53 and 56 contain **none** of the target labels. Scenes 50 and 51 are reserved for paired calibration examples and excluded from detector testing.
- The detector is a class-wise, from-scratch HOG plus six HSV color statistics and balanced linear SVM. It fits only synthetic visible boxes with at least 24 pixels per side and at least 0.25 visible fraction, plus synthetic background windows. A synthetic-only median box aspect ratio is used per class. Five fixed window heights (64, 96, 128, 192, 256 pixels), 16-pixel search stride in normalized coordinates, class-wise NMS IoU 0.4, and top 30 boxes per class and image were fixed in code. No real image, real label, or real score was used for fitting or choosing settings.
- Evaluation uses `bbox_visib` (visible object rectangle), one-to-one greedy IoU matching, and 101-point interpolated AP at IoU 0.50 and mean IoU 0.50:0.05:0.95. Class macro means exclude absent classes within a scene. Recall below is the maximum recall over the retained top 30 boxes, *without* a confidence cutoff. Scores are raw SVM margins, not calibrated probabilities.

| Split | Images | Instances ID 2 / 3 / 5 | Macro AP50 | Macro AP50:95 | Macro recall50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic holdout scene 40 | 20 | 16 / 14 / 19 | 0.001256 | 0.000275 | 0.133 |
| Independent real scenes | 88 | 22 / 44 / 11 | **0.001985** | **0.000334** | **0.197** |

Real AP50 by class: ID 2 `0.002889`, ID 3 `0.003067`, ID 5 `0`. Per-scene results are in [metrics.json](metrics.json); all predictions and scores are in [predictions.json](predictions.json). The synthetic holdout is also very poor. Therefore this run establishes a reproducible, weak detector floor and **does not isolate a synthetic-to-real gap**. It should not be used to support deployment accuracy. On the 22 wholly target-negative images, 1,605 retained boxes have SVM margin at least zero (810 in scene 53, 795 in scene 56); the [overlays](overlays) visibly show many false positives.

## Calibration pairs and visual review

`calibration/poses.json` records real RGB, per-frame intrinsics, visible masks, boxes, and 6D camera-to-object poses for two **non-test** frames: scene 50 frame 000620 (IDs 2 and 5) and scene 51 frame 000001 (ID 3). The sampled intrinsics are `fx=1066.778, fy=1067.487, cx=312.9869, cy=241.3109` at 640×480. Translation is in millimeters. The corresponding copied RGB and mask files have member hashes in the JSON. They are ground-truth calibration references; no predicted poses were produced. See [review.md](review.md) for visual inspection of test overlays.

## Reproduce

With the repository `.venv` and network access:

```sh
uv pip install --python .venv/bin/python -r benchmarks/transfer/requirements.txt
.venv/bin/python benchmarks/transfer/prepare_bop.py train --root /tmp/issac-transfer-bench/data
.venv/bin/python benchmarks/transfer/prepare_bop.py test --root /tmp/issac-transfer-bench/data
.venv/bin/python benchmarks/transfer/detector.py --train-index /tmp/issac-transfer-bench/data/train.jsonl --test-index /tmp/issac-transfer-bench/data/test.jsonl --classes 2 3 5 --exclude-train-scene 40 --output evidence/transfer-published
.venv/bin/python benchmarks/transfer/render_overlays.py --test-index /tmp/issac-transfer-bench/data/test.jsonl --predictions evidence/transfer-published/predictions.json --output evidence/transfer-published/overlays
.venv/bin/python -m pytest tests/test_transfer_detector.py -q
```

`prepare_bop.py test` can instead receive `--local-archive PATH` to use an already downloaded real ZIP. Python 3.12, NumPy 2.5.5, OpenCV 4.14.0, and scikit-learn 1.9.1 were used here. The small `model.pkl` is a Python pickle and should be loaded only from trusted local artifacts.

For the published synthetic holdout, pass `train.jsonl` as **both** indexes with `--exclude-train-scene 40 --only-test-scene 40` and a distinct output directory. The image-byte audit then verifies that the fitted and evaluated RGB files are disjoint even though both subsets came from one raw index.

### Generic index and native comparison

For an Isaac-native synthetic training run against the **same** real split, write a JSONL index alongside your images. Each record has a stable `id` (or `archive_member`), `image` (absolute, or relative to the JSONL), and `objects`: `[{"category_id":2,"bbox_xywh":[x,y,w,h]}, ...]`. `scene` or `group` is optional and controls the per-group metric table. If `id` and `archive_member` are absent, the image path as written in the index becomes the stable ID. Category IDs can be any integers; choose them with `--classes`. Other IDs in `objects` become background examples. No scene is silently excluded in a generic run.

```sh
.venv/bin/python benchmarks/transfer/detector.py --train-index /path/native-synthetic.jsonl --test-index /tmp/issac-transfer-bench/data/test.jsonl --classes 2 3 5 --output /tmp/native-transfer-eval
```

Use the **same** real `test.jsonl` and class IDs for a fair native-versus-published comparison. The published baseline must pass `--exclude-train-scene 40` to reproduce the 78-frame fit; that scene is not special to a native index. Native synthetic versus published BlenderProc is a separate comparison; this published result is not itself an Isaac render.

Before fitting, the command writes `input_audit.json` with raw train/test index SHA-256, selected image counts, per-class instance counts and example IDs, SHA-256 of each actual RGB file, and a byte-identical cross-split overlap count. It raises an error on any identical RGB bytes. Use `--audit-only` to run this check without training. The published inputs yielded train index SHA-256 `64579f939053585a37c563c1b63f47fb86662a021237c9f053d64119930acec4` (98 indexed, 78 fitted), real test index SHA-256 `dbadaa6c9145d1666355e31b829007b7a3e57963e74dd49778c0fe58d583d5c3` (88 evaluated), and **zero** identical fitted-train/real-test RGB files. The full [input audit](input_audit.json) records individual hashes and examples.
