# Synthetic-only SIFT detector: published BOP YCB-V to real YCB-V

This is a second, stronger detector benchmark using the **same exact** [BOP YCB-V](https://bop.felk.cvut.cz/datasets/#YCB-V) frame lists and class IDs as the [HOG floor](../transfer-published/README.md). BOP IDs are 2 cracker box, 3 sugar box, 5 mustard bottle. The dataset's MIT notice is in [dataset_info.md](../transfer-published/ycbv_dataset_info.md). This is published BlenderProc synthetic training to real YCB-V evaluation. It is not an Isaac render, industrial validation, or an official BOP challenge score.

## Method and freeze

OpenCV SIFT descriptors are computed from visible synthetic object pixels using BOP `mask_visib` and box annotations. For each class and synthetic scene, the five crops with the most visibility-weighted keypoints are retained (20 per class from four fitting scenes). In each evaluation image, SIFT ratio matches and RANSAC homography map a template crop to a candidate visible box. Fixed rules use ratio 0.72, at least seven matches, six homography inliers, five-pixel RANSAC tolerance, and IoU 0.4 nonmaximum suppression. No learned pretrained weights, downloaded model, real RGB, real labels, or real threshold tuning were used for the model. The serialized model contains only synthetic descriptors and points.

The source, model, and synthetic holdout metrics were hashed in [frozen_before_real.json](frozen_before_real.json) at `2026-11-25T00:34:33.814036Z`, **before** the first real evaluation. The exact then-running source is preserved as [frozen_sift_detector.py](frozen_sift_detector.py), SHA-256 `cbe269ee05707fa1d92fd96f5eb8e566aad5c51140266e137c6f972ded3691e2`. Later changes in the normal script only added generic native `instance_mask` and group handling. Frozen model SHA-256 is `9b5f8f2b749f920276f6e0d7751c491cdba1ac92c6d1cd744fe0534f80c5855b`. [model_manifest.json](model_manifest.json) lists every synthetic template source ID and the [109 mask member hashes](template_mask_sha256.json).

| Split | Images | Target instances 2 / 3 / 5 | Macro AP50 | Macro AP50:95 | Macro recall50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent synthetic scene 40 | 20 | 16 / 14 / 19 | **0.21296** | **0.08660** | **0.26723** |
| Independent real scenes 49, 52, 53, 54, 55, 56, 58, 59 | 88 | 22 / 44 / 11 | **0.22063** | **0.04583** | **0.33333** |

Real AP50 by class: cracker box **0.36462**, sugar box **0.29729**, mustard bottle **0.0**, with zero mustard predictions. The real per-scene variation is large: cracker box AP50 is 0 in scene 54 and 0.902 in scene 59. Scenes 53 and 56 have no target objects, yet contain 9 and 3 false cracker-box detections across 11 frames each. The [real metrics](real/metrics.json), [synthetic metrics](synthetic_holdout/metrics.json), predictions, and RGB-byte [audits](real/input_audit.json) give full counts and scores. Both audits report **zero byte-identical fitted-train/evaluation RGB files**. The train index SHA-256 is `64579f939053585a37c563c1b63f47fb86662a021237c9f053d64119930acec4` (98 listed; 78 fitted after excluding scene 40). The real test index SHA-256 is `dbadaa6c9145d1666355e31b829007b7a3e57963e74dd49778c0fe58d583d5c3` (88 tested).

The synthetic and real AP50 means are similar, but AP50:95 is lower on real images. These are small, uneven subsets, so this is a measured transfer result for these three objects and eight scenes, not a general domain-gap estimate. [Visual review](review.md) records representative successes and failures.

## Reproduce and native hook

Environment: Apple arm64 Mac, 8 GiB RAM, Python 3.12.12, OpenCV 4.14.0, NumPy 2.5.5, scikit-learn 1.9.1 (the SIFT route uses OpenCV and NumPy; the shared evaluator imports scikit-learn). No new large package was installed. CPU time after masked input preparation: **1.82 s training**, **3.51 s synthetic evaluation**, **13.64 s real evaluation**. Raw timing and extraction logs are under `/tmp/issac-transfer-bench/`.

```sh
.venv/bin/python benchmarks/transfer/prepare_bop.py train --root /tmp/issac-transfer-bench/data
.venv/bin/python benchmarks/transfer/prepare_template_masks.py
.venv/bin/python benchmarks/transfer/prepare_bop.py test --root /tmp/issac-transfer-bench/data
.venv/bin/python benchmarks/transfer/sift_detector.py train --train-index /tmp/issac-transfer-bench/data/train.jsonl --classes 2 3 5 --exclude-train-scene 40 --output evidence/transfer-model
.venv/bin/python benchmarks/transfer/sift_detector.py evaluate --model evidence/transfer-model/model.pkl --train-index /tmp/issac-transfer-bench/data/train.jsonl --test-index /tmp/issac-transfer-bench/data/train.jsonl --exclude-train-scene 40 --only-test-scene 40 --output evidence/transfer-model/synthetic_holdout
.venv/bin/python benchmarks/transfer/sift_detector.py evaluate --model evidence/transfer-model/model.pkl --train-index /tmp/issac-transfer-bench/data/train.jsonl --test-index /tmp/issac-transfer-bench/data/test.jsonl --exclude-train-scene 40 --output evidence/transfer-model/real
```

For native synthetic frames, each JSONL row supports `{"id":"stable-frame","image":"relative/rgb.png","instance_mask":"relative/ids.png","group":"layout_A","objects":[{"category_id":2,"instance_id":257,"bbox_xywh":[x,y,w,h]}]}`. `instance_mask` is loaded unchanged as an integer PNG, so 16-bit instance ID 257 remains distinct from 1; object descriptors use only pixels equal to that ID. If `instance_mask` is omitted, a per-object `mask` or the crop rectangle is used. Category IDs are selected with `--classes`; `group` determines per-group template selection and evaluation.

To reserve one native layout, train with `--exclude-train-group layout_A`. Evaluate synthetic holdout with the same JSONL as both indexes plus `--exclude-train-group layout_A --only-test-group layout_A`. Then evaluate real with the same frozen native model and `--exclude-train-group layout_A --test-index /tmp/issac-transfer-bench/data/test.jsonl`. Each output receives an RGB hash audit and rejects exact train/test image leakage. Use distinct output directories for native results. The current entry point is `benchmarks/transfer/sift_detector.py`; the archived frozen source only documents the published run.
