# Random-initialized SSDLite transfer baseline

`benchmarks/transfer/neural_detector.py` fits TorchVision's SSDLite320 MobileNetV3 Large with `weights=None` and `weights_backbone=None`. The model uses a contiguous foreground label for each supplied category ID, with label 0 reserved for background. Images are loaded as RGB and scaled to `[0, 1]`; TorchVision applies its model transform and 320-pixel resize. Boxes are clipped to image bounds. Images with no selected objects remain in training as negative scenes.

Use matching PyTorch and TorchVision builds with Pillow and NumPy. The pinned GCP Lab image has PyTorch 2.11.0+cu130 and TorchVision 0.26.0+cu130. The benchmark uses one CUDA L4. CPU execution can also be selected. No download or pretrained checkpoint is used.

```sh
python benchmarks/transfer/neural_detector.py train \
  --train-index /data/synthetic/train.jsonl --classes 2 3 5 \
  --exclude-train-group cross_pick --output out/ssdlite
python benchmarks/transfer/neural_detector.py evaluate \
  --model out/ssdlite/model.pt --train-index /data/synthetic/train.jsonl \
  --test-index /data/real/test.jsonl --output out/ssdlite-real
```

The fit defaults are 24 epochs, batch size 8, AdamW at 0.001, and seed 42. `train` writes `model.pt` (state dict only) and `model_manifest.json` with the class mapping, settings, input index hash, fit image count, elapsed time, per-epoch losses, versions, and checkpoint SHA256. `evaluate` checks the manifest and train index, computes the shared byte-level train/test audit, and writes `input_audit.json`, `predictions.json`, and `metrics.json`. The default excluded training group is `cross_pick`. `--only-test-group` and `--only-test-scene` select evaluation subsets. Keep each output directory and the original JSONL/image files to reproduce the result. Metrics describe the selected test index only; the GPU run and real-image performance must be reported separately.
