# TUM Mono-VO camera profile

This optional sensor transform uses the **IDS uEye UI-3241LE-M-GL monochrome global-shutter camera, wide Lensagon lens, TUM Mono-VO sequence 01**. It is an independently published rig. The inspection worktable, YCB assets, and campaign renderer were not captured by this camera. Applying this profile to RTX output is a what-if simulation, not a calibration of the worktable camera.

Source: [TUM dataset and CC BY 4.0 license](https://cvg.cit.tum.de/data/datasets/mono-dataset), [original paper](https://cvg.cit.tum.de/_media/spezial/bib/engel2016monodataset.pdf). Files in `examples/sensors/tum_mono_vo_sequence_01/` were extracted by HTTP Range from `all_sequences.zip/sequence_01/`. `camera.txt` contains normalized FOV intrinsics, `pcalib.txt` has the measured inverse response U for 8-bit monochrome code values, `vignette.png` is the dense measured lens attenuation, and `times.txt` records per-frame reported exposure. Real frame `00000.jpg` is included for provenance. TUM states the camera is 1280 × 1024 and sequence images are JPEG-compressed.

The transform is: output raw-sensor coordinates → inverse FOV distortion → ideal pinhole source ray → bilinear sample of linearized source RGB luminance. The radiance is multiplied by exposure ratio and the measured vignette, then the measured inverse response LUT is inverted to 8-bit sensor code. RGB has three identical monochrome channels. Class and instance IDs and range use the same source coordinates with nearest sampling; invalid rays yield RGB and ID zero and range NaN. The caller supplies the RTX source intrinsic matrix `K`; native 1280 × 1024 input may omit it. The model does not undo RTX tone mapping, infer actual irradiance, model JPEG artifacts, rolling shutter, or Bayer color. The sequence 01 global shutter camera has no Bayer color output in this dataset.

`noise_sigma_linear` is **off by default**. A supplied positive value is an explicit assumption recorded with its seed; TUM does not publish a sequence 01 noise profile in these files. Three repeated `calib_wide_whitePaper` images at equal reported exposure show a temporal residual of **0.99–1.07 8-bit codes** in two flat paper regions (`scripts/fit_sensor_profile.py`, `evidence/sensor-calibration/noise_validation.json`). This includes JPEG compression and any residual paper or camera motion. That calibration run had hardware gamma disabled, whereas sequence 01 uses wide-gamma response, so this residual is not applied as sequence 01 noise. The sequence exposure value `3.6060349941` is used as both example exposure and reference; exposure ratio is therefore 1. The normalization of renderer brightness to physical irradiance is assumed. Saturation and absolute photometry cannot be validated against the unrelated RTX scene.

Example on a saved RTX image:

```sh
PYTHONPATH=. python scripts/prepare_sensor_profile.py evidence/inspection/nominal/overview.png --output evidence/sensor-calibration/rtx-overview
```

On a campaign frame containing `rgb.png`, `classes.png`, `instances.png`, `range_m.npz`, and `annotation.json` with `sample.camera.K`:

```sh
PYTHONPATH=. python scripts/prepare_sensor_profile.py /path/to/frames/00000000 --output /path/to/tum-mono/00000000
```

The campaign output has registered `rgb.png`, `classes.png`, `instances.png`, `range_m.npz`, and `annotation.json` with source provenance and the source K. A standalone PNG is resized for a **visual demonstration** only; there are no true semantic or range labels to export. The target 1280 × 1024 field of view is wider than the inspection RTX overview, so out-of-source rays are black. The first saved example has 50.7% valid source rays, visible in `evidence/sensor-calibration/rtx-overview/rgb.png`.

Validation: `PYTHONPATH=. pytest -q tests/test_sensor_model.py` checks official file shape/range, an off-axis geometric ray, registered label/range warping, seed determinism, and invalid exposure. A separate `all_calib_sequences.zip/calib_wideGamma_scene1` download contains byte-identical `camera.txt` and `pcalib.txt` to sequence 01; SHA-256 and real frame statistics are in `evidence/sensor-calibration/validation.json`. This cross-checks that the selected calibration is for the same published wide-gamma rig. It does not establish that the worktable has the TUM camera or that noise or absolute irradiance is calibrated. Calibration scene frames under `validation/` are real TUM data.
