# Registered real and simulated image quality

Run the camera-registered gate after the native campaign `check` has passed:

```sh
python -m harness.paired /path/to/capture \
  --references examples/campaign/ycb/real \
  --config examples/campaign/ycb/quality.toml \
  --out /path/to/quality
```

The command exits 0 only if **every configured scene** has rendered frames and passes all three preset limits. It writes `quality.json` with per-frame, per-class and per-scene metrics, failure reasons and the campaign key. `strips/` contains real RGB, simulated RGB and a simulated RGB overlay, left to right. Red traces real target contours, cyan traces simulated contours, and yellow shows coincident contours. Open the PNGs at original resolution for visual review. A missing reference, unsupported class, corrupt campaign frame, absent valid overlap or absent configured scene fails the gate. The JSON records all failed frames, even when valid frames pass.

Each simulated frame's `annotation.json` supplies the `sample.reference`, `sample.depth_scale_mm` and the effective `readback.camera.K`. `campaign.json` supplies resolution, class ontology and source provenance. The real reference directory contains `rgb.png`, visible `classes.png` and optical Z `z-depth.png`; the captured frame contains `rgb.png`, `classes.png`, `instances.png` and `range_m.npz`. The scorer invokes `harness.campaign.inspect_frame` first, including content hashes. It records SHA-256 for all three real planes per frame and verifies `sample.rgb_sha256` when present. Keep the real reference directory and threshold file with the report so the result can be reproduced.

The comparison assumes the real and simulated pixels are registered, have the same resolution, share the class IDs, and use the same undistorted pinhole camera. It does **not** search for a better alignment. Native `distance_to_camera` is Euclidean radial range. Per pixel, the scorer converts it to optical Z using the inverse effective K and pixel centre `(x+0.5, y+0.5)`. Real integer depth is multiplied by the recorded millimetres-per-unit scale and divided by 1000. Invalid/nonpositive depths are excluded. Z error is the median absolute error over valid pixels lying inside both target masks after a two-pixel erosion. RGB error is the mean absolute **sRGB encoded** channel difference divided by 255 over the same matched interiors. IoU is computed on full visible class masks; macro IoU averages each class that appears in either image, then each class/frame sample in the scene. Missing detections therefore score IoU 0. Background and class 0 are excluded.

The YCB preset requires scene macro IoU ≥ 0.80, pooled median optical Z error ≤ 0.015 m, and pooled sRGB MAE ≤ 0.10 for **each** recorded scene. These are application gates chosen before capture, not claims that the renderer meets them. RGB MAE is sensitive to exposure, white balance, tone mapping, lens distortion, occlusion and the unknown source lighting. It is not a perceptual image score, linear-light radiometric comparison, or proof that the source geometry/material/physics is correct. Assess the strips and image conditions before claiming appearance agreement; keep failing results visible rather than tuning a threshold to a capture.

For another registered dataset, prepare the same native campaign format and real directory layout, give each sample a `reference` and `depth_scale_mm`, and write a TOML file with `[scenes.SCENE_NAME]` and the three required thresholds. The scene names must match the sample `layout` fields. No YCB-specific class names or image paths are embedded in the scorer.

## Freeze one exposure bracket before held-out evaluation

Capture a **native RTX** bracket with multiple real reference views and the same observed scalar light setting for each view. The recorded sample must contain `sample.reference` and `sample.edits.light_gain` (or another named numeric field); no image scaling is performed by the fitter. Give only calibration reference IDs to the selection command:

```sh
python -m harness.exposure_fit /path/to/bracket-capture \
  --references examples/campaign/ycb/real \
  --calibration-ref 000050_000620 \
  --calibration-ref 000051_000001 \
  --gain-field edits.light_gain \
  --out /path/to/exposure-freeze.json
```

The fitter checks each native frame's campaign hash and planes, then uses the same two-pixel interior rule and normalized sRGB channel difference as the registered scorer. It requires exactly one shot for each named reference at **every** observed scalar value. The chosen scalar minimizes the mean of the per-reference MAEs, giving each calibration view equal weight. Ties choose the lower numeric scalar. The freeze file records every observed candidate, pooled and per-reference MAE, matched channel counts, simulated and real clipping fractions (channels ≥250), campaign key, source hashes and the chosen scalar. It does not estimate per-channel gains, interpolate between captures, include unlisted held-out references, or alter the predeclared IoU/Z/RGB acceptance thresholds. Use the frozen scalar for a separate capture of unseen physical real-image indices, then apply the registered quality gate to that capture. Source lighting and camera response remain limits on the photometric interpretation.
