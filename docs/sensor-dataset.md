# Calibrated sensor derivatives

`harness.sensor_dataset` remaps a **verified native campaign** into a separate,
sealed camera derivative. It uses each native frame's effective camera `K`
(`readback.camera.K` when present, otherwise `sample.camera.K`) and the measured
target profile's `K`, Brown–Conrady `D`, width and height. The result is derived
data, **not** a capture emitted by an Isaac camera sensor.

```sh
python -m harness campaign sensor /path/to/native \
  --profile /path/to/calibrated-camera.json \
  --out /path/to/sensor-derivative --seed 20261125

# Optional deterministic subset and measured noise fit:
python -m harness campaign sensor /path/to/native \
  --profile /path/to/calibrated-camera.json \
  --out /path/to/sensor-derivative --seed 20261125 \
  --start 128 --count 128 --noise-fit /path/to/noise-fit.json

python -m harness campaign check /path/to/sensor-derivative \
  --out /path/to/sensor-export --start 0 --count 256 --coco
```

The target profile follows `harness.color_sensor`'s JSON schema: `name`,
`provenance`, `width`, `height`, a row-major 3×3 `K`, and `D` in OpenCV order
`[k1, k2, p1, p2, k3]`. The distortion model is `plumb_bob`. A ROS CameraInfo
style `camera_matrix.data` and `distortion_coefficients.data` also work.
Use a profile measured for the same camera mode and resolution as the intended
real images. The optional noise fit is generated from matched flat exposures
with `python -m harness.color_sensor fit-noise`.

Each output frame retains RGB, class and instance masks, radial range and a
binary `valid.png`. RGB uses bilinear sampling; labels and radial range use
nearest sampling. Out-of-view pixels are black/background with NaN range.
Visible instance rectangles and areas are recomputed from the **warped** masks;
an instance cropped completely out of view has no detection annotation. The
`sensor_derivative` record includes the source campaign key, source annotation
hash and K, target K and D, calibration profile hash and provenance, noise fit
hash, per-index seed, valid mask hash and valid fraction. The campaign header
has `kind: sensor-derived`, its own sealed key, and the source campaign key.
The four standard planes retain the native campaign hash contract; the fifth
plane is checked by `validate_valid_mask` (also called by `inspect_frame` for
sensor derivatives). A rerun with the same inputs validates existing frames
and adds missing ones. A different source, calibration, noise fit or seed
requires a new output directory.

`campaign check`, gallery and transfer can consume this derivative using the
standard frame layout. The `campaign paired` scorer takes undistorted, registered views. Use the original native campaign and matching real pinhole images for that score. The training export retains the measured distortion, the source view family, and the full valid field of view mask.
